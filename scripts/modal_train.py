"""Run segmentation training on a GPU via Modal.

The training code itself lives in :mod:`oilspill.training`; this module only wires
it to remote GPU hardware. The dataset is uploaded once to a persistent Modal
Volume (see ``upload_data``); each run mounts that volume, trains, and writes the
best/last checkpoints plus a metrics JSON to the artifacts volume. Training is
*spawned* (fire-and-forget) so it runs to completion server-side even if the
local client disconnects; results are collected afterwards with ``fetch``.

Usage (from the repo root, with the ``gpu`` dependency group installed)::

    # one-time: push the extracted dataset to the data volume
    uv run modal run scripts/modal_train.py::upload_data

    # spawn training, then collect results when it finishes
    uv run modal run scripts/modal_train.py::main --config configs/unet.yaml --gpu L4
    uv run modal run scripts/modal_train.py::fetch --run-name <run_name printed above>

The dataset must already be extracted locally at ``data/datasets/oil_spill``
(run ``make data`` first).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "oilspill-train"
DATA_VOLUME = "oilspill-data"
ARTIFACTS_VOLUME = "oilspill-artifacts"
DATA_MOUNT = "/data"
ARTIFACTS_MOUNT = "/artifacts"
REMOTE_DATASET_DIR = f"{DATA_MOUNT}/oil_spill"

# Training-only dependencies. Deliberately excludes the project's geospatial and
# serving stack (rasterio, fastapi, ...) which training does not need. On Linux
# the default PyPI torch wheels are CUDA-enabled, so no special index is required.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.9.0",
        "torchvision",
        "segmentation-models-pytorch",
        "transformers",
        "torchmetrics",
        "albumentations",
        "mlflow",
        "numpy",
        "pillow",
        "pyyaml",
        "tqdm",
        "pydantic",
        "pydantic-settings",
        "opencv-python-headless",
    )
    .add_local_python_source("oilspill")
)

app = modal.App(APP_NAME, image=image)

data_volume = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME, create_if_missing=True)


@app.function(volumes={DATA_MOUNT: data_volume}, timeout=3600)
def _list_data() -> list[str]:
    """Return a short listing of the data volume (sanity check after upload)."""
    root = Path(REMOTE_DATASET_DIR)
    if not root.exists():
        return []
    out: list[str] = []
    for split in ("train", "test"):
        img_dir = root / split / "images"
        n = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        out.append(f"{split}/images: {n}")
    return out


@app.function(
    gpu="L4",
    volumes={DATA_MOUNT: data_volume, ARTIFACTS_MOUNT: artifacts_volume},
    timeout=4 * 3600,
)
def train_remote(config_dict: dict[str, Any], run_name: str | None = None) -> dict[str, Any]:
    """Train on the GPU and return metrics + the best/last checkpoint bytes.

    ``run_name`` fixes the output directory name (``checkpoints/<run_name>/``) so a
    caller that spawned the job (and may have disconnected) knows exactly where to
    fetch the results from the artifacts volume afterwards.
    """
    import torch

    from oilspill.training.config import TrainConfig
    from oilspill.training.trainer import fit

    # Point the config at the mounted volumes and the GPU regardless of what the
    # YAML said locally.
    config_dict = dict(config_dict)
    config_dict.setdefault("data", {})
    config_dict["data"] = {**config_dict["data"], "root": REMOTE_DATASET_DIR}
    config_dict["checkpoint"] = {
        **config_dict.get("checkpoint", {}),
        "dir": f"{ARTIFACTS_MOUNT}/checkpoints",
    }
    config_dict["runtime"] = {
        **config_dict.get("runtime", {}),
        "device": "cuda",
        "amp": True,
        "precision": "bf16",
    }
    # Keep MLflow on fast container-local disk. The MLflow file store reads back
    # run directories immediately after writing them, which races against a Modal
    # Volume's eventual consistency ("Run not found"). The deliverables (checkpoint
    # + metrics.json) are persisted to the artifacts volume and returned instead.
    config_dict["mlflow"] = {
        **config_dict.get("mlflow", {}),
        "tracking_uri": "/root/mlruns",
    }
    if run_name is not None:
        config_dict["mlflow"]["run_name"] = run_name

    cfg = TrainConfig.model_validate(config_dict)
    print(f"CUDA available: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0)}")

    result = fit(cfg)

    # Persist metrics next to the checkpoints on the volume so the run is fully
    # recoverable via `modal volume get` even if the client has disconnected.
    metrics_json = json.dumps(result.metrics, indent=2)
    (result.best_checkpoint.parent / "metrics.json").write_text(metrics_json, encoding="utf-8")
    artifacts_volume.commit()

    def _read(path: Path) -> bytes:
        return path.read_bytes() if path.exists() else b""

    return {
        "metrics": result.metrics,
        "best_metric": result.best_metric,
        "run_name": result.best_checkpoint.parent.name,
        "mlflow_run_id": result.mlflow_run_id,
        "best_checkpoint": _read(result.best_checkpoint),
        "last_checkpoint": _read(result.last_checkpoint),
        "metrics_json": metrics_json,
    }


def _download_run(run_name: str) -> Path:
    """Download a run's checkpoints + metrics from the artifacts volume to local."""
    out_dir = Path("artifacts/checkpoints") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"checkpoints/{run_name}"
    for fname in ("best.pt", "last.pt", "metrics.json"):
        try:
            data = b"".join(artifacts_volume.read_file(f"{remote_dir}/{fname}"))
        except (FileNotFoundError, KeyError):
            continue
        (out_dir / fname).write_bytes(data)
    return out_dir


@app.local_entrypoint()
def main(config: str = "configs/unet.yaml", gpu: str = "L4", run_name: str = "") -> None:
    """Load a YAML config and SPAWN training on Modal (disconnect-proof).

    The job runs to completion server-side regardless of the client connection
    (a plain ``.remote()`` call can be cancelled if the local caller drops, which
    corrupts a checkpoint mid-write). Outputs land on the artifacts volume under
    ``checkpoints/<run_name>/``; collect them afterwards with the ``fetch``
    entrypoint. Writing ``metrics.json`` is the last step, so its presence on the
    volume signals the run finished cleanly.
    """
    import yaml

    config_path = Path(config)
    config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not run_name:
        run_name = datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")

    call = train_remote.spawn(config_dict, run_name=run_name)
    print(f"spawned GPU training on Modal ({gpu}) with {config_path}")
    print(f"run_name: {run_name}")
    print(f"call_id : {call.object_id}")
    print(f"collect when done with: modal run scripts/modal_train.py::fetch --run-name {run_name}")


@app.local_entrypoint()
def fetch(run_name: str) -> None:
    """Download a finished run's checkpoints + metrics from the artifacts volume."""
    artifacts_volume.reload()
    out_dir = _download_run(run_name)
    metrics_path = out_dir / "metrics.json"
    best_path = out_dir / "best.pt"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"=== run {run_name} complete ===")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    elif best_path.exists():
        print(
            f"=== run {run_name}: checkpoints present but metrics.json missing (still running?) ==="
        )
    else:
        print(f"=== run {run_name}: nothing on the volume yet ===")
    print(f"downloaded to {out_dir}")


@app.local_entrypoint()
def upload_data() -> None:
    """Upload the locally-extracted dataset to the Modal data volume (one-time)."""
    local_root = Path("data/datasets/oil_spill")
    if not local_root.exists():
        raise SystemExit(f"{local_root} not found; run `make data` first")

    print(f"uploading {local_root} -> volume '{DATA_VOLUME}':/oil_spill (this can take a while)")
    with data_volume.batch_upload(force=True) as batch:
        batch.put_directory(str(local_root), "/oil_spill")
    print("upload complete; verifying...")
    for line in _list_data.remote():
        print(" ", line)
