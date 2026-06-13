"""Run segmentation training on a GPU via Modal.

The training code itself lives in :mod:`oilspill.training`; this module only wires
it to remote GPU hardware. The dataset is uploaded once to a persistent Modal
Volume (see ``upload_data``); each run mounts that volume read-only, trains, and
writes the best/last checkpoints plus a metrics JSON back to an artifacts volume.
The local entrypoint also returns the best checkpoint bytes so they land directly
in the local ``artifacts/checkpoints/`` tree for evaluation.

Usage (from the repo root, with the ``gpu`` dependency group installed)::

    # one-time: push the extracted dataset to the data volume
    uv run modal run scripts/modal_train.py::upload_data

    # train using a config from configs/
    uv run modal run scripts/modal_train.py --config configs/unet.yaml --gpu L4

The dataset must already be extracted locally at ``data/datasets/oil_spill``
(run ``make data`` first).
"""

from __future__ import annotations

import json
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
def train_remote(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Train on the GPU and return metrics + the best/last checkpoint bytes."""
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


@app.local_entrypoint()
def main(config: str = "configs/unet.yaml", gpu: str = "L4") -> None:
    """Load a local YAML config, run training remotely, save outputs locally."""
    import yaml

    config_path = Path(config)
    config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    print(f"launching GPU training on Modal ({gpu}) with {config_path}")
    result = train_remote.remote(config_dict)

    run_name = result["run_name"]
    out_dir = Path("artifacts/checkpoints") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if result["best_checkpoint"]:
        (out_dir / "best.pt").write_bytes(result["best_checkpoint"])
    if result["last_checkpoint"]:
        (out_dir / "last.pt").write_bytes(result["last_checkpoint"])
    (out_dir / "metrics.json").write_text(result["metrics_json"], encoding="utf-8")

    print(f"\n=== training complete (run {run_name}) ===")
    print(f"best metric: {result['best_metric']:.4f}")
    for key, value in sorted(result["metrics"].items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"checkpoints saved to {out_dir}")


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
