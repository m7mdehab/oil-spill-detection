"""Tests for ONNX export, parity, the model card, and the HF publish dry-run.

Fast tests use a tiny convolutional model and a synthetic metrics JSON so they
exercise the real export / card / dry-run code paths without building the heavy
``smp`` model or touching the network. A ``slow`` test exports the real newest
checkpoint and checks parity end to end.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from oilspill.metrics import CLASS_NAMES, NUM_CLASSES
from oilspill.packaging.model_card import build_model_card
from oilspill.packaging.onnx_export import INPUT_NAME, OUTPUT_NAME

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str) -> ModuleType:
    """Import a module from the repo's ``scripts/`` dir by file path.

    ``scripts/`` is not an installed package, so import it directly rather than
    relying on it being on ``sys.path``.
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TinySegModel(nn.Module):
    """Fully-convolutional 3->NUM_CLASSES model; stands in for the real seg net."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, NUM_CLASSES, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _export_tiny(out_path: Path) -> _TinySegModel:
    """Export a tiny model with the same export settings as the real path."""
    model = _TinySegModel().eval()
    dummy = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(out_path),
            dynamo=False,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                INPUT_NAME: {0: "batch", 2: "height", 3: "width"},
                OUTPUT_NAME: {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=17,
        )
    return model


def _synthetic_run_json(path: Path) -> Path:
    """Write a synthetic metrics JSON in the eval-harness schema."""

    def per_class(values: list[float | None]) -> dict[str, float | None]:
        return dict(zip(CLASS_NAMES, values, strict=True))

    data = {
        "meta": {
            "run_name": "synthetic",
            "tag": "test",
            "split": "test",
            "image_size": [256, 256],
            "num_images": 110,
            "device": "cpu",
        },
        "metrics": {
            "class_names": list(CLASS_NAMES),
            "per_class": {
                "iou": per_class([0.95, 0.5418, 0.46, 0.30, 0.91]),
                "precision": per_class([0.97, 0.7213, 0.61, 0.35, 0.92]),
                "recall": per_class([0.97, 0.6852, 0.64, 0.66, 0.98]),
                "f1": per_class([0.97, 0.70, 0.63, 0.46, 0.95]),
            },
            "aggregate": {
                "mean_iou": 0.6353,
                "macro_precision": 0.72,
                "macro_recall": 0.79,
                "macro_f1": 0.7467,
                "pixel_accuracy": 0.9533,
                "oil_iou": 0.5418,
                "oil_recall": 0.6852,
            },
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Fast tests
# --------------------------------------------------------------------------- #


def test_export_sets_dynamic_axes(tmp_path: Path) -> None:
    """The exported graph names its IO and marks batch + spatial dims dynamic."""
    onnx_path = tmp_path / "tiny.onnx"
    _export_tiny(onnx_path)

    graph = onnx.load(str(onnx_path)).graph
    assert [i.name for i in graph.input] == [INPUT_NAME]
    assert [o.name for o in graph.output] == [OUTPUT_NAME]

    def dims(value_info: onnx.ValueInfoProto) -> list[str | int]:
        out: list[str | int] = []
        for d in value_info.type.tensor_type.shape.dim:
            out.append(d.dim_param if d.dim_param else d.dim_value)
        return out

    in_dims = dims(graph.input[0])
    out_dims = dims(graph.output[0])
    # batch, height, width are symbolic (dynamic); channel dim is fixed.
    assert in_dims[0] == "batch" and in_dims[2] == "height" and in_dims[3] == "width"
    assert in_dims[1] == 3
    assert out_dims[0] == "batch" and out_dims[2] == "height" and out_dims[3] == "width"
    assert out_dims[1] == NUM_CLASSES


def test_export_accepts_variable_sizes(tmp_path: Path) -> None:
    """A dynamically-exported graph runs at a size other than the trace size."""
    onnx_path = tmp_path / "tiny.onnx"
    _export_tiny(onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # Different batch AND spatial size than the (1, 3, 16, 16) trace.
    x = np.random.rand(2, 3, 24, 32).astype(np.float32)
    (logits,) = session.run(None, {INPUT_NAME: x})
    assert np.asarray(logits).shape == (2, NUM_CLASSES, 24, 32)


def test_tiny_parity(tmp_path: Path) -> None:
    """Tiny model: torch and ONNX logits match to tight tolerance."""
    onnx_path = tmp_path / "tiny.onnx"
    model = _export_tiny(onnx_path)
    x = torch.randn(4, 3, 16, 16)
    with torch.no_grad():
        torch_logits = model(x).numpy().astype(np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (onnx_logits,) = session.run(None, {INPUT_NAME: x.numpy().astype(np.float32)})
    assert float(np.max(np.abs(torch_logits - np.asarray(onnx_logits)))) < 1e-4


def test_model_card_contains_sections_and_real_numbers(tmp_path: Path) -> None:
    """The card includes the required sections and the run JSON's real numbers."""
    run_json = _synthetic_run_json(tmp_path / "run.json")
    card = build_model_card(
        run_json,
        repo_id="someuser/oil-spill-segmentation",
        arch="U-Net (ResNet-34 encoder)",
        onnx_filename="model.onnx",
    )

    # Required sections.
    for heading in (
        "## Intended use",
        "## Training dataset",
        "## Evaluation results",
        "## Class colour legend",
        "## Limitations",
    ):
        assert heading in card, f"missing section: {heading}"

    # Real numbers pulled from the run JSON (not invented).
    assert "0.5418" in card  # oil IoU
    assert "0.6852" in card  # oil recall
    assert "0.6353" in card  # mean IoU

    # Dataset, split, classes, colour legend, limitations content.
    assert "MKLab Oil Spill Detection Dataset" in card
    assert "1002" in card and "110" in card
    for name in CLASS_NAMES:
        assert name in card
    assert "#00FFFF" in card  # oil-spill legend colour
    assert "look-alike" in card.lower()
    assert "VV polarization" in card or "single VV" in card.lower()
    assert "not for sole operational" in card.lower()

    # No placeholder leftovers.
    for placeholder in ("TODO", "FIXME", "XXX", "{}", "<placeholder>", "PLACEHOLDER"):
        assert placeholder not in card, f"placeholder leftover: {placeholder}"


def test_publish_dry_run_builds_without_network(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """publish_hf --dry-run builds the card and lists files, no network."""
    publish_hf = _load_script("publish_hf")

    run_json = _synthetic_run_json(tmp_path / "run.json")
    onnx_path = tmp_path / "model.onnx"
    _export_tiny(onnx_path)

    # The checkpoint is only consulted for the architecture label on the card.
    # Stub the loader so the dry-run test stays fast and independent of whatever
    # real checkpoints are on disk (and so it never builds a heavy smp model).
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"stub")
    config = {"model": {"arch": "unet", "encoder_name": "resnet34"}}
    monkeypatch.setattr(
        publish_hf, "load_model_from_checkpoint", lambda *a, **k: (_TinySegModel(), config)
    )

    rc = publish_hf.main(
        [
            "--onnx",
            str(onnx_path),
            "--checkpoint",
            str(ckpt),
            "--run-json",
            str(run_json),
            "--repo-id",
            "someuser/oil-spill-segmentation",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "model.onnx" in out
    assert "README.md" in out
    assert "metrics.json" in out


# --------------------------------------------------------------------------- #
# Slow test: real checkpoint
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_real_checkpoint_export_parity(tmp_path: Path) -> None:
    """Export the newest loadable real checkpoint and assert tight ONNX parity."""
    from oilspill.evaluation.model_loading import load_model_from_checkpoint
    from oilspill.packaging.onnx_export import export_to_onnx, verify_parity

    ckpts = sorted(
        REPO_ROOT.glob("artifacts/checkpoints/*/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not ckpts:
        pytest.skip("no trained checkpoint available")

    # Pick the newest checkpoint that actually loads. A concurrently-written or
    # half-finished checkpoint can be incomplete/incompatible; skip past it
    # rather than fail the packaging path on it.
    config = None
    ckpt = None
    for candidate in ckpts:
        try:
            _model, config = load_model_from_checkpoint(candidate, device="cpu")
        except (RuntimeError, ValueError, KeyError):
            continue
        ckpt = candidate
        break
    if ckpt is None or config is None:
        pytest.skip("no loadable checkpoint available")

    cfg_size = config.get("data", {}).get("image_size", 256)
    image_size = cfg_size if isinstance(cfg_size, int) else int(cfg_size[0])

    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(ckpt, onnx_path, image_size=image_size, opset=17)
    assert onnx_path.exists()

    max_diff = verify_parity(ckpt, onnx_path, image_size=image_size, n=8, atol=1e-4)
    assert max_diff < 1e-4
