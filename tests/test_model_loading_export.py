"""Tests for the model-loading helpers and the real ONNX export entry points.

These exercise :mod:`oilspill.evaluation.model_loading` and the
:func:`export_to_onnx` / :func:`verify_parity` functions in
:mod:`oilspill.packaging.onnx_export` using a tiny fully-convolutional model
written to a real trainer-style checkpoint. No heavy ``smp`` model is built and
no network is touched, so the tests stay fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn

from oilspill.evaluation.model_loading import (
    load_model_from_checkpoint,
    load_onnx_session,
    predict_logits,
)
from oilspill.metrics import NUM_CLASSES
from oilspill.packaging.onnx_export import (
    INPUT_NAME,
    _as_hw,
    export_to_onnx,
    verify_parity,
)


class _TinySegModel(nn.Module):
    """Fully-convolutional 3->NUM_CLASSES model standing in for the seg net."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, NUM_CLASSES, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _write_checkpoint(path: Path) -> _TinySegModel:
    """Write a trainer-style checkpoint whose config builds an smp Unet.

    The config selects a real (but tiny) smp architecture so the round-trip
    through :func:`build_model_from_config` is genuinely exercised.
    """
    model = _TinySegModel().eval()
    # A minimal smp Unet config; mobilenet_v2 is light and weight-free here
    # because pretrained=False is used on the eval path.
    config = {
        "model": {
            "arch": "Unet",
            "encoder": "mobilenet_v2",
            "encoder_weights": None,
            "in_channels": 3,
            "num_classes": NUM_CLASSES,
        }
    }
    # Build the real smp model so the state dict matches what loading rebuilds.
    from oilspill.models import build_model
    from oilspill.training.config import ModelConfig

    smp_model = build_model(ModelConfig.model_validate(config["model"]), pretrained=False)
    torch.save({"model_state_dict": smp_model.state_dict(), "config": config}, path)
    return model


def test_as_hw_normalises_int_and_tuple() -> None:
    assert _as_hw(64) == (64, 64)
    assert _as_hw((32, 48)) == (32, 48)


def test_load_model_from_checkpoint_rebuilds_and_runs(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.pt"
    _write_checkpoint(ckpt)

    model, config = load_model_from_checkpoint(ckpt, device="cpu")
    assert config["model"]["num_classes"] == NUM_CLASSES
    assert not model.training  # eval mode
    with torch.no_grad():
        out = model(torch.rand(1, 3, 64, 64))
    assert out.shape == (1, NUM_CLASSES, 64, 64)


def test_load_model_from_checkpoint_rejects_non_checkpoint(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"not_a": "checkpoint"}, bad)
    with pytest.raises(ValueError, match="missing 'config'"):
        load_model_from_checkpoint(bad)


def test_load_onnx_session_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        load_onnx_session(tmp_path / "nope.onnx")


def test_predict_logits_accepts_torch_module() -> None:
    model = _TinySegModel().eval()
    batch = torch.randn(2, 3, 16, 16)
    logits = predict_logits(model, batch)
    assert logits.shape == (2, NUM_CLASSES, 16, 16)
    assert logits.device.type == "cpu"


def test_predict_logits_accepts_onnx_session(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.pt"
    _write_checkpoint(ckpt)
    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(ckpt, onnx_path, image_size=32, opset=17)

    session = load_onnx_session(onnx_path)
    assert isinstance(session, ort.InferenceSession)
    batch = torch.randn(1, 3, 32, 32)
    logits = predict_logits(session, batch)
    assert logits.shape == (1, NUM_CLASSES, 32, 32)


def test_export_to_onnx_and_parity_roundtrip(tmp_path: Path) -> None:
    """The real export + parity helpers run on a tiny checkpoint."""
    ckpt = tmp_path / "best.pt"
    _write_checkpoint(ckpt)
    onnx_path = tmp_path / "model.onnx"

    out = export_to_onnx(ckpt, onnx_path, image_size=(32, 32), opset=17)
    assert out == onnx_path
    assert onnx_path.exists() and onnx_path.stat().st_size > 0

    max_diff = verify_parity(ckpt, onnx_path, image_size=32, n=2, atol=1e-3)
    assert max_diff < 1e-3

    # The exported session reproduces shapes for the input it was traced at.
    session = load_onnx_session(onnx_path)
    x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    (logits,) = session.run(None, {INPUT_NAME: x})
    assert np.asarray(logits).shape == (1, NUM_CLASSES, 32, 32)
