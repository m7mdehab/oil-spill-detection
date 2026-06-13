"""Tests for the Hugging Face SegFormer wrapper and its registry integration.

The shape tests build the model with ``pretrained=False`` from a small MiT-b0
config and therefore run offline (no weight download). The single download-
dependent path is marked ``slow``.
"""

from __future__ import annotations

import pytest
import torch

from oilspill.models import build_model, registered_architectures
from oilspill.models.segformer import SegFormer, build_segformer
from oilspill.training.config import ModelConfig


def _b0_cfg(num_classes: int = 5) -> ModelConfig:
    """A small MiT-b0 model config that builds offline."""
    return ModelConfig(
        arch="segformer",
        in_channels=3,
        num_classes=num_classes,
        params={"hf_model": "nvidia/mit-b0"},
    )


def test_segformer_is_registered() -> None:
    assert "segformer" in registered_architectures()


def test_segformer_forward_shape_and_dtype() -> None:
    # pretrained=False builds from the built-in mit-b0 variant table, no network.
    model = build_segformer(_b0_cfg(), False)
    assert isinstance(model, SegFormer)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    # Logits must come back at the INPUT resolution (HF emits H/4 x W/4 natively).
    assert out.shape == (2, 5, 64, 64)
    assert out.dtype == torch.float32


def test_build_model_via_registry_offline() -> None:
    # The registry resolves arch="segformer" to our builder and round-trips shapes.
    model = build_model(_b0_cfg(num_classes=5), pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5, 32, 32)
    assert out.dtype.is_floating_point


def test_segformer_respects_num_classes() -> None:
    model = build_segformer(_b0_cfg(num_classes=3), False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 3, 64, 64)


def test_segformer_rejects_non_three_channels() -> None:
    cfg = ModelConfig(
        arch="segformer",
        in_channels=1,
        num_classes=5,
        params={"hf_model": "nvidia/mit-b0"},
    )
    with pytest.raises(ValueError, match="in_channels=3"):
        build_segformer(cfg, False)


@pytest.mark.slow
def test_segformer_pretrained_downloads_and_runs() -> None:
    # Exercises the pretrained path (downloads mit-b0 base weights once).
    model = build_segformer(_b0_cfg(), True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 5, 64, 64)
