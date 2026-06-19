"""Tests for the EO foundation model (DINOv2 + conv head) and registry wiring.

The shape tests build the model with ``pretrained=False`` from a small
``dinov2-small`` config and therefore run **offline** (no weight download). The
single download-dependent path is marked ``slow``.
"""

from __future__ import annotations

import pytest
import torch

from oilspill.models import build_model, registered_architectures
from oilspill.models.foundation import Foundation, build_foundation
from oilspill.training.config import ModelConfig


def _small_cfg(num_classes: int = 5, **params: object) -> ModelConfig:
    """A small dinov2-small model config that builds offline."""
    base: dict[str, object] = {"backbone": "facebook/dinov2-small", "head_hidden": 64}
    base.update(params)
    return ModelConfig(
        arch="foundation",
        in_channels=3,
        num_classes=num_classes,
        params=base,
    )


def test_offline_pos_embeddings_match_pretrained_size() -> None:
    """Offline build must size position embeddings to DINOv2's native 518px grid
    (1370 tokens), so a checkpoint trained from the pretrained backbone loads.

    Guards a regression where the offline config used the default 224px image size
    (257 tokens), making trained checkpoints fail to load for evaluation.
    """
    model = build_model(_small_cfg(), pretrained=False)
    pos = dict(model.named_parameters())["backbone.embeddings.position_embeddings"]
    assert tuple(pos.shape) == (1, 1370, 384)
    # Two independent offline builds round-trip strictly (architecture is stable).
    other = build_model(_small_cfg(), pretrained=False)
    other.load_state_dict(model.state_dict(), strict=True)


def test_foundation_is_registered() -> None:
    assert "foundation" in registered_architectures()


def test_foundation_forward_shape_and_dtype() -> None:
    # pretrained=False builds from the built-in dinov2-small variant table; no net.
    model = build_foundation(_small_cfg(), False)
    assert isinstance(model, Foundation)
    model.eval()
    # 28 = 2 * 14 is a multiple of the DINOv2 patch size.
    x = torch.randn(2, 3, 28, 28)
    with torch.no_grad():
        out = model(x)
    # Logits must come back at the INPUT resolution.
    assert out.shape == (2, 5, 28, 28)
    assert out.dtype == torch.float32


def test_build_model_via_registry_offline() -> None:
    # The registry resolves arch="foundation" to our builder and round-trips shapes.
    model = build_model(_small_cfg(num_classes=5), pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 42, 42)  # 42 = 3 * 14
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5, 42, 42)
    assert out.dtype.is_floating_point


def test_foundation_respects_num_classes() -> None:
    model = build_foundation(_small_cfg(num_classes=3), False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 28, 28))
    assert out.shape == (1, 3, 28, 28)


def test_foundation_non_square_input() -> None:
    # The head reshapes tokens to the patch grid; non-square inputs must still
    # round-trip back to the exact input resolution.
    model = build_foundation(_small_cfg(), False)
    model.eval()
    x = torch.randn(1, 3, 28, 42)  # 2x14 by 3x14
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5, 28, 42)


def test_foundation_freeze_backbone() -> None:
    # With the backbone frozen, only the decode head should require grad.
    model = build_foundation(_small_cfg(freeze_backbone=True), False)
    assert isinstance(model, Foundation)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert any(p.requires_grad for p in model.head.parameters())


def test_foundation_rejects_non_three_channels() -> None:
    cfg = ModelConfig(
        arch="foundation",
        in_channels=1,
        num_classes=5,
        params={"backbone": "facebook/dinov2-small"},
    )
    with pytest.raises(ValueError, match="in_channels=3"):
        build_foundation(cfg, False)


@pytest.mark.slow
def test_foundation_pretrained_downloads_and_runs() -> None:
    # Exercises the pretrained path (downloads dinov2-small base weights once).
    model = build_foundation(_small_cfg(), True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 28, 28))
    assert out.shape == (1, 5, 28, 28)
