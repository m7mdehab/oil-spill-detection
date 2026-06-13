"""Tests for the training engine: losses, seeding, config, and a fit() smoke run.

The pure-function tests (losses, seeding, config) are fast and unmarked. The
end-to-end ``fit()`` test runs a real (tiny) training loop and is marked
``slow`` so it can be skipped in quick test passes.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from oilspill.metrics import NUM_CLASSES
from oilspill.training.config import LossConfig, TrainConfig
from oilspill.training.datasets import SyntheticSegmentationDataset
from oilspill.training.losses import build_loss, compute_auto_class_weights
from oilspill.training.seed import seed_everything


def _tiny_logits_targets() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.randn(2, NUM_CLASSES, 8, 8, requires_grad=True)
    targets = torch.randint(0, NUM_CLASSES, (2, 8, 8))
    return logits, targets


@pytest.mark.parametrize("loss_type", ["ce", "dice", "focal", "dice_focal"])
def test_build_loss_finite_and_backward(loss_type: str) -> None:
    logits, targets = _tiny_logits_targets()
    loss_fn = build_loss(LossConfig(type=loss_type))  # type: ignore[arg-type]
    loss = loss_fn(logits, targets)
    assert loss.ndim == 0
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_build_loss_with_explicit_class_weights() -> None:
    logits, targets = _tiny_logits_targets()
    cfg = LossConfig(type="ce", class_weights=[1.0, 5.0, 1.0, 1.0, 1.0])
    loss = build_loss(cfg)(logits, targets)
    assert math.isfinite(float(loss.detach()))


def test_build_loss_rejects_unknown_type() -> None:
    cfg = LossConfig(type="ce")
    object.__setattr__(cfg, "type", "bogus")
    with pytest.raises(ValueError, match="unknown loss type"):
        build_loss(cfg)


def test_compute_auto_class_weights_inverse_frequency() -> None:
    # class 0 dominant, class 1 rare -> class 1 gets the larger weight.
    masks = [torch.zeros(10, 10, dtype=torch.int64)]
    masks[0][0, 0] = 1
    weights = compute_auto_class_weights(masks, num_classes=NUM_CLASSES)
    assert weights.shape == (NUM_CLASSES,)
    assert weights[1] > weights[0]
    assert torch.isfinite(weights).all()
    assert weights.mean() == pytest.approx(1.0, rel=1e-5)


def test_compute_auto_class_weights_empty_is_uniform() -> None:
    weights = compute_auto_class_weights([torch.zeros(0, dtype=torch.int64)], num_classes=3)
    assert torch.allclose(weights, torch.ones(3))


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))


def test_seed_everything_is_deterministic() -> None:
    seed_everything(123)
    model_a = _tiny_model()
    x = torch.randn(2, 4)
    out_a = model_a(x)

    seed_everything(123)
    model_b = _tiny_model()
    x_b = torch.randn(2, 4)
    out_b = model_b(x_b)

    assert torch.allclose(out_a, out_b)
    assert torch.allclose(x, x_b)


def test_seed_everything_returns_seed() -> None:
    assert seed_everything(7) == 7


def test_config_loads_from_yaml(tmp_path: Path) -> None:
    yaml_text = """
model:
  arch: Unet
  encoder: resnet34
  encoder_weights: null
  num_classes: 5
optim:
  epochs: 3
  lr: 0.01
loss:
  type: dice_focal
runtime:
  device: cpu
  amp: false
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    cfg = TrainConfig.from_yaml(path)
    assert cfg.model.arch == "Unet"
    assert cfg.optim.epochs == 3
    assert cfg.optim.lr == 0.01
    assert cfg.loss.type == "dice_focal"
    assert cfg.runtime.device == "cpu"


def test_config_defaults() -> None:
    cfg = TrainConfig()
    assert cfg.model.num_classes == NUM_CLASSES
    assert cfg.loss.type == "dice_focal"
    flat = cfg.flatten()
    assert flat["model.arch"] == "Unet"
    assert "optim.lr" in flat


def test_config_rejects_mismatched_class_weights() -> None:
    with pytest.raises(ValueError, match="class_weights"):
        TrainConfig.model_validate(
            {"model": {"num_classes": 5}, "loss": {"class_weights": [1.0, 2.0]}}
        )


def test_synthetic_dataset_shapes_and_determinism() -> None:
    ds = SyntheticSegmentationDataset(length=3, image_size=16, in_channels=3, num_classes=5, seed=1)
    assert len(ds) == 3
    img, mask = ds[0]
    assert img.shape == (3, 16, 16)
    assert mask.shape == (16, 16)
    assert mask.dtype == torch.int64
    assert int(mask.max()) < 5
    img2, mask2 = ds[0]
    assert torch.allclose(img, img2)
    assert torch.equal(mask, mask2)


@pytest.mark.slow
def test_fit_smoke_writes_checkpoint(tmp_path: Path) -> None:
    from oilspill.training.trainer import fit

    cfg = TrainConfig.model_validate(
        {
            "model": {
                "arch": "Unet",
                "encoder": "resnet18",
                "encoder_weights": None,
                "num_classes": NUM_CLASSES,
            },
            "data": {"image_size": 32, "batch_size": 2, "num_workers": 0},
            "optim": {"epochs": 1, "scheduler": "none"},
            "loss": {"type": "dice_focal"},
            "runtime": {"device": "cpu", "amp": False, "seed": 0},
            "early_stopping": {"enabled": False},
            "checkpoint": {"dir": tmp_path / "ckpt"},
            "mlflow": {
                "tracking_uri": (tmp_path / "mlruns").as_uri(),
                "experiment_name": "test",
                "log_samples": 1,
            },
        }
    )
    train_ds = SyntheticSegmentationDataset(
        length=4, image_size=32, num_classes=NUM_CLASSES, seed=1
    )
    val_ds = SyntheticSegmentationDataset(length=2, image_size=32, num_classes=NUM_CLASSES, seed=2)

    result = fit(cfg, train_dataset=train_ds, val_dataset=val_ds)

    assert result.best_checkpoint.exists()
    assert result.last_checkpoint.exists()
    assert result.mlflow_run_id
    assert "train_loss" in result.metrics
    ckpt = torch.load(result.best_checkpoint, weights_only=False)
    assert "model_state_dict" in ckpt
    assert ckpt["config"]["model"]["arch"] == "Unet"
