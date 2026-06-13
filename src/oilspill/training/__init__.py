"""Training loops, experiment tracking, and checkpointing."""

from __future__ import annotations

from oilspill.training.config import TrainConfig
from oilspill.training.losses import build_loss, compute_auto_class_weights
from oilspill.training.seed import seed_everything
from oilspill.training.trainer import (
    SMOKE_SUBSET_FRACTION,
    FitResult,
    fit,
    make_smoke_config,
)

__all__ = [
    "SMOKE_SUBSET_FRACTION",
    "FitResult",
    "TrainConfig",
    "build_loss",
    "compute_auto_class_weights",
    "fit",
    "make_smoke_config",
    "seed_everything",
]
