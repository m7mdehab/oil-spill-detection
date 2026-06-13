"""Typed, YAML-backed training configuration.

The whole training engine is config-driven: every knob lives in a YAML file that
is parsed into the :class:`TrainConfig` model below. Pydantic gives us validation
and sane defaults, and :meth:`TrainConfig.flatten` produces the dotted-key dict
that gets logged to MLflow as run params.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from oilspill.metrics import NUM_CLASSES

LossType = Literal["ce", "dice", "focal", "dice_focal"]


class ModelConfig(BaseModel):
    """Segmentation model architecture (passed through to ``smp``)."""

    model_config = ConfigDict(extra="forbid")

    arch: str = "Unet"
    encoder: str = "resnet34"
    # ``imagenet`` downloads pretrained weights; ``null`` trains from scratch
    # (used by the smoke config to avoid a network round-trip).
    encoder_weights: str | None = "imagenet"
    in_channels: int = 3
    num_classes: int = NUM_CLASSES


class DataConfig(BaseModel):
    """Dataset location, image geometry and dataloader settings."""

    model_config = ConfigDict(extra="forbid")

    root: Path = Path("data/datasets/oil_spill")
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 0
    # Fraction of the training split held out for validation when the dataset
    # does not provide its own validation split.
    val_split: float = Field(default=0.2, ge=0.0, lt=1.0)
    split_seed: int = 42


class OptimConfig(BaseModel):
    """Optimiser and learning-rate schedule."""

    model_config = ConfigDict(extra="forbid")

    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = Field(default=50, ge=1)
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    scheduler: Literal["none", "cosine", "step"] = "cosine"
    # Used only by the ``step`` scheduler.
    step_size: int = 10
    gamma: float = 0.1


class LossConfig(BaseModel):
    """Loss selection and its hyper-parameters."""

    model_config = ConfigDict(extra="forbid")

    type: LossType = "dice_focal"
    # Per-class weights for cross-entropy. Either an explicit list of length
    # ``num_classes``, the string ``"auto"`` (inverse-frequency, computed from
    # the training masks), or null (uniform).
    class_weights: list[float] | Literal["auto"] | None = None
    focal_gamma: float = 2.0
    # Convex-ish combo weights for ``dice_focal``: total = w_dice*Dice + w_focal*Focal.
    dice_focal_weights: tuple[float, float] = (0.5, 0.5)


class RuntimeConfig(BaseModel):
    """Execution environment: device, seed and mixed precision."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    amp: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["fp16", "bf16"] = "fp16"
    deterministic: bool = True


class EarlyStoppingConfig(BaseModel):
    """Early stopping on a monitored validation metric."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    monitor: str = "val_oil_iou"
    mode: Literal["max", "min"] = "max"
    patience: int = Field(default=10, ge=1)
    min_delta: float = 0.0


class CheckpointConfig(BaseModel):
    """Where and when to write checkpoints (under ``artifacts/``)."""

    model_config = ConfigDict(extra="forbid")

    dir: Path = Path("artifacts/checkpoints")
    save_best: bool = True
    save_last: bool = True
    monitor: str = "val_oil_iou"
    mode: Literal["max", "min"] = "max"


class MLflowConfig(BaseModel):
    """MLflow experiment tracking (local file store by default)."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: str = "oil-spill-segmentation"
    tracking_uri: str = "./mlruns"
    run_name: str | None = None
    log_samples: int = 4


class TrainConfig(BaseModel):
    """Top-level training configuration."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    @model_validator(mode="after")
    def _check_class_weights(self) -> TrainConfig:
        weights = self.loss.class_weights
        if isinstance(weights, list) and len(weights) != self.model.num_classes:
            raise ValueError(
                f"loss.class_weights has {len(weights)} entries but "
                f"model.num_classes={self.model.num_classes}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load and validate a config from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
        return cls.model_validate(data)

    def flatten(self) -> dict[str, Any]:
        """Dotted-key view of the config for MLflow param logging."""

        def _flatten(prefix: str, value: Any) -> dict[str, Any]:
            out: dict[str, Any] = {}
            if isinstance(value, dict):
                for key, sub in value.items():
                    out.update(_flatten(f"{prefix}.{key}" if prefix else str(key), sub))
            elif isinstance(value, (list, tuple)):
                out[prefix] = ",".join(str(v) for v in value)
            else:
                out[prefix] = value
            return out

        return _flatten("", self.model_dump(mode="json"))
