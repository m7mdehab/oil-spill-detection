"""Segmentation losses and a config-driven ``build_loss`` factory.

Four losses are selectable from config: weighted cross-entropy, multiclass Dice,
multiclass Focal, and a weighted Dice+Focal combination. Dice and Focal are the
well-tested implementations from ``segmentation_models_pytorch.losses`` wrapped
so every loss exposes the same ``callable(logits, target) -> scalar`` interface,
where ``logits`` is ``(N, C, H, W)`` and ``target`` is ``(N, H, W)`` int64.

Class weights
-------------
``LossConfig.class_weights`` may be an explicit list, ``None`` (uniform), or
``"auto"``. In ``auto`` mode weights are inverse-frequency normalised:

    w_c = (total_pixels / (num_classes * count_c))

clamped to avoid blow-up on classes absent from the sample, then rescaled so the
mean weight is 1. This up-weights the rare oil-spill class without changing the
overall loss scale.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
from torch import nn

from oilspill.metrics import NUM_CLASSES
from oilspill.training.config import LossConfig


def compute_auto_class_weights(
    targets: Iterable[torch.Tensor],
    num_classes: int = NUM_CLASSES,
    *,
    eps: float = 1.0,
) -> torch.Tensor:
    """Inverse-frequency class weights from integer target masks.

    ``targets`` is any iterable of int label tensors (e.g. the training masks).
    Counts are accumulated across all tensors; see the module docstring for the
    formula. Returns a float32 tensor of length ``num_classes`` with mean 1.
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for target in targets:
        binc = torch.bincount(target.reshape(-1).to(torch.int64), minlength=num_classes)
        counts += binc[:num_classes].to(torch.float64)

    total = counts.sum()
    if total == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    weights = total / (num_classes * (counts + eps))
    weights = weights / weights.mean()
    return weights.to(torch.float32)


class DiceFocalLoss(nn.Module):
    """Weighted sum ``w_dice * Dice + w_focal * Focal`` over the same logits."""

    def __init__(
        self,
        dice: DiceLoss,
        focal: FocalLoss,
        weights: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        super().__init__()
        self.dice = dice
        self.focal = focal
        self.w_dice, self.w_focal = weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w_dice * self.dice(logits, target) + self.w_focal * self.focal(logits, target)


def build_loss(
    cfg: LossConfig,
    *,
    num_classes: int = NUM_CLASSES,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """Build the loss module selected by ``cfg``.

    ``class_weights`` may be passed explicitly (e.g. the result of
    :func:`compute_auto_class_weights` when ``cfg.class_weights == "auto"``);
    otherwise an explicit list in the config is used. Weights apply to the
    cross-entropy term only (Dice/Focal handle imbalance differently).
    """
    weight = _resolve_weight(cfg, class_weights)

    if cfg.type == "ce":
        return nn.CrossEntropyLoss(weight=weight)

    dice = DiceLoss(mode="multiclass", from_logits=True)
    focal = FocalLoss(mode="multiclass", gamma=cfg.focal_gamma)

    if cfg.type == "dice":
        return dice
    if cfg.type == "focal":
        return focal
    if cfg.type == "dice_focal":
        return DiceFocalLoss(dice, focal, weights=cfg.dice_focal_weights)

    raise ValueError(f"unknown loss type: {cfg.type!r}")


def _resolve_weight(cfg: LossConfig, class_weights: torch.Tensor | None) -> torch.Tensor | None:
    if class_weights is not None:
        return class_weights.to(torch.float32)
    if isinstance(cfg.class_weights, list):
        return torch.tensor(cfg.class_weights, dtype=torch.float32)
    return None
