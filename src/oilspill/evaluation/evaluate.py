"""Core evaluation loop: run a model over a dataset and accumulate metrics.

All metric computation routes through :class:`oilspill.metrics.SegmentationMetrics`
(the single metric source). In addition to the dataset-wide confusion matrix, the
loop records the per-image oil-class IoU so the gallery harness can pick the best
and worst predictions, and optionally accumulates per-class softmax probabilities
so precision-recall curves can be plotted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset

from oilspill.data.dataset import Sample
from oilspill.metrics import (
    NUM_CLASSES,
    OIL_CLASS_INDEX,
    MetricResult,
    SegmentationMetrics,
    compute_metrics,
    logits_to_labels,
)

from .model_loading import ModelOrSession, predict_logits


@dataclass
class EvaluationOutput:
    """Everything the reporting layer needs from one evaluation pass.

    ``per_image_oil_iou`` maps a sample name to its oil-class IoU (``nan`` when the
    oil class is absent from both prediction and ground truth for that image).
    ``pr_curve_data``, when collected, holds flattened per-class probabilities and
    the binary ground-truth target arrays used to build PR curves.
    """

    result: MetricResult
    per_image_oil_iou: dict[str, float]
    num_images: int
    pr_probs: torch.Tensor | None = field(default=None)  # (num_pixels, C)
    pr_targets: torch.Tensor | None = field(default=None)  # (num_pixels,) int


def _collate(batch: Sequence[Sample]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    names = [item["name"] for item in batch]
    return images, masks, names


def _image_oil_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Oil-class IoU for a single (H, W) prediction/target pair."""
    one = compute_metrics(pred.unsqueeze(0), target.unsqueeze(0), num_classes=NUM_CLASSES)
    return one.oil_iou


def evaluate_model(
    model: ModelOrSession,
    dataset: Dataset[Sample],
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 4,
    num_workers: int = 0,
    max_images: int | None = None,
    collect_pr: bool = True,
    pr_pixel_stride: int = 4,
) -> EvaluationOutput:
    """Evaluate ``model`` over ``dataset`` and return metrics plus per-image stats.

    Parameters
    ----------
    model:
        A torch module or ONNX session (see :func:`predict_logits`).
    dataset:
        A dataset yielding :class:`~oilspill.data.dataset.Sample` items.
    device:
        Device for torch inference (ignored for ONNX).
    batch_size, num_workers:
        DataLoader settings.
    max_images:
        If set, evaluate at most this many images (handy for fast smoke runs).
    collect_pr:
        If True, subsample pixel-level softmax probabilities/targets so
        precision-recall curves can be plotted. Subsampling keeps memory bounded.
    pr_pixel_stride:
        Take every ``pr_pixel_stride``-th pixel (per image, flattened) for the PR
        sample. ``1`` keeps every pixel.
    """
    metric = SegmentationMetrics(num_classes=NUM_CLASSES)
    per_image_oil_iou: dict[str, float] = {}
    pr_probs_chunks: list[torch.Tensor] = []
    pr_targets_chunks: list[torch.Tensor] = []

    loader: DataLoader[Sample] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    seen = 0
    for images, masks, names in loader:
        if max_images is not None and seen >= max_images:
            break
        if max_images is not None and seen + images.shape[0] > max_images:
            keep = max_images - seen
            images, masks, names = images[:keep], masks[:keep], names[:keep]

        logits = predict_logits(model, images)  # (N, C, H, W) on CPU
        preds = logits_to_labels(logits)  # (N, H, W)
        masks_cpu = masks.cpu()
        metric.update(preds, masks_cpu)

        for i, name in enumerate(names):
            per_image_oil_iou[name] = _image_oil_iou(preds[i], masks_cpu[i])

        if collect_pr:
            probs = torch.softmax(logits.float(), dim=1)  # (N, C, H, W)
            # Flatten to (pixels, C) and (pixels,), then subsample.
            flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)
            flat_targets = masks_cpu.reshape(-1)
            if pr_pixel_stride > 1:
                flat_probs = flat_probs[::pr_pixel_stride]
                flat_targets = flat_targets[::pr_pixel_stride]
            pr_probs_chunks.append(flat_probs)
            pr_targets_chunks.append(flat_targets)

        seen += images.shape[0]

    result = metric.compute()
    pr_probs = torch.cat(pr_probs_chunks) if pr_probs_chunks else None
    pr_targets = torch.cat(pr_targets_chunks) if pr_targets_chunks else None

    return EvaluationOutput(
        result=result,
        per_image_oil_iou=per_image_oil_iou,
        num_images=seen,
        pr_probs=pr_probs,
        pr_targets=pr_targets,
    )


def rank_by_oil_iou(per_image_oil_iou: dict[str, float]) -> list[tuple[str, float]]:
    """Sort images by oil-class IoU ascending; ``nan`` scores sort last.

    Images where the oil class is wholly absent (IoU ``nan``) are uninformative
    for a best/worst oil gallery, so they are pushed to the end.
    """

    def key(item: tuple[str, float]) -> tuple[int, float]:
        value = item[1]
        is_nan = value != value
        return (1 if is_nan else 0, value)

    return sorted(per_image_oil_iou.items(), key=key)


__all__ = [
    "OIL_CLASS_INDEX",
    "EvaluationOutput",
    "evaluate_model",
    "rank_by_oil_iou",
]
