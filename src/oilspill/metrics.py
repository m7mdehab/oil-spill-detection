"""Canonical segmentation metrics for the oil-spill task.

This module is the *single* place metrics are computed anywhere in the project.
Training, evaluation, the API, and the docs all route through
:class:`SegmentationMetrics`. There is deliberately no second implementation.

Design
------
Every metric is derived from one accumulated multiclass confusion matrix
(rows = ground truth, columns = prediction). torchmetrics
(:class:`~torchmetrics.classification.MulticlassConfusionMatrix`) does the
accumulation and handles ``ignore_index``; the per-class statistics are then
read straight off the matrix using textbook definitions:

* ``TP_i = C[i, i]``
* ``FP_i = sum_j C[j, i] - TP_i``   (predicted i, truth was not i)
* ``FN_i = sum_j C[i, j] - TP_i``   (truth i, predicted something else)

from which, per class,

* ``IoU_i       = TP / (TP + FP + FN)``
* ``precision_i = TP / (TP + FP)``
* ``recall_i    = TP / (TP + FN)``
* ``F1_i        = 2 TP / (2 TP + FP + FN)``

Undefined classes
-----------------
A class that never appears in either the targets or the predictions has
``TP + FP + FN = 0``; its IoU/precision/recall/F1 are genuinely undefined and
are reported as ``nan`` rather than a misleading ``0`` or ``1``. Macro averages
are ``nanmean`` over the classes, i.e. undefined classes are excluded from the
average instead of dragging it toward zero. This choice is what makes the macro
numbers honest on an imbalanced dataset where a rare class may be absent from a
given split; see ``docs/metrics.md``.

Inputs are integer label tensors of identical shape (any shape; they are
flattened). Model logits of shape ``(N, C, H, W)`` should be converted first
with :func:`logits_to_labels`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchmetrics.classification import MulticlassConfusionMatrix

# Canonical 5-class scheme, recovered from the original 2024 project and fixed
# for the entire repository. Order is class id 0..4.
CLASS_NAMES: tuple[str, ...] = (
    "Sea Surface",
    "Oil Spill",
    "Look-alike",
    "Ship",
    "Land",
)
NUM_CLASSES: int = len(CLASS_NAMES)

# Index of the headline class. Oil-class IoU and recall are the metrics the
# project is selected and reported on (not pixel accuracy).
OIL_CLASS_INDEX: int = 1


def logits_to_labels(logits: torch.Tensor) -> torch.Tensor:
    """Convert model logits ``(N, C, H, W)`` to label indices ``(N, H, W)``."""
    if logits.ndim < 2:
        raise ValueError(f"expected logits with a class dim, got shape {tuple(logits.shape)}")
    return logits.argmax(dim=1)


@dataclass(frozen=True)
class MetricResult:
    """Computed metrics. Per-class tensors are length ``num_classes``.

    Undefined per-class entries are ``nan`` (see module docstring). Scalar
    aggregates ignore those ``nan`` entries.
    """

    confusion_matrix: torch.Tensor  # (C, C) int64, rows=truth, cols=pred
    iou: torch.Tensor  # (C,)
    precision: torch.Tensor  # (C,)
    recall: torch.Tensor  # (C,)
    f1: torch.Tensor  # (C,)
    mean_iou: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    pixel_accuracy: float
    class_names: tuple[str, ...]

    @property
    def oil_iou(self) -> float:
        """IoU of the oil-spill class — a headline metric for this project."""
        return float(self.iou[OIL_CLASS_INDEX])

    @property
    def oil_recall(self) -> float:
        """Recall of the oil-spill class — a headline metric for this project."""
        return float(self.recall[OIL_CLASS_INDEX])

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable view. The single source for any number in the docs."""

        def per_class(values: torch.Tensor) -> dict[str, float | None]:
            return {
                name: (None if torch.isnan(v) else float(v))
                for name, v in zip(self.class_names, values, strict=True)
            }

        return {
            "class_names": list(self.class_names),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class": {
                "iou": per_class(self.iou),
                "precision": per_class(self.precision),
                "recall": per_class(self.recall),
                "f1": per_class(self.f1),
            },
            "aggregate": {
                "mean_iou": self.mean_iou,
                "macro_precision": self.macro_precision,
                "macro_recall": self.macro_recall,
                "macro_f1": self.macro_f1,
                "pixel_accuracy": self.pixel_accuracy,
                "oil_iou": self.oil_iou,
                "oil_recall": self.oil_recall,
            },
        }


def _metrics_from_confusion(cm: torch.Tensor, class_names: tuple[str, ...]) -> MetricResult:
    """Derive all metrics from a confusion matrix ``cm`` (rows=truth, cols=pred)."""
    cm = cm.to(torch.float64)
    tp = torch.diagonal(cm)
    fp = cm.sum(dim=0) - tp  # predicted class i, truth differed
    fn = cm.sum(dim=1) - tp  # truth class i, predicted differed

    iou = tp / (tp + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)

    total = cm.sum()
    pixel_accuracy = float(tp.sum() / total) if total > 0 else float("nan")

    return MetricResult(
        confusion_matrix=cm.to(torch.int64),
        iou=iou,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=float(torch.nanmean(iou)),
        macro_precision=float(torch.nanmean(precision)),
        macro_recall=float(torch.nanmean(recall)),
        macro_f1=float(torch.nanmean(f1)),
        pixel_accuracy=pixel_accuracy,
        class_names=class_names,
    )


class SegmentationMetrics:
    """Stateful metric aggregator. Accumulate batches, then ``compute()``.

    Example
    -------
    >>> metric = SegmentationMetrics()
    >>> metric.update(pred_labels, target_labels)  # doctest: +SKIP
    >>> result = metric.compute()                   # doctest: +SKIP
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        ignore_index: int | None = None,
        class_names: tuple[str, ...] | None = None,
    ) -> None:
        if class_names is None:
            class_names = CLASS_NAMES if num_classes == NUM_CLASSES else None  # type: ignore[assignment]
        if class_names is None:
            class_names = tuple(f"class_{i}" for i in range(num_classes))
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names has {len(class_names)} entries but num_classes={num_classes}"
            )
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names
        self._cm = MulticlassConfusionMatrix(
            num_classes=num_classes,
            ignore_index=ignore_index,
            normalize="none",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch of integer label tensors (identical shape)."""
        if preds.shape != target.shape:
            raise ValueError(
                f"preds shape {tuple(preds.shape)} != target shape {tuple(target.shape)}; "
                "convert logits with logits_to_labels() first"
            )
        self._cm.update(preds.reshape(-1), target.reshape(-1))

    def reset(self) -> None:
        self._cm.reset()

    def compute(self) -> MetricResult:
        """Return all metrics derived from the accumulated confusion matrix.

        Note: ``MulticlassConfusionMatrix`` returns rows=truth, cols=pred, which
        is the convention this module's derivations assume.
        """
        return _metrics_from_confusion(self._cm.compute(), self.class_names)


def compute_metrics(
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int | None = None,
    class_names: tuple[str, ...] | None = None,
) -> MetricResult:
    """One-shot convenience wrapper around :class:`SegmentationMetrics`."""
    metric = SegmentationMetrics(
        num_classes=num_classes, ignore_index=ignore_index, class_names=class_names
    )
    metric.update(preds, target)
    return metric.compute()
