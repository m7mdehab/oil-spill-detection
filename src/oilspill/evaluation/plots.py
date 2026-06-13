"""Matplotlib figures for the evaluation report: confusion matrix and PR curves.

A headless ``Agg`` backend is selected at import time so the harness runs on CI
and servers without a display. Every figure is closed after saving to keep memory
flat over long runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.classification import MulticlassPrecisionRecallCurve

from oilspill.metrics import NUM_CLASSES, MetricResult


def plot_confusion_matrix(result: MetricResult, out_path: Path | str) -> Path:
    """Plot the confusion matrix (row-normalised colour + raw counts) to a PNG.

    Rows are ground truth, columns are predictions (the matrix convention used by
    :mod:`oilspill.metrics`). Cells show the raw count and the colour encodes the
    row-normalised fraction, so reading across a row shows where a true class's
    pixels actually went.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = result.confusion_matrix.cpu().numpy().astype(np.int64)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normed = np.where(row_sums > 0, cm / row_sums, 0.0)

    names = list(result.class_names)
    n = len(names)

    fig, ax = plt.subplots(figsize=(1.6 * n + 2, 1.6 * n + 2))
    im = ax.imshow(normed, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised fraction")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix (counts; colour = row fraction)")

    for i in range(n):
        for j in range(n):
            frac = normed[i, j]
            colour = "white" if frac > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}\n{frac * 100:.1f}%",
                ha="center",
                va="center",
                color=colour,
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_pr_curves(
    probs: torch.Tensor,
    targets: torch.Tensor,
    class_names: tuple[str, ...],
    out_path: Path | str,
) -> Path:
    """Plot per-class one-vs-rest precision-recall curves to a PNG.

    Parameters
    ----------
    probs:
        ``(N, C)`` softmax probabilities over the sampled pixels.
    targets:
        ``(N,)`` integer ground-truth class labels.
    class_names:
        Class names (length ``C``).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = probs.shape[1]
    pr_curve = MulticlassPrecisionRecallCurve(num_classes=num_classes)
    pr_curve.update(probs, targets.long())
    precision, recall, _ = pr_curve.compute()

    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(num_classes):
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        ax.plot(recall[c].cpu().numpy(), precision[c].cpu().numpy(), label=name, linewidth=1.8)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Per-class precision-recall curves (one-vs-rest)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


__all__ = ["NUM_CLASSES", "plot_confusion_matrix", "plot_pr_curves"]
