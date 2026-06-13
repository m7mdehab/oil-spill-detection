"""Render best/worst prediction galleries ranked by oil-class IoU.

Each panel is a horizontal strip: the input SAR image, the colourised ground-truth
mask, and the colourised prediction (using the project's authoritative class
legend via :func:`oilspill.data.colorize_mask`). Images are ranked by per-image
oil-class IoU so the gallery surfaces both the model's strongest oil predictions
and its failures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from oilspill.data import colorize_mask
from oilspill.data.dataset import Sample
from oilspill.metrics import logits_to_labels

from .evaluate import rank_by_oil_iou
from .model_loading import ModelOrSession, predict_logits


def _denormalise_image(image: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor to a displayable HxWx3 uint8 array.

    The exact normalisation statistics are not needed for display; a per-image
    min-max stretch yields a clear visualisation of the SAR backscatter.
    """
    img = image.detach().cpu().float()
    if img.ndim == 3:
        img = img.mean(dim=0)  # collapse replicated channels to grayscale
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    gray = (img.numpy() * 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def _index_of_name(dataset: Dataset[Sample], name: str) -> int | None:
    stems = getattr(dataset, "stems", None)
    if stems is not None and name in stems:
        return list(stems).index(name)
    return None


def _save_panel(sample: Sample, pred: torch.Tensor, score: float, out_path: Path) -> None:
    image = _denormalise_image(sample["image"])
    gt_rgb = colorize_mask(sample["mask"])
    pred_rgb = colorize_mask(pred)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, img, title in zip(
        axes,
        [image, gt_rgb, pred_rgb],
        ["Input (SAR)", "Ground truth", "Prediction"],
        strict=True,
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{sample['name']}  -  oil IoU = {score:.4f}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_prediction_gallery(
    model: ModelOrSession,
    dataset: Dataset[Sample],
    per_image_oil_iou: dict[str, float],
    out_dir: Path | str,
    *,
    n_best: int = 4,
    n_worst: int = 4,
) -> list[Path]:
    """Render ``n_best`` highest and ``n_worst`` lowest oil-IoU prediction panels.

    Parameters
    ----------
    model:
        Torch module or ONNX session (re-run for the selected images).
    dataset:
        The same dataset that produced ``per_image_oil_iou``.
    per_image_oil_iou:
        Mapping of image name to oil-class IoU (from :func:`evaluate_model`).
    out_dir:
        Directory to write the PNG panels into.
    n_best, n_worst:
        How many best/worst panels to render.

    Returns
    -------
    list[Path]
        Paths of the written panels.
    """
    out_dir = Path(out_dir)
    ranked = rank_by_oil_iou(per_image_oil_iou)
    # ranked is ascending (worst first); informative entries (non-nan) come first.
    informative = [(name, score) for name, score in ranked if score == score]
    worst = informative[:n_worst]
    best = list(reversed(informative))[:n_best]

    selections: list[tuple[str, str, float]] = []
    selections += [("worst", name, score) for name, score in worst]
    selections += [("best", name, score) for name, score in best]

    written: list[Path] = []
    for rank, (kind, name, score) in enumerate(selections):
        idx = _index_of_name(dataset, name)
        if idx is None:
            continue
        sample = dataset[idx]
        logits = predict_logits(model, sample["image"].unsqueeze(0))
        pred = logits_to_labels(logits)[0]
        out_path = out_dir / f"{kind}_{rank:02d}_{name}.png"
        _save_panel(sample, pred, score, out_path)
        written.append(out_path)
    return written


__all__ = ["save_prediction_gallery"]
