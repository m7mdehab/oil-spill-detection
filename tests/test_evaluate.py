"""Tests for the evaluation and reporting harness.

The fast tests use a tiny synthetic dataset and a trivial identity-ish "model"
(an :class:`torch.nn.Module` that emits logits) so they exercise the real
``evaluate_model`` / report code paths without building a heavy ``smp`` model or
touching the real checkpoint. Tests that load the real ``smp`` model or the real
checkpoint are marked ``slow`` so the default (``-m "not slow"``) pass stays fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from oilspill.data.dataset import Sample
from oilspill.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_pr_curves,
    update_results_markdown,
    write_results_json,
)
from oilspill.evaluation.report import TABLE_BEGIN, TABLE_END
from oilspill.metrics import NUM_CLASSES, compute_metrics


class _TinyDataset(Dataset[Sample]):
    """A handful of synthetic (image, mask, name) items for evaluation tests."""

    def __init__(self, n: int = 4, size: int = 8) -> None:
        self.size = size
        self.stems = [f"img_{i:03d}" for i in range(n)]
        torch.manual_seed(0)
        self._images = [torch.rand(3, size, size) for _ in range(n)]
        self._masks = [torch.randint(0, NUM_CLASSES, (size, size)) for _ in range(n)]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> Sample:
        return {
            "image": self._images[index],
            "mask": self._masks[index],
            "name": self.stems[index],
        }


class _OracleModel(nn.Module):
    """Emits logits that argmax to a fixed per-call mask (set externally).

    Used to control predictions deterministically. Input is ignored except for its
    spatial shape; output is ``(N, C, H, W)`` one-hot-ish logits of ``self.target``.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        # A real (trainable) parameter so .parameters() is non-empty (device probe).
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        # Predict argmax of a cheap deterministic transform of the input.
        cls = (x.mean(dim=1) * self.num_classes).long().clamp(0, self.num_classes - 1)  # (N,H,W)
        logits = torch.zeros(n, self.num_classes, h, w)
        logits.scatter_(1, cls.unsqueeze(1), 10.0)
        return logits * self.scale


def test_evaluate_model_returns_sane_result() -> None:
    dataset = _TinyDataset(n=4, size=8)
    model = _OracleModel()
    output = evaluate_model(model, dataset, batch_size=2, collect_pr=True)

    result = output.result
    assert result.confusion_matrix.shape == (NUM_CLASSES, NUM_CLASSES)
    assert 0.0 <= result.pixel_accuracy <= 1.0
    assert output.num_images == 4
    assert set(output.per_image_oil_iou) == set(dataset.stems)
    # PR data collected for every (subsampled) pixel.
    assert output.pr_probs is not None
    assert output.pr_targets is not None
    assert output.pr_probs.shape[1] == NUM_CLASSES
    assert output.pr_probs.shape[0] == output.pr_targets.shape[0]


def test_evaluate_model_respects_max_images() -> None:
    dataset = _TinyDataset(n=6, size=8)
    output = evaluate_model(_OracleModel(), dataset, batch_size=4, max_images=3)
    assert output.num_images == 3
    assert len(output.per_image_oil_iou) == 3


def test_plots_write_files(tmp_path: Path) -> None:
    dataset = _TinyDataset(n=4, size=8)
    output = evaluate_model(_OracleModel(), dataset, batch_size=2, collect_pr=True)

    cm_path = plot_confusion_matrix(output.result, tmp_path / "cm.png")
    assert cm_path.exists() and cm_path.stat().st_size > 0

    assert output.pr_probs is not None and output.pr_targets is not None
    pr_path = plot_pr_curves(
        output.pr_probs, output.pr_targets, output.result.class_names, tmp_path / "pr.png"
    )
    assert pr_path.exists() and pr_path.stat().st_size > 0


def _dummy_result() -> object:
    preds = torch.randint(0, NUM_CLASSES, (1, 16, 16))
    targets = torch.randint(0, NUM_CLASSES, (1, 16, 16))
    return compute_metrics(preds, targets)


def test_write_results_json_roundtrip(tmp_path: Path) -> None:
    import json

    result = _dummy_result()
    json_path = write_results_json(result, tmp_path / "metrics.json", {"run_name": "r1"})  # type: ignore[arg-type]
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["meta"]["run_name"] == "r1"
    assert "aggregate" in payload["metrics"]
    assert "oil_iou" in payload["metrics"]["aggregate"]


def test_update_results_markdown_creates_and_updates(tmp_path: Path) -> None:
    md = tmp_path / "results.md"
    result = _dummy_result()
    json_path = tmp_path / "metrics.json"

    # First run: file is created with a header and one row.
    update_results_markdown(md, "run-A", result, json_path, tag="smoke", num_images=10)  # type: ignore[arg-type]
    text = md.read_text(encoding="utf-8")
    assert TABLE_BEGIN in text and TABLE_END in text
    assert text.count("| run-A |") == 1
    assert "auto-generated" in text

    # Re-run for the same run name: row is UPDATED, not duplicated.
    update_results_markdown(md, "run-A", result, json_path, tag="smoke", num_images=20)  # type: ignore[arg-type]
    text2 = md.read_text(encoding="utf-8")
    assert text2.count("| run-A |") == 1
    assert "| 20 |" in text2  # the updated image count
    assert "| 10 |" not in text2

    # A different run name adds a second row.
    update_results_markdown(md, "run-B", result, json_path, tag="full", num_images=110)  # type: ignore[arg-type]
    text3 = md.read_text(encoding="utf-8")
    assert text3.count("| run-A |") == 1
    assert text3.count("| run-B |") == 1


@pytest.mark.slow
def test_load_real_checkpoint_and_build_model() -> None:
    """Build the real smp model from the newest smoke checkpoint, if present."""
    import glob

    from oilspill.evaluation import load_model_from_checkpoint

    candidates = sorted(glob.glob("artifacts/checkpoints/*/best.pt"))
    if not candidates:
        pytest.skip("no checkpoint available")
    model, config = load_model_from_checkpoint(candidates[-1], device="cpu")
    assert config["model"]["num_classes"] == NUM_CLASSES
    with torch.no_grad():
        out = model(torch.rand(1, config["model"]["in_channels"], 64, 64))
    assert out.shape[1] == NUM_CLASSES
