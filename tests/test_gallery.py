"""Tests for the best/worst prediction gallery renderer.

Uses a tiny synthetic dataset and a trivial torch module so the real
:func:`save_prediction_gallery` code path (ranking, selection, panel rendering
with matplotlib's Agg backend) runs fast and writes real PNG files.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from oilspill.data.dataset import Sample
from oilspill.evaluation.gallery import (
    _denormalise_image,
    _index_of_name,
    save_prediction_gallery,
)
from oilspill.metrics import NUM_CLASSES


class _TinyDataset(Dataset[Sample]):
    def __init__(self, n: int = 6, size: int = 8) -> None:
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


class _ConstModel(nn.Module):
    """Emits constant logits so predictions are deterministic and cheap."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        logits = torch.zeros(n, NUM_CLASSES, h, w)
        logits[:, 1] = 5.0  # always predict class 1
        return logits * self.scale


def test_denormalise_image_returns_uint8_rgb() -> None:
    img = _denormalise_image(torch.rand(3, 8, 8))
    assert img.shape == (8, 8, 3)
    assert img.dtype.kind == "u"
    assert int(img.min()) >= 0 and int(img.max()) <= 255


def test_index_of_name_found_and_missing() -> None:
    ds = _TinyDataset(n=3)
    assert _index_of_name(ds, "img_001") == 1
    assert _index_of_name(ds, "does_not_exist") is None


def test_save_prediction_gallery_writes_panels(tmp_path: Path) -> None:
    ds = _TinyDataset(n=6)
    # Distinct oil-IoU scores so ranking is unambiguous; include a NaN to verify
    # non-informative entries are dropped.
    per_image = {
        "img_000": 0.10,
        "img_001": 0.20,
        "img_002": 0.30,
        "img_003": 0.40,
        "img_004": 0.50,
        "img_005": float("nan"),
    }
    written = save_prediction_gallery(_ConstModel(), ds, per_image, tmp_path, n_best=2, n_worst=2)
    # 2 worst + 2 best = 4 panels, all real PNG files.
    assert len(written) == 4
    for path in written:
        assert path.exists() and path.stat().st_size > 0
    names = {p.name for p in written}
    assert any(n.startswith("worst_") for n in names)
    assert any(n.startswith("best_") for n in names)


def test_save_prediction_gallery_skips_unknown_names(tmp_path: Path) -> None:
    ds = _TinyDataset(n=2)
    # Score references a name not in the dataset; it must be skipped, not error.
    per_image = {"img_000": 0.5, "ghost": 0.9}
    written = save_prediction_gallery(_ConstModel(), ds, per_image, tmp_path, n_best=2, n_worst=2)
    assert all(p.exists() for p in written)
    assert all("ghost" not in p.name for p in written)
