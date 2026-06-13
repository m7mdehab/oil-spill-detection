"""Dataset wiring for training.

The real dataset is owned by :mod:`oilspill.data`. This module isolates the
trainer's coupling to that package behind :func:`load_real_datasets`, and
provides a synthetic fallback so the smoke run and unit tests never depend on the
data package being present or finalised.

``oilspill.data`` interface used here
-------------------------------------
``from oilspill.data import OilSpillDataset, build_transforms`` where

    build_transforms(*, train: bool, image_size: tuple[int, int], ...) -> A.Compose
    OilSpillDataset(root, split, *, transform=..., val_fraction=..., seed=...)

``split`` is ``"train" | "val" | "test"`` and each item is a mapping
``{"image": float32 (C, H, W), "mask": int64 (H, W), "name": str}``.
:class:`_DictToTupleDataset` adapts that to the ``(image, mask)`` tuples the
trainer's loops expect.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from oilspill.metrics import NUM_CLASSES

if TYPE_CHECKING:
    from oilspill.data import Sample

logger = logging.getLogger(__name__)

SegItem = tuple[torch.Tensor, torch.Tensor]
SegDataset = Dataset[SegItem]


class SyntheticSegmentationDataset(Dataset[SegItem]):
    """Deterministic in-memory dataset of random images and label masks.

    Used as a fallback for the smoke run and as fixed test data. Generation is
    seeded per-index so the dataset is reproducible across processes.
    """

    def __init__(
        self,
        length: int = 8,
        image_size: int = 64,
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
        seed: int = 0,
    ) -> None:
        self.length = length
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> SegItem:
        if not 0 <= index < self.length:
            raise IndexError(index)
        gen = torch.Generator().manual_seed(self.seed * 1000 + index)
        image = torch.rand(self.in_channels, self.image_size, self.image_size, generator=gen)
        mask = torch.randint(
            0,
            self.num_classes,
            (self.image_size, self.image_size),
            generator=gen,
            dtype=torch.int64,
        )
        return image, mask


class _DictToTupleDataset(Dataset[SegItem]):
    """Adapt an ``{"image", "mask", ...}`` dataset to ``(image, mask)`` tuples."""

    def __init__(self, base: Dataset[Sample]) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> SegItem:
        item = self.base[index]
        image = item["image"]
        mask = item["mask"]
        assert isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor)
        return image, mask.long()


def load_real_datasets(
    root: Path | str, image_size: int, val_fraction: float, seed: int
) -> tuple[SegDataset, SegDataset] | None:
    """Build train/val datasets from :mod:`oilspill.data`, or ``None`` on failure.

    Returns ``None`` (and logs a warning) if the data package or dataset root is
    unavailable so callers can fall back to synthetic data.
    """
    try:
        from oilspill.data import OilSpillDataset, build_transforms
    except (ImportError, AttributeError):
        logger.warning("oilspill.data API unavailable; using synthetic data")
        return None

    try:
        train_tf = build_transforms(train=True, image_size=(image_size, image_size))
        val_tf = build_transforms(train=False, image_size=(image_size, image_size))
        train_base = OilSpillDataset(
            root, "train", transform=train_tf, val_fraction=val_fraction, seed=seed
        )
        val_base = OilSpillDataset(
            root, "val", transform=val_tf, val_fraction=val_fraction, seed=seed
        )
    except Exception:
        logger.warning("OilSpillDataset construction failed; using synthetic data", exc_info=True)
        return None

    return _DictToTupleDataset(train_base), _DictToTupleDataset(val_base)
