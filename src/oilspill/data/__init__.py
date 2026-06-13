"""Dataset loading, preprocessing, and augmentation for SAR imagery."""

from __future__ import annotations

from oilspill.data.colors import CLASS_COLORS, colorize_mask, rgb_to_class
from oilspill.data.dataset import (
    DatasetSplits,
    OilSpillDataset,
    Sample,
    Split,
    make_splits,
)
from oilspill.data.extract import prepare_dataset, verify_archive
from oilspill.data.transforms import SpeckleNoise, build_transforms

__all__ = [
    "CLASS_COLORS",
    "DatasetSplits",
    "OilSpillDataset",
    "Sample",
    "SpeckleNoise",
    "Split",
    "build_transforms",
    "colorize_mask",
    "make_splits",
    "prepare_dataset",
    "rgb_to_class",
    "verify_archive",
]
