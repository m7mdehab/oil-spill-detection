"""Dataset loading, preprocessing, and augmentation for SAR imagery.

The colour helpers are imported eagerly (they only need NumPy). The dataset and
augmentation symbols are exposed lazily via :pep:`562` ``__getattr__`` so that
importing this package for inference (e.g. ``oilspill.data.colors``) does not pull
in the augmentation stack (``albumentations``). This keeps the serving runtime and
its container image lean; the heavy modules load on first attribute access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from oilspill.data.colors import CLASS_COLORS, colorize_mask, rgb_to_class

if TYPE_CHECKING:
    from oilspill.data.dataset import (
        DatasetSplits,
        OilSpillDataset,
        Sample,
        Split,
        make_splits,
    )
    from oilspill.data.extract import prepare_dataset, verify_archive
    from oilspill.data.transforms import SpeckleNoise, build_transforms

# Symbol -> defining submodule, imported lazily on first access.
_LAZY: dict[str, str] = {
    "OilSpillDataset": "oilspill.data.dataset",
    "DatasetSplits": "oilspill.data.dataset",
    "Sample": "oilspill.data.dataset",
    "Split": "oilspill.data.dataset",
    "make_splits": "oilspill.data.dataset",
    "prepare_dataset": "oilspill.data.extract",
    "verify_archive": "oilspill.data.extract",
    "SpeckleNoise": "oilspill.data.transforms",
    "build_transforms": "oilspill.data.transforms",
}


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


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
