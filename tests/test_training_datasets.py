"""Tests for training dataset wiring and the model registry.

Covers the synthetic fallback dataset, the dict->tuple adapter, the
``load_real_datasets`` fallback behaviour when the data package/root is
unavailable, and the registry's resolution + error paths. All fast and CPU-only.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from oilspill.data.dataset import Sample
from oilspill.metrics import NUM_CLASSES
from oilspill.models import build_model, register_model, registered_architectures
from oilspill.models.registry import _build_smp
from oilspill.training.config import ModelConfig
from oilspill.training.datasets import (
    SyntheticSegmentationDataset,
    _DictToTupleDataset,
    load_real_datasets,
)


def test_synthetic_dataset_shapes_and_determinism() -> None:
    ds = SyntheticSegmentationDataset(length=4, image_size=16, in_channels=3)
    assert len(ds) == 4
    image, mask = ds[0]
    assert image.shape == (3, 16, 16)
    assert mask.shape == (16, 16)
    assert mask.dtype == torch.int64
    assert int(mask.max()) < NUM_CLASSES
    # Deterministic per index.
    image2, mask2 = ds[0]
    assert torch.equal(image, image2) and torch.equal(mask, mask2)


def test_synthetic_dataset_index_bounds() -> None:
    ds = SyntheticSegmentationDataset(length=2, image_size=8)
    with pytest.raises(IndexError):
        ds[5]
    with pytest.raises(IndexError):
        ds[-1]


class _DictDataset(Dataset[Sample]):
    def __init__(self) -> None:
        self._items = [
            {
                "image": torch.rand(3, 8, 8),
                "mask": torch.randint(0, NUM_CLASSES, (8, 8)).float(),
                "name": f"x_{i}",
            }
            for i in range(3)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Sample:
        return self._items[index]  # type: ignore[return-value]


def test_dict_to_tuple_adapter_casts_mask_to_long() -> None:
    base = _DictDataset()
    adapted = _DictToTupleDataset(base)
    assert len(adapted) == 3
    image, mask = adapted[1]
    assert image.shape == (3, 8, 8)
    assert mask.dtype == torch.int64  # cast from float


def test_load_real_datasets_returns_none_on_bad_root() -> None:
    # A non-existent dataset root makes OilSpillDataset construction fail, so the
    # helper logs a warning and returns None (the synthetic fallback signal).
    result = load_real_datasets(
        "definitely/not/a/real/dataset/root", image_size=64, val_fraction=0.2, seed=0
    )
    assert result is None


def test_registry_lists_custom_architectures() -> None:
    archs = registered_architectures()
    assert isinstance(archs, list)
    # segformer is registered via _CUSTOM_MODULES import side effects.
    assert "segformer" in archs


def test_register_model_roundtrip() -> None:
    sentinel = nn.Identity()

    @register_model("test_arch_xyz")
    def _builder(model_cfg: ModelConfig, pretrained: bool) -> nn.Module:
        return sentinel

    assert "test_arch_xyz" in registered_architectures()
    built = build_model(ModelConfig(arch="test_arch_xyz"), pretrained=False)
    assert built is sentinel


def test_build_smp_unknown_arch_raises() -> None:
    with pytest.raises(ValueError, match="unknown architecture"):
        _build_smp(ModelConfig(arch="NotARealSmpModel"), pretrained=False)
