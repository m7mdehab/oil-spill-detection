"""Tests for the dataset module: loading, splits, augmentation, and colours."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from oilspill.data import (
    CLASS_COLORS,
    OilSpillDataset,
    build_transforms,
    colorize_mask,
    make_splits,
)
from oilspill.data.colors import rgb_to_class
from oilspill.metrics import CLASS_NAMES, NUM_CLASSES

_REAL_ROOT = Path(__file__).resolve().parents[1] / "data" / "datasets" / "oil_spill"


# --------------------------------------------------------------------------- #
# Synthetic fixtures (fast; no dependency on the full dataset)
# --------------------------------------------------------------------------- #
def _make_synthetic_root(tmp_path: Path, n_train: int = 20, n_test: int = 5) -> Path:
    """Create a tiny on-disk dataset mirroring the real layout.

    Train and test stems use disjoint numbering here only so the test/train
    disjointness assertion is meaningful on synthetic data; in the real dataset
    train and test live in separate folders and may reuse the same stems.
    """
    root = tmp_path / "oil_spill"
    rng = np.random.default_rng(0)
    for split, n, start in (("train", n_train, 1), ("test", n_test, 1001)):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels_1D"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i in range(start, start + n):
            stem = f"img_{i:04d}"
            img = rng.integers(0, 256, size=(16, 24, 3), dtype=np.uint8)
            Image.fromarray(img, mode="RGB").save(img_dir / f"{stem}.jpg")
            mask = rng.integers(0, NUM_CLASSES, size=(16, 24), dtype=np.uint8)
            Image.fromarray(mask, mode="L").save(lbl_dir / f"{stem}.png")
    return root


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    return _make_synthetic_root(tmp_path)


# --------------------------------------------------------------------------- #
# Split determinism
# --------------------------------------------------------------------------- #
def test_split_determinism(synthetic_root: Path) -> None:
    a = make_splits(synthetic_root, seed=1337)
    b = make_splits(synthetic_root, seed=1337)
    assert a.train == b.train
    assert a.val == b.val
    assert a.test == b.test


def test_splits_disjoint_and_cover_train(synthetic_root: Path) -> None:
    s = make_splits(synthetic_root)
    train_set, val_set, test_set = set(s.train), set(s.val), set(s.test)
    # train/val partition the official train folder; test is separate.
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set == set(s.train) | set(s.val)
    assert len(s.train) + len(s.val) == 20  # n_train in the fixture


def test_different_seed_changes_val(synthetic_root: Path) -> None:
    a = make_splits(synthetic_root, seed=1)
    b = make_splits(synthetic_root, seed=2)
    assert a.val != b.val


# --------------------------------------------------------------------------- #
# Dataset item: shape / dtype / value range
# --------------------------------------------------------------------------- #
def test_item_shapes_and_dtypes(synthetic_root: Path) -> None:
    ds = OilSpillDataset(synthetic_root, "train")
    item = ds[0]
    image = item["image"]
    mask = item["mask"]
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.dtype == torch.float32
    assert image.ndim == 3 and image.shape[0] == 3  # CHW, 3 channels
    assert mask.dtype == torch.long
    assert mask.shape == image.shape[1:]  # spatial dims match
    assert isinstance(item["name"], str)


def test_mask_value_range(synthetic_root: Path) -> None:
    ds = OilSpillDataset(synthetic_root, "train")
    for i in range(len(ds)):
        mask = ds[i]["mask"]
        assert int(mask.min()) >= 0
        assert int(mask.max()) < NUM_CLASSES


def test_test_split_uses_official_folder(synthetic_root: Path) -> None:
    ds = OilSpillDataset(synthetic_root, "test")
    assert len(ds) == 5  # n_test in the fixture


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def test_train_transform_keeps_mask_integer_and_in_range(synthetic_root: Path) -> None:
    tf = build_transforms(train=True, image_size=(16, 24))
    ds = OilSpillDataset(synthetic_root, "train", transform=tf)
    for _ in range(5):  # several draws to exercise the random ops
        item = ds[0]
        image, mask = item["image"], item["mask"]
        assert image.shape == (3, 16, 24)
        assert mask.shape == (16, 24)
        assert mask.dtype == torch.long
        # nearest-neighbour resampling -> exact integer class ids only.
        uniq = torch.unique(mask)
        assert int(uniq.min()) >= 0 and int(uniq.max()) < NUM_CLASSES


def test_eval_transform_is_resize_only(synthetic_root: Path) -> None:
    tf = build_transforms(train=False, image_size=(32, 40))
    ds = OilSpillDataset(synthetic_root, "val", transform=tf)
    item = ds[0]
    assert item["image"].shape == (3, 32, 40)
    assert item["mask"].shape == (32, 40)


def test_speckle_noise_is_multiplicative_identity_at_zero_sigma() -> None:
    from oilspill.data.transforms import SpeckleNoise

    img = np.full((8, 8, 3), 100, dtype=np.uint8)
    tf = SpeckleNoise(sigma_range=(0.0, 0.0), p=1.0)
    out = tf(image=img)["image"]
    # zero-sigma multiplicative noise == multiply by 1.0 == identity.
    assert np.array_equal(out, img)


# --------------------------------------------------------------------------- #
# Colours
# --------------------------------------------------------------------------- #
def test_colorize_mask_shape_dtype() -> None:
    mask = np.zeros((4, 6), dtype=np.int64)
    rgb = colorize_mask(mask)
    assert rgb.shape == (4, 6, 3)
    assert rgb.dtype == np.uint8


def test_colorize_mask_class_colors_roundtrip() -> None:
    ids = np.arange(NUM_CLASSES, dtype=np.int64).reshape(1, NUM_CLASSES)
    rgb = colorize_mask(ids)
    for i in range(NUM_CLASSES):
        assert tuple(int(v) for v in rgb[0, i]) == CLASS_COLORS[i]
    # inverse recovers the class ids exactly.
    recovered = rgb_to_class(rgb)
    assert np.array_equal(recovered, ids)


def test_colorize_mask_accepts_tensor() -> None:
    mask = torch.zeros((3, 3), dtype=torch.long)
    rgb = colorize_mask(mask)
    assert rgb.shape == (3, 3, 3) and rgb.dtype == np.uint8


def test_class_colors_count_matches_classes() -> None:
    assert len(CLASS_COLORS) == NUM_CLASSES == len(CLASS_NAMES)


# --------------------------------------------------------------------------- #
# Real-data tests (skipped gracefully if the dataset is not extracted)
# --------------------------------------------------------------------------- #
_HAS_REAL_DATA = (_REAL_ROOT / "train" / "labels_1D").is_dir()
real_data = pytest.mark.skipif(not _HAS_REAL_DATA, reason="extracted dataset not present")


@real_data
def test_real_official_test_split_size() -> None:
    s = make_splits(_REAL_ROOT)
    assert len(s.test) == 110
    assert len(s.train) + len(s.val) == 1002


@real_data
def test_real_item_loads() -> None:
    tf = build_transforms(train=False, image_size=(320, 320))
    ds = OilSpillDataset(_REAL_ROOT, "test", transform=tf)
    item = ds[0]
    assert item["image"].shape == (3, 320, 320)
    assert item["mask"].shape == (320, 320)
    assert item["mask"].dtype == torch.long
    assert int(item["mask"].min()) >= 0 and int(item["mask"].max()) < NUM_CLASSES
