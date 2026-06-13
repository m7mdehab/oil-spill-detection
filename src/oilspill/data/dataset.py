"""``OilSpillDataset`` and the deterministic train/val/test split.

Split policy
------------
The dataset ships an official ``train`` (1002) / ``test`` (110) split. The test
set is the *untouched official* test set and is never resampled. A validation set
is carved deterministically out of the 1002 training samples with a fixed seed
(default 15% -> ~150 val / ~852 train). The same seed always yields the same file
lists, and the three splits are disjoint. See :func:`make_splits`.

Item format
-----------
Each item is a dict ``{"image", "mask", "name"}``:

* ``image``: ``FloatTensor[C, H, W]`` (C=3 by default; see below).
* ``mask``:  ``LongTensor[H, W]`` with class indices in ``{0..NUM_CLASSES-1}``.
* ``name``:  the filename stem, e.g. ``"img_0001"``.

Channels
--------
The SAR signal is a single channel replicated to 3-channel RGB on disk. We keep
the 3-channel read because the downstream encoders are ImageNet-pretrained and
expect 3 input channels; collapsing to 1 channel would forfeit pretrained stem
weights. Normalisation/augmentation are configured accordingly in
:mod:`oilspill.data.transforms`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from oilspill.metrics import NUM_CLASSES

Split = Literal["train", "val", "test"]


class Sample(TypedDict):
    """One dataset item: a normalised image, its mask, and the filename stem."""

    image: torch.Tensor  # FloatTensor[C, H, W]
    mask: torch.Tensor  # LongTensor[H, W]
    name: str


# Fixed seed for the deterministic val carve-out from the official train split.
DEFAULT_SPLIT_SEED: int = 1337
# Fraction of the official 1002-sample train split held out for validation.
DEFAULT_VAL_FRACTION: float = 0.15

_IMAGE_SUBDIR = "images"
_LABEL_SUBDIR = "labels_1D"
_IMAGE_EXT = ".jpg"
_LABEL_EXT = ".png"


@dataclass(frozen=True)
class DatasetSplits:
    """Deterministic file-stem lists for each split (sorted within a split)."""

    train: list[str]
    val: list[str]
    test: list[str]


def _list_stems(directory: Path, ext: str) -> list[str]:
    """Return sorted filename stems of ``*<ext>`` files in ``directory``."""
    if not directory.is_dir():
        raise FileNotFoundError(f"missing dataset directory: {directory}")
    return sorted(p.stem for p in directory.iterdir() if p.suffix.lower() == ext)


def make_splits(
    root: Path | str,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> DatasetSplits:
    """Compute the deterministic train/val/test file-stem lists.

    The official ``test`` split is returned untouched. ``val`` is a deterministic
    ``val_fraction`` random subset of the official ``train`` stems (seeded), and
    ``train`` is the remainder. Calling this twice with the same arguments returns
    identical lists; the three lists are pairwise disjoint.

    Parameters
    ----------
    root:
        Dataset root containing ``train/`` and ``test/`` subdirectories.
    val_fraction:
        Fraction of the official train split to hold out for validation.
    seed:
        RNG seed controlling the val carve-out.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    root = Path(root)

    train_stems = _list_stems(root / "train" / _LABEL_SUBDIR, _LABEL_EXT)
    test_stems = _list_stems(root / "test" / _LABEL_SUBDIR, _LABEL_EXT)

    n_val = round(len(train_stems) * val_fraction)
    # Seeded, order-independent shuffle of a sorted copy -> reproducible.
    rng = random.Random(seed)
    shuffled = list(train_stems)
    rng.shuffle(shuffled)
    val_set = set(shuffled[:n_val])

    train_split = sorted(s for s in train_stems if s not in val_set)
    val_split = sorted(val_set)
    return DatasetSplits(train=train_split, val=val_split, test=test_stems)


class OilSpillDataset(Dataset[Sample]):
    """Torch ``Dataset`` over the oil-spill SAR images and 1-D label masks.

    Parameters
    ----------
    root:
        Dataset root containing ``train/`` and ``test/`` subdirectories.
    split:
        One of ``"train"``, ``"val"``, ``"test"``. ``train``/``val`` read from the
        official ``train/`` folder (carved by :func:`make_splits`); ``test`` reads
        the official ``test/`` folder.
    transform:
        Optional albumentations pipeline (see
        :func:`oilspill.data.transforms.build_transforms`). It must accept
        ``image=`` and ``mask=`` and return normalised images. If ``None``, the
        raw uint8 image is returned scaled to ``[0, 1]`` with no augmentation.
    val_fraction, seed:
        Forwarded to :func:`make_splits` for the deterministic val carve-out.
    """

    def __init__(
        self,
        root: Path | str,
        split: Split,
        *,
        transform: A.Compose | None = None,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        seed: int = DEFAULT_SPLIT_SEED,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.root = Path(root)
        self.split: Split = split
        self.transform = transform

        splits = make_splits(self.root, val_fraction=val_fraction, seed=seed)
        self.stems: list[str] = getattr(splits, split)

        # train/val both live under the official train/ folder.
        source = "test" if split == "test" else "train"
        self._image_dir = self.root / source / _IMAGE_SUBDIR
        self._label_dir = self.root / source / _LABEL_SUBDIR

    def __len__(self) -> int:
        return len(self.stems)

    def _load_image(self, stem: str) -> np.ndarray:
        """Load a SAR image as HxWx3 uint8 RGB (single channel replicated on disk)."""
        with Image.open(self._image_dir / f"{stem}{_IMAGE_EXT}") as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def _load_mask(self, stem: str) -> np.ndarray:
        """Load a 1-D class-index mask as HxW int64 with values in {0..C-1}."""
        with Image.open(self._label_dir / f"{stem}{_LABEL_EXT}") as msk:
            mask = np.asarray(msk.convert("L"), dtype=np.int64)
        if mask.min() < 0 or mask.max() >= NUM_CLASSES:
            raise ValueError(
                f"mask {stem} has class ids outside [0, {NUM_CLASSES - 1}]: "
                f"range [{int(mask.min())}, {int(mask.max())}]"
            )
        return mask

    def __getitem__(self, index: int) -> Sample:
        stem = self.stems[index]
        image = self._load_image(stem)
        mask = self._load_mask(stem)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image_arr = np.asarray(out["image"], dtype=np.float32)
            mask_arr = np.asarray(out["mask"], dtype=np.int64)
        else:
            # No transform: scale to [0, 1], keep raw mask. Useful for analysis.
            image_arr = image.astype(np.float32) / 255.0
            mask_arr = mask

        # HWC -> CHW float tensor; HW long tensor.
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_arr.transpose(2, 0, 1)))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_arr)).long()
        return {"image": image_tensor, "mask": mask_tensor, "name": stem}
