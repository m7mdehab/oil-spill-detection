"""Class colour legend for the oil-spill segmentation masks.

The RGB values below are the *authoritative* legend taken verbatim from the
dataset's own ``README.txt`` (see ``data/datasets/oil_spill/README.txt``). They
map each class id (0..4) to the colour used in the dataset's ``labels/`` RGB
masks. Visualisation and any RGB<->index conversion in the project reuse these
constants so there is a single source of truth.

Class names, the class count, and the oil class index live in
:mod:`oilspill.metrics`; this module only adds the colour mapping.
"""

from __future__ import annotations

import numpy as np
import torch

from oilspill.metrics import NUM_CLASSES

# Class id -> (R, G, B), in class-id order 0..4. From the dataset README legend:
#   0 Sea Surface (0,0,0); 1 Oil Spill (0,255,255); 2 Look-alike (255,0,0);
#   3 Ship (153,76,0); 4 Land (0,153,0).
CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),  # 0 Sea Surface
    (0, 255, 255),  # 1 Oil Spill
    (255, 0, 0),  # 2 Look-alike
    (153, 76, 0),  # 3 Ship
    (0, 153, 0),  # 4 Land
)

assert len(CLASS_COLORS) == NUM_CLASSES, "CLASS_COLORS must have one entry per class"

# Lookup table form, shape (NUM_CLASSES, 3) uint8, handy for fast indexing.
_COLOR_LUT: np.ndarray = np.asarray(CLASS_COLORS, dtype=np.uint8)


def _to_numpy_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return ``mask`` as a 2-D ``int`` numpy array of class indices."""
    arr = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D HxW class-index mask, got shape {arr.shape}")
    return arr.astype(np.int64)


def colorize_mask(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    """Map a class-index mask to an RGB image using :data:`CLASS_COLORS`.

    Parameters
    ----------
    mask:
        2-D array/tensor of integer class indices in ``{0..NUM_CLASSES-1}``
        (shape ``HxW``).

    Returns
    -------
    np.ndarray
        ``HxWx3`` ``uint8`` RGB image.
    """
    arr = _to_numpy_mask(mask)
    if arr.min() < 0 or arr.max() >= NUM_CLASSES:
        raise ValueError(
            f"mask contains class ids outside [0, {NUM_CLASSES - 1}]: "
            f"found range [{int(arr.min())}, {int(arr.max())}]"
        )
    return _COLOR_LUT[arr]


def rgb_to_class(rgb: np.ndarray | torch.Tensor) -> np.ndarray:
    """Inverse of :func:`colorize_mask`: map an RGB mask back to class indices.

    Any pixel whose colour is not in :data:`CLASS_COLORS` is assigned to the
    nearest legend colour by Euclidean distance in RGB space. Exact-match pixels
    are therefore always recovered losslessly; this only matters for masks that
    were resampled or JPEG-compressed.

    Parameters
    ----------
    rgb:
        ``HxWx3`` RGB array/tensor (uint8 or float in 0..255).

    Returns
    -------
    np.ndarray
        ``HxW`` ``int64`` array of class indices.
    """
    arr = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected an HxWx3 RGB mask, got shape {arr.shape}")
    pixels = arr.reshape(-1, 3).astype(np.int64)  # (N, 3)
    # (N, 1, 3) - (1, K, 3) -> (N, K, 3); squared distance to each legend colour.
    dist = ((pixels[:, None, :] - _COLOR_LUT[None, :, :].astype(np.int64)) ** 2).sum(axis=2)
    return dist.argmin(axis=1).reshape(arr.shape[:2]).astype(np.int64)
