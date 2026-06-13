"""Augmentation pipelines for SAR oil-spill segmentation.

Design notes (SAR-specific)
---------------------------
SAR pixel values are *backscatter intensity*, not optical colour. The three
stored channels are a single-channel SAR signal replicated to RGB so that
ImageNet-pretrained encoders can be used. Two consequences drive the augmentation
choices here:

1. **No photometric/colour jitter.** Hue, saturation, brightness/contrast jitter
   and channel shuffling all assume optical RGB semantics. Applied to SAR
   backscatter they are physically meaningless and actively harmful: they distort
   the radiometric values the model must learn from and break the (intentional)
   equality of the three replicated channels. We therefore deliberately exclude
   ``A.ColorJitter`` / ``A.HueSaturationValue`` / ``A.RandomBrightnessContrast``
   and any channel-wise photometric op.

2. **Speckle is multiplicative.** SAR images are corrupted by speckle, which is a
   *multiplicative* noise (``y = x * n``) arising from coherent imaging, not the
   additive Gaussian noise of optical sensors. We model augmentation speckle the
   same way via :class:`SpeckleNoise` (per-pixel multiplicative gain centred at
   1.0). Additive Gaussian noise would not match the SAR noise model.

Geometric augmentations (flips, 90-degree rotations, random crop) are safe and
useful: oil slicks have no canonical orientation. Masks are always resampled with
nearest-neighbour interpolation and are never normalised, so class indices stay
exact integers in ``{0..NUM_CLASSES-1}``.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

# ImageNet statistics: used because the SAR signal is read as 3-channel RGB to
# feed ImageNet-pretrained encoders. Configurable via the factory below for
# experiments that train from scratch on raw SAR statistics.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class SpeckleNoise(ImageOnlyTransform):
    """Multiplicative speckle-noise augmentation for SAR imagery.

    Each pixel is multiplied by an i.i.d. gain ``n ~ N(1, sigma^2)`` (clipped to
    be non-negative), i.e. ``y = x * n``. This matches the multiplicative nature
    of SAR speckle, unlike additive Gaussian noise. Applied to the image only;
    masks are untouched.

    Parameters
    ----------
    sigma_range:
        Range from which the noise standard deviation is sampled per call.
    p:
        Probability of applying the transform.
    """

    def __init__(
        self,
        sigma_range: tuple[float, float] = (0.05, 0.2),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.sigma_range = sigma_range

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        low, high = self.sigma_range
        sigma = float(np.random.uniform(low, high))
        noise = np.random.normal(loc=1.0, scale=sigma, size=img.shape).astype(np.float32)
        np.clip(noise, 0.0, None, out=noise)
        out = img.astype(np.float32) * noise
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            return np.clip(out, info.min, info.max).astype(img.dtype)
        return out.astype(img.dtype)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("sigma_range",)


def build_transforms(
    *,
    train: bool,
    image_size: tuple[int, int] = (650, 1250),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    speckle_sigma_range: tuple[float, float] = (0.05, 0.2),
) -> A.Compose:
    """Build the albumentations pipeline for a split.

    Parameters
    ----------
    train:
        If True, returns the training pipeline (geometric augmentation + speckle).
        If False, returns the deterministic val/test pipeline (resize + normalise).
    image_size:
        Target ``(height, width)`` after resize/crop.
    mean, std:
        Per-channel normalisation statistics. Default to ImageNet (3-channel SAR).
    speckle_sigma_range:
        Standard-deviation range for :class:`SpeckleNoise` (train only).

    Returns
    -------
    A.Compose
        A pipeline expecting ``image=`` (HxWx3 uint8) and ``mask=`` (HxW) inputs.
        Masks are resampled with nearest-neighbour and are never normalised.
    """
    height, width = image_size
    normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0)

    if not train:
        # Deterministic evaluation pipeline: geometry-preserving resize +
        # normalisation only. No augmentation, no photometric ops.
        return A.Compose(
            [
                A.Resize(height=height, width=width, interpolation=1),
                normalize,
            ]
        )

    return A.Compose(
        [
            # Geometric augmentation: orientation-agnostic for oil slicks.
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Random resized crop to the target size for scale/translation
            # variety; mask uses nearest-neighbour (interpolation=0).
            A.RandomResizedCrop(
                size=(height, width),
                scale=(0.6, 1.0),
                ratio=(0.75, 1.3333),
                interpolation=1,
                mask_interpolation=0,
                p=1.0,
            ),
            # SAR speckle (multiplicative). Deliberately NO colour/photometric
            # jitter -- see module docstring.
            SpeckleNoise(sigma_range=speckle_sigma_range, p=0.5),
            normalize,
        ]
    )
