"""Sentinel-1 SAR preprocessing for oil-spill inference.

This module turns a raw Sentinel-1 GRD SAFE product into a model-ready tensor and
provides the land mask used to suppress over-land false positives. Each step is a
small, independently testable function operating on NumPy arrays (or xarray for the
SAFE-reading step), so the radiometric chain can be unit-tested without a real SAFE:

    SAFE -> calibrated sigma0  (``calibrate_safe``, xarray-sentinel)
         -> Lee speckle filter (``lee_filter``)
         -> decibels            (``to_db``)
         -> model-input mapping (``normalize_for_model`` / ``model_ready_chw``)

Land masking is handled separately by :func:`land_mask_from_coastlines`, which
rasterises Natural Earth land polygons (downloaded by ``scripts/download_coastlines.py``)
onto the scene grid.

Radiometric / training-distribution note
-----------------------------------------
The segmentation model in this project was trained on the MKLab oil-spill dataset,
whose SAR scenes are distributed as 8-bit JPGs (a single SAR channel replicated to
RGB) and then ImageNet-normalised by ``oilspill.data.transforms.build_transforms``.
The *exact* preprocessing that produced those JPGs is undocumented. To run the
trained model on freshly calibrated Sentinel-1 data we must approximate that mapping:
calibrated sigma0 in dB is linearly windowed into ``[0, 1]`` (the 8-bit-equivalent
range of the training JPGs) and replicated to three channels, *before* the model's
own ImageNet normalisation. The dB window is the key tunable -- see
:func:`normalize_for_model`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.ndimage import uniform_filter

if TYPE_CHECKING:
    import xarray as xr
    from affine import Affine
    from rasterio.crs import CRS

# ImageNet statistics, mirrored from ``oilspill.data.transforms`` so this pipeline
# stays usable even if torchvision/albumentations are absent. These MUST match the
# constants the model was trained with; they are duplicated (not imported) only to
# keep this module dependency-light, and are asserted-equal by the test suite.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Default dB window for ocean Sentinel-1 VV backscatter. Open water sits low
# (roughly -25..-15 dB at typical incidence angles, lower still for the dark
# slicks oil produces) while bright targets (ships, land, breaking waves)
# saturate near 0 dB. This window is the principal knob for matching the
# training-data histogram -- see :func:`normalize_for_model`.
DEFAULT_DB_WINDOW: tuple[float, float] = (-25.0, 0.0)

# Floor for ``to_db`` so that zero/negative intensities do not produce -inf.
DEFAULT_DB_EPS: float = 1e-10


class CalibratedScene(NamedTuple):
    """Calibrated sigma0 with the georeferencing needed for masking/inference.

    Attributes
    ----------
    sigma0:
        Linear-power calibrated sigma-nought as a NumPy array, shape ``(H, W)``.
    transform:
        Affine pixel-to-CRS transform for the array.
    crs:
        Coordinate reference system of ``transform``.
    """

    sigma0: np.ndarray
    transform: Affine
    crs: CRS


def lee_filter(image: np.ndarray, size: int = 7) -> np.ndarray:
    r"""Apply the classic Lee adaptive speckle filter.

    Lee (1980) models SAR speckle as *multiplicative* noise and derives the
    minimum-mean-square-error estimate of the true reflectivity within a local
    window. For a window centred on each pixel the estimate is

    .. math::

        \hat{x} = \bar{y} + W \, (y - \bar{y}),

    where the adaptive weight is

    .. math::

        W = \frac{\mathrm{Var}_{\text{signal}}}{\mathrm{Var}(y)}, \qquad
        \mathrm{Var}_{\text{signal}} =
            \frac{\mathrm{Var}(y) - \bar{y}^2 \, C_u^2}{1 + C_u^2},

    with the noise coefficient of variation :math:`C_u` estimated from the whole
    image as ``std/mean``. In homogeneous regions ``Var(y)`` reflects pure speckle,
    so ``W -> 0`` and the output approaches the local mean (strong smoothing). Near
    edges/point targets ``Var(y)`` is large, ``W -> 1`` and the pixel is preserved.

    The filter is *mean-preserving* on homogeneous regions (it interpolates between
    the local mean and the observation) while reducing variance -- this is the
    behaviour the unit test asserts.

    Parameters
    ----------
    image:
        2-D backscatter image (linear power or amplitude), any float dtype.
    size:
        Side length of the square moving window (odd, default 7).

    Returns
    -------
    np.ndarray
        Filtered image, same shape as ``image``, dtype ``float64``.
    """
    if image.ndim != 2:
        raise ValueError(f"lee_filter expects a 2-D array, got shape {image.shape}")
    if size < 1:
        raise ValueError(f"window size must be >= 1, got {size}")

    img = np.asarray(image, dtype=np.float64)

    # Local mean and local mean-of-squares -> local variance, via a box filter.
    local_mean = uniform_filter(img, size=size)
    local_sqr_mean = uniform_filter(img**2, size=size)
    local_var = local_sqr_mean - local_mean**2
    local_var = np.maximum(local_var, 0.0)  # guard tiny negatives from rounding

    # Global noise variation coefficient C_u for the multiplicative model.
    global_mean = float(img.mean())
    global_var = float(img.var())
    cu2 = global_var / (global_mean**2) if global_mean != 0.0 else 0.0

    # Estimated signal variance and adaptive weight. Where the local variance is
    # at or below the speckle floor, weight collapses to 0 (full smoothing).
    signal_var = (local_var - (local_mean**2) * cu2) / (1.0 + cu2)
    signal_var = np.maximum(signal_var, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(local_var > 0.0, signal_var / local_var, 0.0)
    weight = np.clip(weight, 0.0, 1.0)
    # Suppress NaNs that can appear where the local variance is exactly zero.
    weight = np.where(np.isfinite(weight), weight, 0.0)

    return local_mean + weight * (img - local_mean)


def to_db(sigma0: np.ndarray, eps: float = DEFAULT_DB_EPS) -> np.ndarray:
    """Convert linear-power sigma0 to decibels: ``10 * log10(max(sigma0, eps))``.

    Parameters
    ----------
    sigma0:
        Linear-power backscatter (non-negative). Values at or below ``eps`` are
        floored to ``eps`` so the result is finite.
    eps:
        Positive floor applied before the logarithm (default ``1e-10`` -> -100 dB).

    Returns
    -------
    np.ndarray
        Backscatter in dB, dtype ``float64``.
    """
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    arr = np.asarray(sigma0, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(arr, eps))


def normalize_for_model(
    db: np.ndarray,
    *,
    in_min: float = DEFAULT_DB_WINDOW[0],
    in_max: float = DEFAULT_DB_WINDOW[1],
) -> np.ndarray:
    """Map SAR dB into the model's expected 3-channel ``[0, 1]`` input range.

    The model was trained on 8-bit JPG SAR images (single SAR channel replicated
    to RGB), which ``oilspill.data.transforms`` rescales to ``[0, 1]`` (dividing by
    255) and then ImageNet-normalises. To feed freshly calibrated Sentinel-1 data
    to that model we reproduce the ``[0, 1]`` 8-bit-equivalent stage here:

        ``out = clip((db - in_min) / (in_max - in_min), 0, 1)``

    and replicate the single SAR channel to three channels. The result is exactly
    the distribution the model expects *before* its own ImageNet normalisation
    (apply that with :func:`model_ready_chw`).

    The ``[in_min, in_max]`` dB window IS the key approximation. The original JPGs'
    dB-to-byte mapping is undocumented, so this linear window is a best-effort match;
    ``in_min``/``in_max`` are the tunables. To validate, run this on a calibrated
    scene and compare its ``[0, 1]`` histogram against a histogram of the training
    JPG pixel values (``pixel/255``); adjust the window until the bulk ocean mode
    and the dark-slick tail line up. The default
    :data:`DEFAULT_DB_WINDOW` ``(-25, 0)`` dB is a sensible starting point for ocean
    S1 VV but should be re-tuned per training set.

    Parameters
    ----------
    db:
        Backscatter in dB, shape ``(H, W)`` (or any shape; a trailing channel axis
        is added).
    in_min, in_max:
        dB window mapped to ``0`` and ``1`` respectively. Must satisfy
        ``in_min < in_max``.

    Returns
    -------
    np.ndarray
        ``(H, W, 3)`` float32 array in ``[0, 1]``, monotonic non-decreasing in the
        input dB and clipped at the window edges.
    """
    if not in_min < in_max:
        raise ValueError(f"require in_min < in_max, got in_min={in_min}, in_max={in_max}")
    arr = np.asarray(db, dtype=np.float64)
    scaled = (arr - in_min) / (in_max - in_min)
    scaled = np.clip(scaled, 0.0, 1.0).astype(np.float32)
    return np.repeat(scaled[..., np.newaxis], 3, axis=-1)


def model_ready_chw(
    db: np.ndarray,
    *,
    in_min: float = DEFAULT_DB_WINDOW[0],
    in_max: float = DEFAULT_DB_WINDOW[1],
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Produce a model-ready CHW tensor: dB -> ``[0, 1]`` -> ImageNet-normalised.

    Combines :func:`normalize_for_model` with the same ImageNet normalisation that
    ``oilspill.data.transforms.build_transforms`` applies during training
    (``(x - mean) / std`` per channel, with ``x`` already in ``[0, 1]``). The
    ImageNet constants are re-applied here from this module's
    :data:`IMAGENET_MEAN` / :data:`IMAGENET_STD`, which mirror the training
    constants exactly (the test suite asserts they match
    ``oilspill.data.transforms``).

    Parameters
    ----------
    db:
        Backscatter in dB, shape ``(H, W)``.
    in_min, in_max:
        dB window, see :func:`normalize_for_model`.
    mean, std:
        Per-channel normalisation statistics (default ImageNet).

    Returns
    -------
    np.ndarray
        ``(3, H, W)`` float32 tensor ready for the model.
    """
    hwc = normalize_for_model(db, in_min=in_min, in_max=in_max)  # (H, W, 3) in [0,1]
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    normed = (hwc - mean_arr) / std_arr
    return np.transpose(normed, (2, 0, 1)).astype(np.float32)


def calibrate_safe(
    safe_path: str | Path,
    *,
    polarisation: str = "vv",
) -> CalibratedScene:
    """Read a Sentinel-1 GRD SAFE and return calibrated sigma0 (linear power).

    Calibration approach
    --------------------
    Sentinel-1 Level-1 GRD products store digital numbers (DN) plus per-product
    calibration look-up tables. Calibrated sigma-nought is ``|DN|^2 / sigmaNought_LUT^2``
    (radiometric calibration to sigma0). We use ``xarray-sentinel`` to read both:

    * ``open_sentinel1_dataset(safe, group="<POL>")`` -> the measurement DataArray
      of digital numbers, carrying ground-range/azimuth coordinates;
    * ``open_sentinel1_dataset(safe, group="<POL>/calibration")`` -> the calibration
      group, whose ``sigmaNought`` LUT we pass to
      :func:`xarray_sentinel.calibrate_intensity` to obtain linear-power sigma0.

    This is radiometric calibration only (no terrain flattening); for the flat ocean
    scenes this pipeline targets, terrain correction via ``sarsen`` is unnecessary.
    The georeferencing (affine transform + CRS) is recovered from the measurement
    DataArray via rioxarray so the result aligns with :func:`land_mask_from_coastlines`.

    This function requires a real SAFE on disk to run; the rest of the radiometric
    chain (Lee/to_db/normalize) is fully testable on synthetic arrays without one.

    Parameters
    ----------
    safe_path:
        Path to a ``.SAFE`` directory (or supported zip) for an S1 GRD product.
    polarisation:
        Polarisation channel, e.g. ``"vv"`` or ``"vh"`` (case-insensitive).

    Returns
    -------
    CalibratedScene
        Linear-power sigma0 array with its affine transform and CRS.
    """
    import xarray_sentinel as xs

    safe = Path(safe_path)
    if not safe.exists():
        raise FileNotFoundError(f"SAFE product not found: {safe}")

    pol = polarisation.upper()

    measurement = xs.open_sentinel1_dataset(str(safe), group=pol)
    calibration = xs.open_sentinel1_dataset(str(safe), group=f"{pol}/calibration")

    dn = measurement["measurement"]
    sigma0_da: xr.DataArray = xs.calibrate_intensity(dn, calibration["sigmaNought"])

    transform, crs = _georef_from_dataarray(sigma0_da)
    sigma0 = np.asarray(sigma0_da.values, dtype=np.float64)
    return CalibratedScene(sigma0=sigma0, transform=transform, crs=crs)


def _georef_from_dataarray(da: xr.DataArray) -> tuple[Affine, CRS]:
    """Recover an affine transform and CRS from a (rio)xarray DataArray.

    Tries rioxarray's accessor first; falls back to an identity transform in a
    geographic CRS if georeferencing is absent (e.g. raw ground-range coordinates).
    """
    import rioxarray  # noqa: F401  (registers the .rio accessor)
    from affine import Affine
    from rasterio.crs import CRS

    try:
        transform = da.rio.transform(recalc=True)
        crs = da.rio.crs
        if crs is not None:
            return transform, crs
    except Exception:
        pass
    return Affine.identity(), CRS.from_epsg(4326)


def land_mask_from_coastlines(
    shape: tuple[int, int],
    transform: Affine,
    crs: CRS,
    coastlines_path: str | Path,
) -> np.ndarray:
    """Rasterise Natural Earth land polygons onto the scene grid.

    Reads land polygons from ``coastlines_path`` (a Natural Earth ``ne_*_land``
    shapefile/GeoJSON, fetched by ``scripts/download_coastlines.py``), reprojects
    them to the scene ``crs`` if needed, and burns them onto a ``shape`` grid with
    the given ``transform`` using :func:`rasterio.features.rasterize`.

    Parameters
    ----------
    shape:
        Output grid shape ``(H, W)``.
    transform:
        Affine pixel-to-CRS transform of the scene grid.
    crs:
        CRS of ``transform`` (target CRS for the polygons).
    coastlines_path:
        Path to the Natural Earth land vector file.

    Returns
    -------
    np.ndarray
        Boolean ``(H, W)`` mask; ``True`` where the cell falls on land.
    """
    import geopandas as gpd
    from rasterio.crs import CRS
    from rasterio.features import rasterize
    from shapely.geometry import mapping

    path = Path(coastlines_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Coastlines file not found: {path}. Run scripts/download_coastlines.py first."
        )

    target_crs = crs if isinstance(crs, CRS) else CRS.from_user_input(crs)

    gdf = gpd.read_file(path)
    # Reproject to the scene CRS so polygons line up with the pixel grid.
    if gdf.crs is not None:
        gdf = gdf.to_crs(target_crs.to_wkt())

    geometries = [mapping(geom) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        return np.zeros(shape, dtype=bool)

    burned = rasterize(
        ((g, 1) for g in geometries),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return np.asarray(burned, dtype=bool)
