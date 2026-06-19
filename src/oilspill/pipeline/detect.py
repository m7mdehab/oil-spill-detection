"""End-to-end oil-spill detection orchestration.

This module wires the four pipeline stages -- :mod:`oilspill.pipeline.ingest`,
:mod:`oilspill.pipeline.preprocess`, :mod:`oilspill.pipeline.infer` and
:mod:`oilspill.pipeline.vectorize` -- into a single detection workflow and is the
function the ``detect`` CLI drives. Three entry points layer on top of one
another so each is independently usable and testable:

* :func:`run_detection` -- the network-free core. Given an *already
  preprocessed* model-ready scene (a ``(3, H, W)`` ImageNet-normalised tensor),
  its geotransform/CRS and an open ONNX session, it runs tiled inference,
  optionally suppresses oil over land, writes the class-mask GeoTIFF, a
  colourised GeoTIFF and the oil-polygon GeoJSON, and returns a
  :class:`DetectionResult` with the output paths and summary statistics. It
  touches neither the network nor a real SAFE, which is what makes the synthetic
  end-to-end test possible.
* :func:`detect_from_safe` -- the full *local* chain from a downloaded SAFE:
  calibrate -> Lee filter -> dB -> model-ready tensor -> :func:`run_detection`,
  optionally building a land mask from a coastlines file. Network-free given a
  local SAFE (but needs the real product, so any test of it is ``slow``).
* :func:`detect_from_aoi` -- the complete pipeline: search the Copernicus Data
  Space Ecosystem for a Sentinel-1 scene over an AOI/date range, download it,
  then run :func:`detect_from_safe`. This is the only entry point that uses the
  network.

Output products (written to ``out_dir``)
----------------------------------------
* ``class_mask.tif`` -- single-band uint8 GeoTIFF of the per-pixel class index.
* ``class_mask_rgb.tif`` -- RGB GeoTIFF colourised with the class legend.
* ``oil_polygons.geojson`` -- vectorised oil polygons in EPSG:4326 with their
  ``area_km2`` and (since an oil-probability map is always available here)
  per-polygon confidence statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from oilspill.metrics import OIL_CLASS_INDEX
from oilspill.pipeline.infer import load_session, tiled_predict
from oilspill.pipeline.preprocess import (
    DEFAULT_DB_WINDOW,
    calibrate_safe,
    land_mask_from_coastlines,
    lee_filter,
    model_ready_chw,
    to_db,
)
from oilspill.pipeline.vectorize import (
    DEFAULT_MIN_AREA_M2,
    colorized_geotiff_from_mask,
    to_geojson,
    vectorize_oil,
    write_geotiff,
)

if TYPE_CHECKING:
    from datetime import datetime

    import onnxruntime as ort
    from affine import Affine
    from rasterio.crs import CRS

    from oilspill.pipeline.ingest import HttpSession

# Standard output filenames written into ``out_dir``.
CLASS_MASK_NAME = "class_mask.tif"
CLASS_MASK_RGB_NAME = "class_mask_rgb.tif"
OIL_POLYGONS_NAME = "oil_polygons.geojson"


@dataclass(frozen=True)
class DetectionResult:
    """Outputs and summary statistics of one detection run.

    Attributes
    ----------
    class_mask_path:
        Single-band uint8 GeoTIFF of the per-pixel class index.
    class_mask_rgb_path:
        RGB GeoTIFF colourising the class mask with the dataset legend.
    geojson_path:
        Oil-spill polygons (EPSG:4326 GeoJSON).
    num_oil_polygons:
        Number of oil polygons retained after the minimum-area filter.
    total_oil_area_km2:
        Combined surface area of the retained oil polygons, in km^2.
    """

    class_mask_path: Path
    class_mask_rgb_path: Path
    geojson_path: Path
    num_oil_polygons: int
    total_oil_area_km2: float


def run_detection(
    *,
    scene_chw: np.ndarray,
    transform: Affine,
    crs: CRS,
    session: ort.InferenceSession,
    land_mask: np.ndarray | None = None,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
    out_dir: Path | str,
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
) -> DetectionResult:
    """Run inference + vectorisation on a preprocessed scene and write products.

    This is the network-free core of the pipeline. It takes an already
    model-ready scene tensor (ImageNet-normalised ``(3, H, W)``), runs
    :func:`oilspill.pipeline.infer.tiled_predict`, optionally suppresses oil
    pixels that fall on land, writes the class-mask GeoTIFF, a colourised
    GeoTIFF and the oil-polygon GeoJSON into ``out_dir``, and returns a
    :class:`DetectionResult`.

    Parameters
    ----------
    scene_chw:
        Model-ready, ImageNet-normalised scene of shape ``(3, H, W)`` float32.
    transform, crs:
        Affine pixel-to-CRS transform and CRS of the scene grid; carried through
        unchanged to the inference outputs and the written products.
    session:
        An open :class:`onnxruntime.InferenceSession` for the segmentation model.
    land_mask:
        Optional boolean ``(H, W)`` mask, ``True`` over land. Where ``True`` the
        oil class is overwritten with the sea-surface class (id 0) and the oil
        probability is zeroed, so over-land false positives never reach the
        vectoriser.
    min_area_m2:
        Minimum oil-polygon surface area to keep (smaller polygons are dropped as
        speckle).
    out_dir:
        Directory for the output products; created if absent.
    tile_size, overlap, batch_size:
        Tiling parameters forwarded to :func:`tiled_predict`.

    Returns
    -------
    DetectionResult
        Output paths and summary statistics.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    class_mask, oil_prob = tiled_predict(
        scene_chw,
        session,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
    )

    if land_mask is not None:
        land = np.asarray(land_mask, dtype=bool)
        if land.shape != class_mask.shape:
            raise ValueError(
                f"land_mask shape {land.shape} must match scene shape {class_mask.shape}"
            )
        # Suppress oil over land: reclassify oil-on-land as sea surface (class 0)
        # and zero its probability so it cannot be vectorised.
        oil_on_land = land & (class_mask == OIL_CLASS_INDEX)
        class_mask = class_mask.copy()
        oil_prob = oil_prob.copy()
        class_mask[oil_on_land] = 0
        oil_prob[land] = 0.0

    class_mask_path = write_geotiff(class_mask, out_path / CLASS_MASK_NAME, transform, crs)
    class_mask_rgb_path = colorized_geotiff_from_mask(
        class_mask, out_path / CLASS_MASK_RGB_NAME, transform, crs
    )

    gdf = vectorize_oil(
        class_mask,
        transform,
        crs,
        oil_prob=oil_prob,
        min_area_m2=min_area_m2,
    )
    geojson_path = to_geojson(gdf, out_path / OIL_POLYGONS_NAME)

    total_area = float(gdf["area_km2"].sum()) if len(gdf) else 0.0
    return DetectionResult(
        class_mask_path=class_mask_path,
        class_mask_rgb_path=class_mask_rgb_path,
        geojson_path=geojson_path,
        num_oil_polygons=len(gdf),
        total_oil_area_km2=total_area,
    )


def detect_from_safe(
    safe_path: Path | str,
    onnx_path: Path | str,
    out_dir: Path | str,
    *,
    polarisation: str = "vv",
    db_window: tuple[float, float] = DEFAULT_DB_WINDOW,
    lee_size: int = 7,
    coastlines_path: Path | str | None = None,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
) -> DetectionResult:
    """Run the full local detection chain on a downloaded Sentinel-1 SAFE.

    Calibrates the SAFE to sigma0, applies the Lee speckle filter, converts to
    dB, maps to the model-ready tensor, then calls :func:`run_detection`. When
    ``coastlines_path`` is given a land mask is built from it and passed through
    to suppress over-land detections. Requires a real SAFE on disk but performs
    no network access.

    Parameters
    ----------
    safe_path:
        Path to a downloaded ``.SAFE`` (directory or zip) for an S1 GRD product.
    onnx_path:
        Path to the exported ``.onnx`` segmentation model.
    out_dir:
        Directory for the output products.
    polarisation:
        SAR polarisation channel to calibrate (default ``"vv"``).
    db_window:
        ``(in_min, in_max)`` dB window mapped to the model's ``[0, 1]`` input.
    lee_size:
        Window size for the Lee speckle filter.
    coastlines_path:
        Optional Natural Earth land vector for land-mask suppression.
    min_area_m2:
        Minimum oil-polygon area to keep.
    tile_size, overlap, batch_size:
        Tiling parameters forwarded to inference.

    Returns
    -------
    DetectionResult
        Output paths and summary statistics.
    """
    scene = calibrate_safe(safe_path, polarisation=polarisation)
    filtered = lee_filter(scene.sigma0, size=lee_size)
    db = to_db(filtered)
    scene_chw = model_ready_chw(db, in_min=db_window[0], in_max=db_window[1])

    land_mask: np.ndarray | None = None
    if coastlines_path is not None:
        land_mask = land_mask_from_coastlines(
            scene.sigma0.shape,  # type: ignore[arg-type]
            scene.transform,
            scene.crs,
            coastlines_path,
        )

    session = load_session(onnx_path)
    return run_detection(
        scene_chw=scene_chw,
        transform=scene.transform,
        crs=scene.crs,
        session=session,
        land_mask=land_mask,
        min_area_m2=min_area_m2,
        out_dir=out_dir,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
    )


def detect_from_aoi(
    aoi_path: Path | str,
    start: datetime | str,
    end: datetime | str,
    onnx_path: Path | str,
    out_dir: Path | str,
    *,
    download_dir: Path | str | None = None,
    user: str | None = None,
    password: str | None = None,
    polarisation: str = "vv",
    db_window: tuple[float, float] = DEFAULT_DB_WINDOW,
    coastlines_path: Path | str | None = None,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
    session: HttpSession | None = None,
) -> DetectionResult:
    """Run the complete pipeline for an AOI: search, download, then detect.

    Searches the Copernicus Data Space Ecosystem for Sentinel-1 GRD scenes over
    the AOI and date range, downloads the most recent match, and runs
    :func:`detect_from_safe` on it. This is the only entry point that touches the
    network; CDSE credentials are read from ``user``/``password`` or, when those
    are ``None``, the ``CDSE_USER`` / ``CDSE_PASS`` environment (or ``.env``).

    Parameters
    ----------
    aoi_path:
        GeoJSON file describing the area of interest (a Polygon).
    start, end:
        Inclusive ``ContentDate/Start`` search bounds (datetime or ISO string).
    onnx_path:
        Path to the exported ``.onnx`` segmentation model.
    out_dir:
        Directory for the output products.
    download_dir:
        Directory for the downloaded SAFE zip (defaults to ``<out_dir>/safe``).
    user, password:
        CDSE credentials; fall back to the environment when ``None``.
    polarisation, db_window, coastlines_path, min_area_m2:
        Forwarded to :func:`detect_from_safe`.
    session:
        Optional HTTP session for the CDSE calls (used to inject a mock in tests).

    Returns
    -------
    DetectionResult
        Output paths and summary statistics.
    """
    from oilspill.pipeline.ingest import (
        download_product,
        get_access_token,
        load_aoi,
        search_products,
    )

    aoi = load_aoi(aoi_path)
    products = search_products(aoi, start, end, polarisation=polarisation.upper(), session=session)
    if not products:
        raise RuntimeError(f"No Sentinel-1 scenes found for the AOI between {start} and {end}.")
    product = products[0]

    safe_dir = Path(download_dir) if download_dir is not None else Path(out_dir) / "safe"
    token = get_access_token(user, password, session=session)
    safe_path = download_product(product, safe_dir, token, session=session)

    return detect_from_safe(
        safe_path,
        onnx_path,
        out_dir,
        polarisation=polarisation,
        db_window=db_window,
        coastlines_path=coastlines_path,
        min_area_m2=min_area_m2,
    )


__all__ = [
    "CLASS_MASK_NAME",
    "CLASS_MASK_RGB_NAME",
    "OIL_POLYGONS_NAME",
    "DetectionResult",
    "detect_from_aoi",
    "detect_from_safe",
    "run_detection",
]
