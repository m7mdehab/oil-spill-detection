"""Geospatial outputs for oil-spill inference: GeoTIFF and vector polygons.

This module turns a model's per-pixel class mask (and optional oil-probability
map) into two georeferenced products:

* a GeoTIFF written with the scene's affine ``transform`` and ``crs`` so it
  drops straight into any GIS (:func:`write_geotiff`); and
* vectorised oil-spill polygons (:func:`vectorize_oil`) carrying a physically
  correct surface area in km^2 and, when an oil-probability map is supplied,
  per-polygon confidence statistics. These can be serialised to a portable
  EPSG:4326 GeoJSON with :func:`to_geojson`.

Area computation
----------------
Polygon area in km^2 is computed by :func:`polygon_area_km2` and is correct
regardless of the scene CRS:

* **Projected (metric) CRS** -- shapely's planar ``geometry.area`` is already in
  the CRS's linear unit (assumed metres for the projected CRSs used here, e.g.
  UTM), so the area is simply ``geometry.area / 1e6``.
* **Geographic CRS (degrees, e.g. EPSG:4326)** -- a planar area in square
  degrees is meaningless, so the *geodesic* area is computed on the WGS84
  ellipsoid with :meth:`pyproj.Geod.geometry_area_perimeter` (taking the
  absolute value, since orientation determines its sign) and divided by 1e6.

This keeps the round-trip area error of a synthetic square well under 2% for
both projected and geographic scenes (see ``tests/test_vectorize.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
from pyproj import Geod
from rasterio.crs import CRS
from rasterio.features import geometry_mask, shapes
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from oilspill.metrics import OIL_CLASS_INDEX

if TYPE_CHECKING:
    from affine import Affine

# Polygons smaller than this surface area are treated as speckle/noise and
# dropped by default. 5000 m^2 (0.005 km^2) is a handful of 10 m Sentinel-1
# pixels -- small enough to keep genuine slicks, large enough to remove
# single-pixel artefacts.
DEFAULT_MIN_AREA_M2: float = 5000.0

# WGS84 lon/lat, the portable CRS GeoJSON is expected to use (RFC 7946).
_WGS84_EPSG: int = 4326

_GEOD_WGS84 = Geod(ellps="WGS84")


def _as_crs(crs: CRS | PyprojCRS | str | int) -> CRS:
    """Coerce any accepted CRS spec to a :class:`rasterio.crs.CRS`."""
    if isinstance(crs, CRS):
        return crs
    if isinstance(crs, PyprojCRS):
        return CRS.from_wkt(crs.to_wkt())
    if isinstance(crs, int):
        return CRS.from_epsg(crs)
    return CRS.from_user_input(crs)


def write_geotiff(
    array: np.ndarray,
    path: Path | str,
    transform: Affine,
    crs: CRS | PyprojCRS | str | int,
    *,
    nodata: float | int | None = None,
) -> Path:
    """Write ``array`` to a GeoTIFF at ``path`` preserving ``transform`` and ``crs``.

    ``array`` may be 2-D ``(H, W)`` (written as a single band) or 3-D
    ``(bands, H, W)``. The dtype is preserved. Returns the written path.
    """
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"array must be 2-D (H, W) or 3-D (bands, H, W), got shape {arr.shape}")

    count, height, width = arr.shape
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": arr.dtype,
        "transform": transform,
        "crs": _as_crs(crs),
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)
    return out_path


def polygon_area_km2(geom: BaseGeometry, crs: CRS | PyprojCRS | str | int) -> float:
    """Return the surface area of ``geom`` in km^2, correct for any ``crs``.

    For a geographic CRS the geodesic area on the WGS84 ellipsoid is used; for a
    projected (metric) CRS the planar shapely area is used directly. See the
    module docstring for details.
    """
    pyproj_crs = PyprojCRS.from_user_input(_as_crs(crs).to_wkt())
    if pyproj_crs.is_geographic:
        # geometry_area_perimeter returns (area_m2, perimeter_m); area is signed
        # by ring orientation, so take the magnitude.
        area_m2, _ = _GEOD_WGS84.geometry_area_perimeter(geom)
        return abs(area_m2) / 1e6
    # Projected CRS: planar area is already in the CRS's (metre) units squared.
    return float(geom.area) / 1e6


def _confidence_stats(
    geom: BaseGeometry,
    oil_prob: np.ndarray,
    transform: Affine,
) -> tuple[float, float]:
    """Return ``(mean, max)`` oil probability over the pixels inside ``geom``.

    The polygon is rasterised onto the scene grid with
    :func:`rasterio.features.geometry_mask` (``all_touched`` so boundary pixels
    are included) and the oil-probability values under that mask are reduced.
    """
    height, width = oil_prob.shape
    # geometry_mask returns True *outside* the geometry; invert for "inside".
    inside = ~geometry_mask(
        [geom],
        out_shape=(height, width),
        transform=transform,
        all_touched=True,
        invert=False,
    )
    if not inside.any():
        return 0.0, 0.0
    values = oil_prob[inside]
    return float(values.mean()), float(values.max())


def vectorize_oil(
    class_mask: np.ndarray,
    transform: Affine,
    crs: CRS | PyprojCRS | str | int,
    *,
    oil_prob: np.ndarray | None = None,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
) -> gpd.GeoDataFrame:
    """Vectorise the oil-spill class of ``class_mask`` into cleaned polygons.

    Oil pixels (``class_mask == OIL_CLASS_INDEX``) are polygonised with
    :func:`rasterio.features.shapes`, cleaned (``buffer(0)`` to repair any
    self-touching rings, invalid/empty geometries dropped), and filtered to keep
    only polygons whose surface area is at least ``min_area_m2``. Area is computed
    with :func:`polygon_area_km2`, so the filter is physically meaningful for both
    projected and geographic scenes.

    The returned :class:`geopandas.GeoDataFrame` is in the scene ``crs`` with an
    ``area_km2`` column and, when ``oil_prob`` is given, ``mean_confidence`` and
    ``max_confidence`` columns (mean/max oil probability over each polygon's
    pixels). It is empty (but with the expected columns) when no oil is present.
    """
    mask = np.asarray(class_mask)
    if mask.ndim != 2:
        raise ValueError(f"class_mask must be 2-D (H, W), got shape {mask.shape}")
    scene_crs = _as_crs(crs)

    if oil_prob is not None:
        oil_prob = np.asarray(oil_prob, dtype=np.float64)
        if oil_prob.shape != mask.shape:
            raise ValueError(
                f"oil_prob shape {oil_prob.shape} must match class_mask shape {mask.shape}"
            )

    oil = (mask == OIL_CLASS_INDEX).astype(np.uint8)

    geoms: list[BaseGeometry] = []
    areas: list[float] = []
    mean_conf: list[float] = []
    max_conf: list[float] = []

    for geom_dict, value in shapes(oil, mask=oil.astype(bool), transform=transform):
        if value != 1:
            continue
        geom = shape(geom_dict)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty or not geom.is_valid:
            continue

        area_km2 = polygon_area_km2(geom, scene_crs)
        if area_km2 * 1e6 < min_area_m2:
            continue

        geoms.append(geom)
        areas.append(area_km2)
        if oil_prob is not None:
            mean_v, max_v = _confidence_stats(geom, oil_prob, transform)
            mean_conf.append(mean_v)
            max_conf.append(max_v)

    data: dict[str, list[float]] = {"area_km2": areas}
    if oil_prob is not None:
        data["mean_confidence"] = mean_conf
        data["max_confidence"] = max_conf

    gdf = gpd.GeoDataFrame(data, geometry=geoms, crs=scene_crs)
    return gdf


def to_geojson(gdf: gpd.GeoDataFrame, path: Path | str) -> Path:
    """Write ``gdf`` to a GeoJSON at ``path``, reprojected to EPSG:4326.

    Only the portable property columns ``area_km2``, ``mean_confidence`` and
    ``max_confidence`` are written (whichever are present). Returns the path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if gdf.crs is None:
        raise ValueError("gdf has no CRS; cannot reproject to EPSG:4326 for GeoJSON output")
    wgs84 = gdf.to_crs(epsg=_WGS84_EPSG)

    keep = [c for c in ("area_km2", "mean_confidence", "max_confidence") if c in wgs84.columns]
    wgs84 = wgs84[[*keep, "geometry"]]

    # GeoJSON must be EPSG:4326 (RFC 7946); write via GeoPandas' GeoJSON driver.
    wgs84.to_file(out_path, driver="GeoJSON")
    return out_path


# Re-export for callers that build a colourised GeoTIFF and want the legend in
# one import alongside the writer. (Imported lazily-friendly names; kept here so
# downstream code has a single geospatial-output entry point.)
def colorized_geotiff_from_mask(
    class_mask: np.ndarray,
    path: Path | str,
    transform: Affine,
    crs: CRS | PyprojCRS | str | int,
) -> Path:
    """Write an RGB GeoTIFF colourising ``class_mask`` with the class legend."""
    from oilspill.data.colors import colorize_mask

    rgb = colorize_mask(class_mask)  # (H, W, 3) uint8
    chw = np.moveaxis(rgb, -1, 0)  # (3, H, W)
    return write_geotiff(chw, path, transform, crs)


__all__ = [
    "DEFAULT_MIN_AREA_M2",
    "colorized_geotiff_from_mask",
    "polygon_area_km2",
    "to_geojson",
    "vectorize_oil",
    "write_geotiff",
]
