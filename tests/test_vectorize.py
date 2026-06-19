"""Tests for the geospatial-output stage (``oilspill.pipeline.vectorize``).

These exercise the full mask -> GeoTIFF / vector-polygon round-trip on small
synthetic scenes, including the headline acceptance check: a square of oil pixels
on a known transform must vectorise to a polygon whose area is within 2% of the
true area, for both a projected and a geographic CRS.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from pyproj import Geod
from rasterio.crs import CRS

from oilspill.metrics import OIL_CLASS_INDEX
from oilspill.pipeline.vectorize import (
    polygon_area_km2,
    to_geojson,
    vectorize_oil,
    write_geotiff,
)

# A non-oil background class and a look-alike, so the mask is genuinely
# multi-class and we are not just vectorising "everything".
_BACKGROUND = 0
_LOOKALIKE = 2


def _square_mask(size: int, square: int, top_left: tuple[int, int]) -> np.ndarray:
    """An ``size x size`` class mask with a ``square x square`` oil block."""
    mask = np.full((size, size), _BACKGROUND, dtype=np.uint8)
    r, c = top_left
    mask[r : r + square, c : c + square] = OIL_CLASS_INDEX
    return mask


# --------------------------------------------------------------------------- #
# Acceptance: <2% area round-trip, projected CRS (UTM, 10 m pixels)
# --------------------------------------------------------------------------- #
def test_area_roundtrip_projected_within_2pct() -> None:
    pixel_m = 10.0
    square_px = 20  # 20 px * 10 m = 200 m side -> 40000 m^2 -> 0.04 km^2
    size = 64
    mask = _square_mask(size, square_px, (10, 12))

    # UTM-like north-up transform: x increases east, y decreases south.
    origin_x, origin_y = 500_000.0, 4_000_000.0
    transform = Affine(pixel_m, 0.0, origin_x, 0.0, -pixel_m, origin_y)
    crs = CRS.from_epsg(32633)

    gdf = vectorize_oil(mask, transform, crs)
    assert len(gdf) == 1

    true_km2 = (square_px * pixel_m) ** 2 / 1e6
    got_km2 = float(gdf["area_km2"].iloc[0])
    rel_err = abs(got_km2 - true_km2) / true_km2
    assert rel_err < 0.02, f"projected area error {rel_err:.4%} (true {true_km2}, got {got_km2})"


# --------------------------------------------------------------------------- #
# Acceptance: <2% area round-trip, geographic CRS (EPSG:4326, geodesic)
# --------------------------------------------------------------------------- #
def test_area_roundtrip_geographic_within_2pct() -> None:
    # Small square in degrees near the equator-ish mid-latitude.
    deg = 0.001  # pixel size in degrees
    square_px = 20
    size = 64
    lon0, lat0 = 12.0, 45.0  # top-left corner longitude / latitude
    mask = _square_mask(size, square_px, (10, 12))

    # Geographic transform: x = lon increasing east, y = lat decreasing south.
    transform = Affine(deg, 0.0, lon0, 0.0, -deg, lat0)
    crs = CRS.from_epsg(4326)

    gdf = vectorize_oil(mask, transform, crs)
    assert len(gdf) == 1
    got_km2 = float(gdf["area_km2"].iloc[0])

    # Independently compute the geodesic area of the same square's footprint.
    c = 12  # column of the square's left edge (top_left col)
    r = 10  # row of the square's top edge
    west = lon0 + c * deg
    east = lon0 + (c + square_px) * deg
    north = lat0 - r * deg
    south = lat0 - (r + square_px) * deg
    geod = Geod(ellps="WGS84")
    lons = [west, east, east, west]
    lats = [north, north, south, south]
    area_m2, _ = geod.polygon_area_perimeter(lons, lats)
    true_km2 = abs(area_m2) / 1e6

    rel_err = abs(got_km2 - true_km2) / true_km2
    assert rel_err < 0.02, f"geodesic area error {rel_err:.4%} (true {true_km2}, got {got_km2})"


def test_polygon_area_km2_geographic_vs_projected_helper() -> None:
    """The helper must dispatch on CRS type and agree with hand computation."""
    from shapely.geometry import box

    # Projected: a 100 m x 100 m box -> 0.01 km^2.
    geom_proj = box(0.0, 0.0, 100.0, 100.0)
    assert polygon_area_km2(geom_proj, CRS.from_epsg(32633)) == pytest.approx(0.01, rel=1e-9)

    # Geographic: same shape in degrees must NOT be treated as planar.
    geom_geo = box(12.0, 45.0, 12.001, 45.001)
    geod = Geod(ellps="WGS84")
    expected, _ = geod.geometry_area_perimeter(geom_geo)
    assert polygon_area_km2(geom_geo, CRS.from_epsg(4326)) == pytest.approx(
        abs(expected) / 1e6, rel=1e-9
    )


# --------------------------------------------------------------------------- #
# Min-area filter + cleanup
# --------------------------------------------------------------------------- #
def test_min_area_filter_drops_speck_keeps_square() -> None:
    pixel_m = 10.0
    size = 64
    mask = _square_mask(size, 20, (5, 5))
    # A 1-pixel speck of oil far from the square (100 m^2).
    mask[60, 60] = OIL_CLASS_INDEX
    # A 2-pixel speck (200 m^2).
    mask[40, 40] = OIL_CLASS_INDEX
    mask[40, 41] = OIL_CLASS_INDEX

    transform = Affine(pixel_m, 0.0, 0.0, 0.0, -pixel_m, 0.0)
    crs = CRS.from_epsg(32633)

    # Threshold between the specks (<=200 m^2) and the square (40000 m^2).
    gdf = vectorize_oil(mask, transform, crs, min_area_m2=1000.0)
    assert len(gdf) == 1
    assert float(gdf["area_km2"].iloc[0]) == pytest.approx(0.04, rel=0.02)

    # With no threshold all three components survive.
    gdf_all = vectorize_oil(mask, transform, crs, min_area_m2=0.0)
    assert len(gdf_all) == 3


def test_no_oil_returns_empty_with_columns() -> None:
    mask = np.full((16, 16), _LOOKALIKE, dtype=np.uint8)
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    gdf = vectorize_oil(mask, transform, CRS.from_epsg(32633), oil_prob=np.zeros((16, 16)))
    assert len(gdf) == 0
    for col in ("area_km2", "mean_confidence", "max_confidence"):
        assert col in gdf.columns


# --------------------------------------------------------------------------- #
# Confidence statistics
# --------------------------------------------------------------------------- #
def test_confidence_stats_reflect_oil_prob() -> None:
    size = 32
    square = 10
    top_left = (8, 8)
    mask = _square_mask(size, square, top_left)

    oil_prob = np.zeros((size, size), dtype=np.float32)
    r, c = top_left
    # Known probabilities inside the square: a known mean and max.
    oil_prob[r : r + square, c : c + square] = 0.6
    oil_prob[r, c] = 0.95  # one high-confidence pixel -> drives the max

    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    gdf = vectorize_oil(mask, transform, CRS.from_epsg(32633), oil_prob=oil_prob)
    assert len(gdf) == 1

    n = square * square
    expected_mean = (0.6 * (n - 1) + 0.95) / n
    assert float(gdf["mean_confidence"].iloc[0]) == pytest.approx(expected_mean, rel=1e-3)
    assert float(gdf["max_confidence"].iloc[0]) == pytest.approx(0.95, rel=1e-6)


# --------------------------------------------------------------------------- #
# GeoTIFF round-trip
# --------------------------------------------------------------------------- #
def test_write_geotiff_preserves_transform_crs_data(tmp_path: Path) -> None:
    import rasterio

    rng = np.random.default_rng(0)
    array = rng.integers(0, 5, size=(24, 32), dtype=np.uint8)
    transform = Affine(10.0, 0.0, 600_000.0, 0.0, -10.0, 5_000_000.0)
    crs = CRS.from_epsg(32633)

    out = write_geotiff(array, tmp_path / "mask.tif", transform, crs, nodata=255)
    assert out.exists()

    with rasterio.open(out) as src:
        assert src.count == 1
        assert src.crs == crs
        assert src.transform == transform
        assert src.nodata == 255
        np.testing.assert_array_equal(src.read(1), array)


def test_write_geotiff_multiband(tmp_path: Path) -> None:
    import rasterio

    array = np.zeros((3, 8, 8), dtype=np.uint8)
    array[0] = 1
    array[1] = 2
    array[2] = 3
    transform = Affine(5.0, 0.0, 0.0, 0.0, -5.0, 0.0)
    out = write_geotiff(array, tmp_path / "rgb.tif", transform, CRS.from_epsg(32633))
    with rasterio.open(out) as src:
        assert src.count == 3
        np.testing.assert_array_equal(src.read(), array)


# --------------------------------------------------------------------------- #
# GeoJSON output
# --------------------------------------------------------------------------- #
def test_to_geojson_is_wgs84_with_properties(tmp_path: Path) -> None:
    pixel_m = 10.0
    mask = _square_mask(48, 20, (5, 5))
    oil_prob = np.zeros((48, 48), dtype=np.float32)
    oil_prob[5:25, 5:25] = 0.8

    transform = Affine(pixel_m, 0.0, 500_000.0, 0.0, -pixel_m, 4_000_000.0)
    crs = CRS.from_epsg(32633)
    gdf = vectorize_oil(mask, transform, crs, oil_prob=oil_prob)

    out = to_geojson(gdf, tmp_path / "spills.geojson")
    assert out.exists()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 1

    feat = payload["features"][0]
    props = feat["properties"]
    assert set(props) == {"area_km2", "mean_confidence", "max_confidence"}
    assert props["mean_confidence"] == pytest.approx(0.8, rel=1e-3)
    assert props["max_confidence"] == pytest.approx(0.8, rel=1e-6)
    assert props["area_km2"] == pytest.approx(0.04, rel=0.02)

    # Coordinates must be lon/lat (EPSG:4326): roughly near the reprojected box.
    coords = feat["geometry"]["coordinates"][0]
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    assert all(-180.0 <= lon <= 180.0 for lon in lons)
    assert all(-90.0 <= lat <= 90.0 for lat in lats)
    # UTM 33N easting 500000 sits on the central meridian (15 deg E); near 36 N.
    assert 14.0 < min(lons) < 16.0
    assert 35.0 < min(lats) < 37.0
