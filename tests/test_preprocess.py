"""Tests for the Sentinel-1 SAR preprocessing chain.

Fast tests exercise the radiometric chain and land masking on synthetic NumPy
arrays / geometries; the SAFE-dependent calibration test is marked ``slow`` and
skips gracefully when no SAFE product is available.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from oilspill.pipeline.preprocess import (
    DEFAULT_DB_WINDOW,
    IMAGENET_MEAN,
    IMAGENET_STD,
    calibrate_safe,
    land_mask_from_coastlines,
    lee_filter,
    model_ready_chw,
    normalize_for_model,
    to_db,
)


# --------------------------------------------------------------------------- #
# Lee speckle filter
# --------------------------------------------------------------------------- #
def test_lee_filter_reduces_variance_and_preserves_mean() -> None:
    """On a homogeneous speckled region the filter must cut variance toward the
    true mean while preserving the mean (the core Lee-filter property)."""
    rng = np.random.default_rng(0)
    true_mean = 5.0
    # Multiplicative speckle: y = x * n, n ~ unit-mean noise. Use a gamma-like
    # multiplicative gain centred at 1.0 to mimic SAR intensity speckle.
    gain = rng.gamma(shape=4.0, scale=0.25, size=(256, 256))  # mean ~ 1.0
    speckled = true_mean * gain

    filtered = lee_filter(speckled, size=7)

    in_var = float(speckled.var())
    out_var = float(filtered.var())

    # Variance is substantially reduced...
    assert out_var < in_var
    assert out_var < 0.5 * in_var
    # ...while the mean is preserved (interior, ignoring box-filter edge effects).
    interior = (slice(8, -8), slice(8, -8))
    assert float(filtered[interior].mean()) == pytest.approx(true_mean, rel=0.02)

    # Report the numbers for the record.
    print(
        f"\nLee filter: in_var={in_var:.4f} out_var={out_var:.4f} "
        f"ratio={out_var / in_var:.3f} mean={float(filtered[interior].mean()):.4f}"
    )


def test_lee_filter_preserves_constant_image() -> None:
    """A perfectly constant image has no speckle; output should equal input."""
    const = np.full((32, 32), 3.0)
    out = lee_filter(const, size=7)
    assert np.allclose(out, 3.0)


def test_lee_filter_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        lee_filter(np.zeros((4, 4, 3)))


# --------------------------------------------------------------------------- #
# dB conversion
# --------------------------------------------------------------------------- #
def test_to_db_known_values() -> None:
    arr = np.array([1.0, 10.0, 100.0, 0.1])
    out = to_db(arr)
    assert out == pytest.approx([0.0, 10.0, 20.0, -10.0])


def test_to_db_floors_zero() -> None:
    out = to_db(np.array([0.0]), eps=1e-10)
    assert out[0] == pytest.approx(-100.0)
    assert np.isfinite(out).all()


def test_to_db_rejects_nonpositive_eps() -> None:
    with pytest.raises(ValueError, match="eps"):
        to_db(np.array([1.0]), eps=0.0)


# --------------------------------------------------------------------------- #
# Model-input normalisation
# --------------------------------------------------------------------------- #
def test_normalize_for_model_range_channels_and_clipping() -> None:
    in_min, in_max = DEFAULT_DB_WINDOW
    db = np.array(
        [
            [in_min - 10.0, in_min, (in_min + in_max) / 2.0, in_max, in_max + 10.0],
        ]
    )
    out = normalize_for_model(db)

    # 3-channel output, all channels identical (single SAR signal replicated).
    assert out.shape == (1, 5, 3)
    assert np.allclose(out[..., 0], out[..., 1])
    assert np.allclose(out[..., 1], out[..., 2])

    vals = out[0, :, 0]
    # Clipping at the window edges.
    assert vals[0] == pytest.approx(0.0)  # below window
    assert vals[1] == pytest.approx(0.0)  # at in_min
    assert vals[2] == pytest.approx(0.5)  # midpoint
    assert vals[3] == pytest.approx(1.0)  # at in_max
    assert vals[4] == pytest.approx(1.0)  # above window
    # Range and dtype.
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    assert out.dtype == np.float32


def test_normalize_for_model_is_monotonic() -> None:
    db = np.linspace(-40.0, 10.0, 100)
    out = normalize_for_model(db)[..., 0]
    diffs = np.diff(out)
    assert (diffs >= -1e-7).all()  # non-decreasing


def test_normalize_for_model_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="in_min < in_max"):
        normalize_for_model(np.zeros((2, 2)), in_min=0.0, in_max=-1.0)


def test_model_ready_chw_shape_and_imagenet_normalisation() -> None:
    db = np.full((4, 6), DEFAULT_DB_WINDOW[1])  # maps to 1.0 in every channel
    chw = model_ready_chw(db)
    assert chw.shape == (3, 4, 6)
    assert chw.dtype == np.float32
    # x=1.0 -> (1 - mean) / std per channel.
    for c in range(3):
        expected = (1.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        assert chw[c].mean() == pytest.approx(expected, rel=1e-5)


def test_imagenet_constants_match_training_transforms() -> None:
    """The duplicated ImageNet constants MUST equal those used at training time."""
    pytest.importorskip("albumentations")
    from oilspill.data.transforms import IMAGENET_MEAN as TRAIN_MEAN
    from oilspill.data.transforms import IMAGENET_STD as TRAIN_STD

    assert IMAGENET_MEAN == TRAIN_MEAN
    assert IMAGENET_STD == TRAIN_STD


# --------------------------------------------------------------------------- #
# Land mask rasterisation
# --------------------------------------------------------------------------- #
def test_land_mask_from_coastlines_rasterises_polygon(tmp_path: Path) -> None:
    """Burn a tiny synthetic land polygon over a known transform and check cells."""
    gpd = pytest.importorskip("geopandas")
    from affine import Affine
    from rasterio.crs import CRS
    from shapely.geometry import Polygon

    # 4x4 grid, 1 unit per pixel, origin at (0, 4) with y decreasing downward.
    transform = Affine.translation(0.0, 4.0) * Affine.scale(1.0, -1.0)
    crs = CRS.from_epsg(4326)
    shape = (4, 4)

    # A 2x2 square covering world coords x in [0,2], y in [2,4] -> top-left cells.
    polygon = Polygon([(0.0, 2.0), (2.0, 2.0), (2.0, 4.0), (0.0, 4.0)])
    shp_path = tmp_path / "land.shp"
    gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326").to_file(shp_path)

    mask = land_mask_from_coastlines(shape, transform, crs, shp_path)

    assert mask.shape == shape
    assert mask.dtype == bool
    # Top-left 2x2 block is land; the rest is sea.
    expected = np.zeros(shape, dtype=bool)
    expected[0:2, 0:2] = True
    assert np.array_equal(mask, expected)


def test_land_mask_missing_file_raises(tmp_path: Path) -> None:
    from affine import Affine
    from rasterio.crs import CRS

    with pytest.raises(FileNotFoundError):
        land_mask_from_coastlines(
            (2, 2), Affine.identity(), CRS.from_epsg(4326), tmp_path / "nope.shp"
        )


# --------------------------------------------------------------------------- #
# SAFE calibration (slow; skips when no SAFE is available)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_calibrate_safe_end_to_end() -> None:
    """End-to-end calibration on a real S1 GRD SAFE.

    Point ``OILSPILL_TEST_SAFE`` at a ``.SAFE`` product to run this; otherwise it
    skips. The SAFE-independent parts of the chain are covered by the fast tests.
    """
    safe_env = os.environ.get("OILSPILL_TEST_SAFE")
    if not safe_env or not Path(safe_env).exists():
        pytest.skip("no SAFE product available (set OILSPILL_TEST_SAFE to run)")

    scene = calibrate_safe(safe_env, polarisation="vv")
    assert scene.sigma0.ndim == 2
    assert np.isfinite(scene.sigma0).any()
    assert (scene.sigma0 >= 0.0).all()  # linear-power sigma0 is non-negative
