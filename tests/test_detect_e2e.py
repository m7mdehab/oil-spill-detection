"""Synthetic end-to-end test for the detection pipeline (the Phase 3 gate).

This proves the *ingest-less* pipeline runs start to finish: a preprocessed,
model-ready scene flows through tiled ONNX inference and into georeferenced
vector/raster products, with no network and no real model download.

The model is a tiny *real* ONNX graph built in-process: a single 1x1 conv from
3 channels to the 5 segmentation classes, with weights chosen deterministically
so the oil class (index 1) wins wherever the input is bright and the sea-surface
class (index 0) wins on the dark background. A 1x1 conv has no receptive-field
edge effect, so a bright square in the input maps cleanly to an oil region in the
output -- a controllable stand-in for the trained model. The scene is given a
real UTM (10 m) transform so the reported polygon area can be checked against the
square's known ground-truth area.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import onnxruntime as ort
import pytest
import rasterio
import torch
from affine import Affine
from rasterio.crs import CRS

from oilspill.metrics import NUM_CLASSES, OIL_CLASS_INDEX
from oilspill.pipeline.detect import (
    CLASS_MASK_NAME,
    CLASS_MASK_RGB_NAME,
    OIL_POLYGONS_NAME,
    run_detection,
)
from oilspill.pipeline.infer import INPUT_NAME, OUTPUT_NAME, load_session

# Scene geometry: a 200x200 grid at 10 m pixels (so 1 px = 100 m^2).
SCENE_H = 200
SCENE_W = 200
PIXEL_M = 10.0
# The bright square (oil) occupies [SQ0:SQ1) on both axes.
SQ0 = 60
SQ1 = 140
SQUARE_SIDE_PX = SQ1 - SQ0
SQUARE_AREA_KM2 = (SQUARE_SIDE_PX * PIXEL_M) ** 2 / 1e6  # 0.64 km^2
# UTM zone 10N (metres), so planar polygon area is correct.
UTM10N_EPSG = 32610


class _BrightToOil(torch.nn.Module):
    """1x1 conv whose logits favour the oil class on bright pixels, sea on dark.

    Each class gets a single 1x1 channel. The oil-class kernel has large positive
    weights so its logit grows with input brightness; the sea-class kernel has a
    positive bias and zero weights so it dominates where the input is ~0. The
    remaining classes are pushed far negative so they never win.
    """

    def __init__(self) -> None:
        super().__init__()
        conv = torch.nn.Conv2d(3, NUM_CLASSES, kernel_size=1)
        weight = conv.weight
        bias = conv.bias
        assert bias is not None  # Conv2d(bias=True) always allocates one
        with torch.no_grad():
            weight.zero_()
            bias.zero_()
            # Sea surface (0): no weight, modest positive bias -> wins on dark bg.
            bias[0] = 1.0
            # Oil (1): strong response to all 3 channels, negative bias so it only
            # overtakes the sea bias once the input is clearly bright.
            weight[OIL_CLASS_INDEX] = 10.0
            bias[OIL_CLASS_INDEX] = -5.0
            # Other classes: large negative bias -> never selected.
            for c in range(NUM_CLASSES):
                if c not in (0, OIL_CLASS_INDEX):
                    bias[c] = -50.0
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@pytest.fixture(scope="module")
def onnx_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export the tiny bright->oil model to ONNX with dynamic axes."""
    torch.manual_seed(0)
    model = _BrightToOil().eval()
    out_path = tmp_path_factory.mktemp("onnx") / "tiny_oil.onnx"
    dummy = torch.randn(1, 3, 16, 16, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(out_path),
            dynamo=False,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                INPUT_NAME: {0: "batch", 2: "height", 3: "width"},
                OUTPUT_NAME: {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=17,
        )
    return out_path


@pytest.fixture(scope="module")
def session(onnx_model_path: Path) -> ort.InferenceSession:
    return load_session(onnx_model_path)


def _synthetic_scene() -> np.ndarray:
    """Model-ready (3, H, W) scene: bright square on a dark background."""
    scene = np.zeros((3, SCENE_H, SCENE_W), dtype=np.float32)
    scene[:, SQ0:SQ1, SQ0:SQ1] = 1.0
    return scene


def _utm_transform() -> Affine:
    """A north-up 10 m UTM transform (pixel (0,0) at a plausible easting/northing)."""
    return Affine(PIXEL_M, 0.0, 500000.0, 0.0, -PIXEL_M, 4000000.0)


def test_run_detection_writes_products_and_finds_oil(
    session: ort.InferenceSession, tmp_path: Path
) -> None:
    """The core gate: preprocessed scene -> infer -> vectorize, products on disk."""
    scene = _synthetic_scene()
    transform = _utm_transform()
    crs = CRS.from_epsg(UTM10N_EPSG)
    out_dir = tmp_path / "run"

    result = run_detection(
        scene_chw=scene,
        transform=transform,
        crs=crs,
        session=session,
        out_dir=out_dir,
        tile_size=64,
        overlap=16,
        batch_size=4,
    )

    # (1) All three products were written where the result says they are.
    assert result.class_mask_path == out_dir / CLASS_MASK_NAME
    assert result.class_mask_rgb_path == out_dir / CLASS_MASK_RGB_NAME
    assert result.geojson_path == out_dir / OIL_POLYGONS_NAME
    assert result.class_mask_path.exists()
    assert result.class_mask_rgb_path.exists()
    assert result.geojson_path.exists()

    # (2) The class-mask GeoTIFF is georeferenced and contains an oil region.
    with rasterio.open(result.class_mask_path) as src:
        assert src.crs == crs
        assert src.transform == transform
        mask = src.read(1)
    assert (mask == OIL_CLASS_INDEX).any(), "expected oil pixels in the class mask"
    # Oil should sit inside the bright square, not the dark background.
    assert mask[SQ0 + 5, SQ0 + 5] == OIL_CLASS_INDEX
    assert mask[5, 5] == 0

    # (3) The GeoJSON has >= 1 oil polygon located over the square.
    gdf = gpd.read_file(result.geojson_path)
    assert result.num_oil_polygons >= 1
    assert len(gdf) == result.num_oil_polygons
    centroid = gdf.to_crs(crs).geometry.union_all().centroid
    # Square centre in UTM coords from the transform.
    cx_px, cy_px = (SQ0 + SQ1) / 2, (SQ0 + SQ1) / 2
    exp_x, exp_y = transform.c + cx_px * transform.a, transform.f + cy_px * transform.e
    assert abs(centroid.x - exp_x) < 5 * PIXEL_M
    assert abs(centroid.y - exp_y) < 5 * PIXEL_M

    # (4) Reported total oil area matches the square's true area within tolerance.
    assert result.total_oil_area_km2 == pytest.approx(SQUARE_AREA_KM2, rel=0.05)


def test_land_mask_suppresses_oil(session: ort.InferenceSession, tmp_path: Path) -> None:
    """Oil falling under a land mask is removed from the outputs."""
    scene = _synthetic_scene()
    transform = _utm_transform()
    crs = CRS.from_epsg(UTM10N_EPSG)

    # Land mask covering the entire bright square -> all its oil must be suppressed.
    land = np.zeros((SCENE_H, SCENE_W), dtype=bool)
    land[SQ0:SQ1, SQ0:SQ1] = True

    result = run_detection(
        scene_chw=scene,
        transform=transform,
        crs=crs,
        session=session,
        land_mask=land,
        out_dir=tmp_path / "masked",
        tile_size=64,
        overlap=16,
        batch_size=4,
    )

    assert result.num_oil_polygons == 0
    assert result.total_oil_area_km2 == 0.0
    with rasterio.open(result.class_mask_path) as src:
        mask = src.read(1)
    assert not (mask == OIL_CLASS_INDEX).any(), "land mask should remove all oil pixels"


def test_partial_land_mask_keeps_unmasked_oil(
    session: ort.InferenceSession, tmp_path: Path
) -> None:
    """Masking only part of the square leaves the unmasked oil detectable."""
    scene = _synthetic_scene()
    transform = _utm_transform()
    crs = CRS.from_epsg(UTM10N_EPSG)

    # Mask only the left half of the square.
    land = np.zeros((SCENE_H, SCENE_W), dtype=bool)
    mid = (SQ0 + SQ1) // 2
    land[SQ0:SQ1, SQ0:mid] = True

    result = run_detection(
        scene_chw=scene,
        transform=transform,
        crs=crs,
        session=session,
        land_mask=land,
        out_dir=tmp_path / "partial",
        tile_size=64,
        overlap=16,
        batch_size=4,
    )

    assert result.num_oil_polygons >= 1
    # Remaining oil is roughly half the square's area.
    assert result.total_oil_area_km2 == pytest.approx(SQUARE_AREA_KM2 / 2, rel=0.1)


def test_land_mask_shape_mismatch_raises(session: ort.InferenceSession, tmp_path: Path) -> None:
    scene = _synthetic_scene()
    with pytest.raises(ValueError):
        run_detection(
            scene_chw=scene,
            transform=_utm_transform(),
            crs=CRS.from_epsg(UTM10N_EPSG),
            session=session,
            land_mask=np.zeros((10, 10), dtype=bool),
            out_dir=tmp_path / "bad",
        )
