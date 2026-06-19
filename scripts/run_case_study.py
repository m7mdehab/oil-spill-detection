"""Run the detection pipeline on the real MV Wakashio Sentinel-1 scene.

Processes the 2020-08-10 Sentinel-1B IW GRDH scene over the Wakashio oil spill
(Pointe d'Esny, SE Mauritius), cropped to the spill AOI, through the full
inference + vectorisation pipeline using the selected best model (SegFormer).

This is a reproducible driver for the case study in docs/case_study/. It uses the
uncalibrated-DN fallback (read_grd_measurement) because full sigma0 calibration via
xarray-sentinel is unavailable for this product (see the case-study writeup); the
dB normalisation window is fitted from scene percentiles to mimic the contrast of
the training imagery.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from oilspill.pipeline.detect import run_detection
from oilspill.pipeline.infer import load_session
from oilspill.pipeline.preprocess import (
    lee_filter,
    model_ready_chw,
    read_grd_measurement,
    to_db,
)

# Wakashio spill AOI (Pointe d'Esny, SE Mauritius); slick drifted NW along the coast.
DEFAULT_BBOX = (57.58, -20.52, 57.85, -20.32)  # (min_lon, min_lat, max_lon, max_lat)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Wakashio case-study detection.")
    parser.add_argument("--safe", default=None, help="Path to the .SAFE dir (auto-detected).")
    parser.add_argument("--onnx", default="artifacts/exports/segformer-mit-b2.onnx")
    parser.add_argument("--out", default="artifacts/case_study/wakashio")
    parser.add_argument("--min-area-m2", type=float, default=5000.0)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument(
        "--db-percentiles",
        type=float,
        nargs=2,
        default=(2.0, 98.0),
        help="Lower/upper percentiles of the relative-dB image used as the [0,1] window.",
    )
    args = parser.parse_args()

    safe = args.safe or next(iter(glob.glob("data/scenes/*.SAFE")), None)
    if safe is None:
        raise SystemExit("no .SAFE found under data/scenes/ (download the scene first)")

    print(f"reading VV measurement (AOI crop) from {Path(safe).name}", flush=True)
    scene = read_grd_measurement(safe, "vv", bbox=DEFAULT_BBOX)
    print(f"  AOI intensity shape {scene.sigma0.shape}", flush=True)

    # Speckle filter on intensity, then relative dB.
    filtered = lee_filter(scene.sigma0, size=7)
    db = to_db(filtered)
    finite = db[np.isfinite(db)]
    lo, hi = np.percentile(finite, args.db_percentiles)
    p_lo, p_hi = args.db_percentiles
    print(f"  relative-dB window [{lo:.2f}, {hi:.2f}] (p{p_lo}-p{p_hi})")

    scene_chw = model_ready_chw(db, in_min=float(lo), in_max=float(hi))

    session = load_session(args.onnx)
    out_dir = Path(args.out)
    result = run_detection(
        scene_chw=scene_chw,
        transform=scene.transform,
        crs=scene.crs,
        session=session,
        min_area_m2=args.min_area_m2,
        out_dir=out_dir,
        tile_size=args.tile_size,
    )

    summary = {
        "scene": Path(safe).name,
        "aoi_bbox": DEFAULT_BBOX,
        "num_oil_polygons": result.num_oil_polygons,
        "total_oil_area_km2": result.total_oil_area_km2,
        "class_mask": str(result.class_mask_path),
        "class_mask_rgb": str(result.class_mask_rgb_path),
        "oil_polygons_geojson": str(result.geojson_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== case study detection complete ===")
    print(f"oil polygons : {result.num_oil_polygons}")
    print(f"detected area: {result.total_oil_area_km2:.2f} km2 (over the AOI)")
    print(f"outputs in   : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
