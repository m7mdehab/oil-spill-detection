"""Command-line entry point for end-to-end oil-spill detection.

Runs the Sentinel-1 detection pipeline and writes a class-mask GeoTIFF, a
colourised GeoTIFF and an oil-polygon GeoJSON to an output directory, then prints
a short summary (number of oil polygons and total oil area in km^2).

Three input modes (pick exactly one):

* ``--aoi``  -- search the Copernicus Data Space Ecosystem for a Sentinel-1
  scene over an area of interest and date range, download it, and detect.
  Requires CDSE credentials (``CDSE_USER`` / ``CDSE_PASS`` from the environment
  or a ``.env`` file). This is the only mode that uses the network.
* ``--safe`` -- run on an already-downloaded ``.SAFE`` product (no network).
* ``--scene`` -- run on a preprocessed model-ready scene saved as ``.npy``
  (shape ``(3, H, W)``) together with ``--scene-transform``/``--scene-epsg``
  for georeferencing (no network, no SAFE).

Examples
--------
AOI mode (search + download + detect)::

    python scripts/detect.py --aoi aoi.geojson --start 2024-01-01 --end 2024-01-31 \\
        --onnx artifacts/exports/model.onnx --out outputs/my_run

Local SAFE mode::

    python scripts/detect.py --safe path/to/S1.SAFE --onnx artifacts/exports/model.onnx \\
        --out outputs/my_run --coastlines data/coastlines/ne_land.geojson
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oilspill.pipeline.detect import (
    DetectionResult,
    detect_from_aoi,
    detect_from_safe,
    run_detection,
)
from oilspill.pipeline.infer import load_session
from oilspill.pipeline.preprocess import DEFAULT_DB_WINDOW


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the ``detect`` CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="detect",
        description=(
            "End-to-end Sentinel-1 oil-spill detection: writes a class-mask GeoTIFF, "
            "a colourised GeoTIFF and an oil-polygon GeoJSON."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--aoi",
        type=Path,
        help="GeoJSON AOI; search CDSE, download a scene, then detect (needs --start/--end).",
    )
    mode.add_argument(
        "--safe",
        type=Path,
        help="Path to a downloaded .SAFE product to detect on (no network).",
    )
    mode.add_argument(
        "--scene",
        type=Path,
        help="Preprocessed model-ready scene as .npy (3, H, W); needs --scene-epsg.",
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
        help="Path to the exported .onnx segmentation model.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for the GeoTIFFs and GeoJSON.",
    )

    # AOI-mode options.
    parser.add_argument("--start", type=str, default=None, help="AOI mode: start date YYYY-MM-DD.")
    parser.add_argument("--end", type=str, default=None, help="AOI mode: end date YYYY-MM-DD.")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="AOI mode: directory for the downloaded SAFE (default <out>/safe).",
    )

    # scene-mode georeferencing.
    parser.add_argument(
        "--scene-epsg",
        type=int,
        default=None,
        help="scene mode: EPSG code of the scene grid (e.g. 32610 for UTM 10N).",
    )
    parser.add_argument(
        "--scene-transform",
        type=float,
        nargs=6,
        default=None,
        metavar=("A", "B", "C", "D", "E", "F"),
        help="scene mode: affine transform as 6 floats (a b c d e f); default 10 m pixels.",
    )

    # Shared detection options.
    parser.add_argument(
        "--polarisation",
        type=str,
        default="vv",
        help="SAR polarisation to calibrate (SAFE/AOI modes). Default vv.",
    )
    parser.add_argument(
        "--coastlines",
        type=Path,
        default=None,
        help="Natural Earth land vector for land-mask suppression (SAFE/AOI modes).",
    )
    parser.add_argument(
        "--db-window",
        type=float,
        nargs=2,
        default=list(DEFAULT_DB_WINDOW),
        metavar=("MIN", "MAX"),
        help=f"dB window mapped to model [0,1] input. Default {DEFAULT_DB_WINDOW}.",
    )
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=5000.0,
        help="Minimum oil-polygon surface area to keep, in m^2. Default 5000.",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Inference tile size.")
    parser.add_argument("--overlap", type=int, default=64, help="Inference tile overlap.")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size.")

    return parser.parse_args(argv)


def _run_scene_mode(args: argparse.Namespace) -> DetectionResult:
    """Detect on a preprocessed model-ready .npy scene."""
    from affine import Affine
    from rasterio.crs import CRS

    if args.scene_epsg is None:
        raise SystemExit("--scene requires --scene-epsg for georeferencing.")

    scene_chw = np.load(args.scene)
    if scene_chw.ndim != 3 or scene_chw.shape[0] != 3:
        raise SystemExit(f"--scene must hold a (3, H, W) array, got shape {scene_chw.shape}.")

    if args.scene_transform is not None:
        transform = Affine(*args.scene_transform)
    else:
        # Default to 10 m pixels at the origin (Sentinel-1 GRD resolution).
        transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)
    crs = CRS.from_epsg(args.scene_epsg)

    session = load_session(args.onnx)
    return run_detection(
        scene_chw=scene_chw.astype(np.float32),
        transform=transform,
        crs=crs,
        session=session,
        min_area_m2=args.min_area_m2,
        out_dir=args.out,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the detection CLI; returns a process exit code."""
    args = parse_args(argv)
    db_window = (float(args.db_window[0]), float(args.db_window[1]))

    if args.aoi is not None:
        if not args.start or not args.end:
            raise SystemExit("--aoi mode requires --start and --end (YYYY-MM-DD).")
        # Load .env so CDSE credentials are available for the search/download.
        from dotenv import load_dotenv

        load_dotenv()
        print(f"mode       : aoi ({args.aoi})")
        print(f"date range : {args.start} .. {args.end}")
        result = detect_from_aoi(
            args.aoi,
            args.start,
            args.end,
            args.onnx,
            args.out,
            download_dir=args.download_dir,
            polarisation=args.polarisation,
            db_window=db_window,
            coastlines_path=args.coastlines,
            min_area_m2=args.min_area_m2,
        )
    elif args.safe is not None:
        print(f"mode       : safe ({args.safe})")
        result = detect_from_safe(
            args.safe,
            args.onnx,
            args.out,
            polarisation=args.polarisation,
            db_window=db_window,
            coastlines_path=args.coastlines,
            min_area_m2=args.min_area_m2,
            tile_size=args.tile_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
        )
    else:
        print(f"mode       : scene ({args.scene})")
        result = _run_scene_mode(args)

    print(f"class mask : {result.class_mask_path}")
    print(f"colourised : {result.class_mask_rgb_path}")
    print(f"polygons   : {result.geojson_path}")
    print(f"oil polygons: {result.num_oil_polygons}")
    print(f"oil area   : {result.total_oil_area_km2:.4f} km^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
