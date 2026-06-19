"""Download Natural Earth land polygons for SAR scene land masking.

The Sentinel-1 preprocessing pipeline (``oilspill.pipeline.preprocess``) masks out
land so that over-land backscatter is not mistaken for an oil slick. The land
geometry comes from Natural Earth's 1:50m "land" physical vector, a public-domain
dataset that is small (sub-megabyte) and more than detailed enough at S1 GRD
resolution for coastline masking.

Source
------
Natural Earth -- 1:50m Physical Vectors -> "Land" (``ne_50m_land``).
Landing page: https://www.naturalearthdata.com/downloads/50m-physical-vectors/

The naturalearthdata.com download links are redirect-based and historically flaky,
so we fetch from the canonical AWS S3 bucket that Natural Earth itself publishes to
(used by ``cartopy``/``geopandas`` and documented on the Natural Earth site):

    https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_land.zip

Verified reachable and returning a valid Shapefile zip (~447 KB) as of 2026-06.
Natural Earth data is public domain (no licence restrictions, no attribution
required, though attribution is appreciated).

Usage
-----
    python scripts/download_coastlines.py [--dest DIR] [--force]

Writes the extracted shapefile set to ``data/coastlines/ne_50m_land/`` (gitignored)
and records the source URL alongside it in ``SOURCE.txt``.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DEST = _REPO_ROOT / "data" / "coastlines" / "ne_50m_land"

# Canonical Natural Earth S3 mirror for the 1:50m land vector. See module docstring.
SOURCE_URL = "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_land.zip"
# Fallback: the official landing page's redirect-style download link.
FALLBACK_URL = (
    "https://www.naturalearthdata.com/http//www.naturalearthdata.com/"
    "download/50m/physical/ne_50m_land.zip"
)

_SHAPEFILE_STEM = "ne_50m_land"
_REQUIRED_SUFFIXES = (".shp", ".shx", ".dbf", ".prj")


def _download_zip(url: str, *, timeout: float = 120.0) -> bytes:
    """Download ``url`` and return its bytes, raising on a non-OK response."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def download_coastlines(dest: Path = _DEFAULT_DEST, *, force: bool = False) -> Path:
    """Download and extract the Natural Earth 1:50m land shapefile to ``dest``.

    Parameters
    ----------
    dest:
        Directory to extract the shapefile set into (created if missing).
    force:
        Re-download even if the shapefile already appears present.

    Returns
    -------
    Path
        Path to the extracted ``.shp`` file.
    """
    shp_path = dest / f"{_SHAPEFILE_STEM}.shp"
    if shp_path.exists() and not force:
        print(f"Coastlines already present: {shp_path}")
        return shp_path

    dest.mkdir(parents=True, exist_ok=True)

    payload: bytes | None = None
    last_error: Exception | None = None
    for url in (SOURCE_URL, FALLBACK_URL):
        try:
            print(f"Downloading {url} ...")
            payload = _download_zip(url)
            used_url = url
            break
        except Exception as exc:
            print(f"  failed: {exc}")
            last_error = exc
    if payload is None:
        raise RuntimeError(f"All download URLs failed; last error: {last_error}")

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(dest)

    missing = [s for s in _REQUIRED_SUFFIXES if not (dest / f"{_SHAPEFILE_STEM}{s}").exists()]
    if missing:
        raise RuntimeError(f"Extracted archive is missing expected files: {missing}")

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    (dest / "SOURCE.txt").write_text(
        "Natural Earth 1:50m Physical Vectors -- Land (ne_50m_land)\n"
        f"Downloaded from: {used_url}\n"
        f"Downloaded at: {now}\n"
        "Landing page: https://www.naturalearthdata.com/downloads/50m-physical-vectors/\n"
        "Licence: public domain (Natural Earth).\n",
        encoding="utf-8",
    )
    print(f"Extracted coastlines to: {dest}")
    return shp_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=_DEFAULT_DEST,
        help=f"Destination directory (default: {_DEFAULT_DEST}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the shapefile already exists.",
    )
    args = parser.parse_args()
    path = download_coastlines(args.dest, force=args.force)
    print(f"Land vector ready: {path}")


if __name__ == "__main__":
    main()
