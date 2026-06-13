"""Verify and extract the oil-spill dataset archive.

The dataset is distributed as a zip in ``data/raw/`` and is not committed. This
module verifies the archive against the recorded SHA-256 in
``data/checksums.sha256`` and extracts it (idempotently) to
``data/datasets/oil_spill/``. :func:`prepare_dataset` is the single entry point
used by the loader and the analysis CLI.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

# Resolve repo paths relative to this file: src/oilspill/data/extract.py ->
# repo root is four parents up.
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_DEFAULT_RAW_DIR: Path = _REPO_ROOT / "data" / "raw"
_DEFAULT_DATASETS_DIR: Path = _REPO_ROOT / "data" / "datasets"
_CHECKSUMS_FILE: Path = _REPO_ROOT / "data" / "checksums.sha256"

ARCHIVE_NAME: str = "oil_spill_dataset.zip"
DATASET_DIRNAME: str = "oil_spill"

# Files/folders that must exist for an extraction to be considered complete.
_REQUIRED_ENTRIES: tuple[str, ...] = (
    "train/images",
    "train/labels_1D",
    "test/images",
    "test/labels_1D",
)


def _read_expected_sha256(archive_name: str, checksums_file: Path = _CHECKSUMS_FILE) -> str | None:
    """Return the recorded SHA-256 for ``archive_name`` from the checksums file.

    The file uses the ``sha256sum`` format: ``<hex>  <name>`` (the name may be
    prefixed with ``*`` for binary mode). Returns ``None`` if not found.
    """
    if not checksums_file.is_file():
        return None
    for line in checksums_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, name = parts
        name = name.lstrip("*").strip()
        if name == archive_name:
            return digest.lower()
    return None


def compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of ``path`` (streamed in chunks)."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_archive(
    archive_path: Path,
    checksums_file: Path = _CHECKSUMS_FILE,
) -> bool:
    """Verify ``archive_path`` against its recorded SHA-256.

    Raises
    ------
    FileNotFoundError
        If the archive does not exist.
    ValueError
        If no checksum is recorded for the archive, or the digest mismatches.
    """
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    expected = _read_expected_sha256(archive_path.name, checksums_file)
    if expected is None:
        raise ValueError(f"no SHA-256 recorded for {archive_path.name} in {checksums_file}")
    actual = compute_sha256(archive_path)
    if actual.lower() != expected:
        raise ValueError(
            f"checksum mismatch for {archive_path.name}: expected {expected}, got {actual}"
        )
    return True


def _is_extracted(dataset_root: Path) -> bool:
    """True if every required entry is present under ``dataset_root``."""
    return dataset_root.is_dir() and all(
        (dataset_root / entry).is_dir() for entry in _REQUIRED_ENTRIES
    )


def _extract_zip(archive_path: Path, dest_dir: Path) -> None:
    """Extract ``archive_path`` into ``dest_dir`` safely (no path traversal)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.namelist():
            target = (dest_dir / member).resolve()
            if not str(target).startswith(str(dest_dir.resolve())):
                raise ValueError(f"unsafe path in archive: {member}")
        zf.extractall(dest_dir)


def prepare_dataset(
    root: Path | str | None = None,
    *,
    archive_path: Path | str | None = None,
    verify: bool = True,
    force: bool = False,
) -> Path:
    """Ensure the oil-spill dataset is extracted and return its root directory.

    Idempotent: if the dataset is already extracted (and ``force`` is False),
    extraction is skipped. The archive checksum is verified before extraction
    when ``verify`` is True.

    Parameters
    ----------
    root:
        Directory that should contain the ``oil_spill/`` dataset folder.
        Defaults to ``data/datasets/`` at the repo root.
    archive_path:
        Path to the zip archive. Defaults to ``data/raw/oil_spill_dataset.zip``.
    verify:
        Whether to verify the archive's SHA-256 before extracting.
    force:
        Re-extract even if the dataset already appears present.

    Returns
    -------
    Path
        The dataset root, i.e. ``<root>/oil_spill``.
    """
    datasets_dir = Path(root) if root is not None else _DEFAULT_DATASETS_DIR
    dataset_root = datasets_dir / DATASET_DIRNAME

    if _is_extracted(dataset_root) and not force:
        return dataset_root

    archive = Path(archive_path) if archive_path is not None else _DEFAULT_RAW_DIR / ARCHIVE_NAME
    if verify:
        verify_archive(archive)
    _extract_zip(archive, datasets_dir)

    if not _is_extracted(dataset_root):
        raise RuntimeError(
            f"extraction completed but expected entries are missing under {dataset_root}"
        )
    return dataset_root
