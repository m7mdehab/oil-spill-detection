"""Runtime configuration for the inference API.

All paths and the HTTP port are resolved here so they can be overridden from the
environment (or a ``.env`` file) without touching code. Every field has a sane
default that works against the repository layout, so the API runs out of the box
when launched from the project root.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API settings, overridable via ``OILSPILL_API_*`` environment variables.

    Attributes
    ----------
    onnx_dir:
        Directory holding exported ``<model-id>.onnx`` files. The default model
        (``model.onnx``) and per-model files are looked up here.
    default_onnx:
        Path to the fallback ONNX model used when no per-model export exists.
    results_dir:
        Directory of committed evaluation JSONs (one per model) used to build the
        ``/models`` response.
    samples_dir:
        Directory of preloaded sample SAR images served at ``/samples``.
    web_dist:
        Directory of the built frontend; mounted at ``/`` when it exists.
    tile_size, overlap, batch_size:
        Tiling parameters forwarded to inference.
    """

    model_config = SettingsConfigDict(
        env_prefix="OILSPILL_API_",
        env_file=".env",
        extra="ignore",
    )

    onnx_dir: Path = Path("artifacts/exports")
    default_onnx: Path = Path("artifacts/exports/model.onnx")
    results_dir: Path = Path("docs/results")
    samples_dir: Path = Path("data/samples")
    web_dist: Path = Path("web/dist")

    tile_size: int = 512
    overlap: int = 64
    batch_size: int = 4


def get_settings() -> Settings:
    """Return a freshly resolved :class:`Settings` instance."""
    return Settings()


__all__ = ["Settings", "get_settings"]
