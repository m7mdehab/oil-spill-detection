"""Segmentation model definitions and the architecture registry."""

from __future__ import annotations

from oilspill.models.registry import (
    build_model,
    register_model,
    registered_architectures,
)

__all__ = ["build_model", "register_model", "registered_architectures"]
