"""Model registry: one entry point that builds any architecture from config.

``build_model`` is the single place a model is constructed, used by both the
trainer and the evaluation harness so training and checkpoint-loading always
agree on architecture. Resolution order for ``ModelConfig.arch``:

1. a name registered with :func:`register_model` (custom architectures such as
   the Hugging Face SegFormer or an EO foundation model), else
2. a ``segmentation_models_pytorch`` architecture (``Unet``, ``DeepLabV3Plus`` …).

A registered builder has the signature ``build(model_cfg, *, pretrained) -> nn.Module``
and must return a module mapping an ``(N, in_channels, H, W)`` float tensor to
``(N, num_classes, H, W)`` logits. ``pretrained`` is ``True`` for training (load
ImageNet/base weights) and ``False`` for evaluation (the trained checkpoint
weights are loaded afterwards, so downloading base weights would be wasteful).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import segmentation_models_pytorch as smp
from torch import nn

if TYPE_CHECKING:
    from oilspill.training.config import ModelConfig

ModelBuilder = Callable[["ModelConfig", bool], nn.Module]

_REGISTRY: dict[str, ModelBuilder] = {}

# Custom-architecture modules to import so their @register_model runs. Extended
# as architectures are added; importing here keeps registration side-effects in
# one discoverable place instead of scattered import-time magic.
_CUSTOM_MODULES: tuple[str, ...] = (
    "oilspill.models.segformer",
    "oilspill.models.second_arch",
    "oilspill.models.foundation",
)


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Decorator registering a builder under ``name`` (case-insensitive)."""

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        _REGISTRY[name.lower()] = builder
        return builder

    return decorator


def _load_custom_modules() -> None:
    """Import custom-architecture modules so their registrations take effect."""
    for module in _CUSTOM_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            # The module may not exist yet (architectures are added over time).
            continue


def _build_smp(model_cfg: ModelConfig, pretrained: bool) -> nn.Module:
    factory = getattr(smp, model_cfg.arch, None)
    if factory is None:
        raise ValueError(
            f"unknown architecture {model_cfg.arch!r}: not registered and not an smp model"
        )
    return factory(
        encoder_name=model_cfg.encoder,
        encoder_weights=model_cfg.encoder_weights if pretrained else None,
        in_channels=model_cfg.in_channels,
        classes=model_cfg.num_classes,
        **model_cfg.params,
    )


def build_model(model_cfg: ModelConfig, *, pretrained: bool = True) -> nn.Module:
    """Build a model from ``model_cfg`` via the registry, falling back to smp."""
    _load_custom_modules()
    builder = _REGISTRY.get(model_cfg.arch.lower())
    if builder is not None:
        return builder(model_cfg, pretrained)
    return _build_smp(model_cfg, pretrained)


def registered_architectures() -> list[str]:
    """Names of all registered custom architectures (for diagnostics/tests)."""
    _load_custom_modules()
    return sorted(_REGISTRY)
