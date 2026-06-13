"""Load trained models for evaluation, from a torch checkpoint or an ONNX file.

The evaluation harness must accept either a training checkpoint (``best.pt``)
or an exported ONNX model and run them through the same code path. To that end
this module exposes:

* :func:`load_model_from_checkpoint` -- rebuild the ``segmentation_models_pytorch``
  model from the config stored in the checkpoint and load the trained weights;
* :func:`load_onnx_session` -- open an :class:`onnxruntime.InferenceSession`;
* :func:`predict_logits` -- a single inference entry point that accepts either a
  :class:`torch.nn.Module` or an ONNX session and returns ``(N, C, H, W)`` logits.

Model construction mirrors :func:`oilspill.training.trainer.build_model` so the
architecture is reconstructed identically (``encoder_weights=None`` here because
the trained weights are loaded from the checkpoint, not from a pretrained stem).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from oilspill.models import build_model as registry_build_model
from oilspill.training.config import ModelConfig

# A model is either a torch module or an ONNX inference session.
ModelOrSession = nn.Module | ort.InferenceSession


def build_model_from_config(config: dict[str, Any]) -> nn.Module:
    """Instantiate the model from a checkpoint's ``config["model"]`` block.

    Routes through :func:`oilspill.models.build_model` with ``pretrained=False``
    so any registered or smp architecture is reconstructed identically to
    training; the trained checkpoint weights are loaded afterwards, so base/
    ImageNet weights are not downloaded here.
    """
    model_cfg = ModelConfig.model_validate(config["model"])
    return registry_build_model(model_cfg, pretrained=False)


def load_model_from_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a trained model and its config from a training checkpoint.

    Parameters
    ----------
    path:
        Path to a checkpoint written by the trainer (a dict with
        ``model_state_dict`` and ``config``).
    device:
        Device to move the model to. The model is returned in ``eval`` mode.

    Returns
    -------
    tuple
        ``(model, config)`` -- the loaded module (eval mode, on ``device``) and
        the nested config dict from the checkpoint.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"checkpoint {path} is missing 'config'/'model_state_dict'; "
            "is this a trainer checkpoint?"
        )
    config: dict[str, Any] = checkpoint["config"]
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def load_onnx_session(path: Path | str) -> ort.InferenceSession:
    """Open an :class:`onnxruntime.InferenceSession` (CPU provider) for ``path``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def predict_logits(model: ModelOrSession, batch: torch.Tensor) -> torch.Tensor:
    """Run inference on a ``(N, C, H, W)`` batch, returning ``(N, C, H, W)`` logits.

    Accepts either a torch module or an ONNX session so the harness can treat both
    uniformly. The returned tensor lives on the CPU.
    """
    if isinstance(model, ort.InferenceSession):
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: batch.detach().cpu().numpy().astype(np.float32)})
        return torch.from_numpy(np.asarray(outputs[0]))
    with torch.no_grad():
        device = next(model.parameters()).device
        logits = model(batch.to(device))
    return logits.detach().cpu()
