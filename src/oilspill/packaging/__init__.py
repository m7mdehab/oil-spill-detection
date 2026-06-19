"""Model packaging: ONNX export and Hugging Face publishing.

This subpackage turns a trained checkpoint into shippable artifacts:

* :mod:`oilspill.packaging.onnx_export` -- export a checkpoint to ONNX (legacy
  ``torch.onnx.export`` path with a dynamic batch/spatial axis) and verify that
  the ONNX graph reproduces the torch model's logits to within a tolerance;
* :mod:`oilspill.packaging.model_card` -- render an honest Hugging Face model
  card (markdown) from a committed metrics JSON, with no invented numbers.
"""

from __future__ import annotations

from oilspill.packaging.model_card import build_model_card
from oilspill.packaging.onnx_export import export_to_onnx, verify_parity

__all__ = ["build_model_card", "export_to_onnx", "verify_parity"]
