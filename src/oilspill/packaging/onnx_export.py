"""Export a trained checkpoint to ONNX and verify numerical parity.

The exporter uses the *legacy* tracing path (``torch.onnx.export(..., dynamo=
False)``) deliberately: the new dynamo exporter requires ``onnxscript``, which is
not a project dependency. The graph is exported with dynamic batch *and* spatial
axes so the served model accepts variable-size inputs (the segmentation head is
fully convolutional, so this is sound).

Parity is checked by running the torch model and an :class:`onnxruntime.
InferenceSession` on the *same* fixed random inputs and comparing logits; the max
absolute difference must fall under a small tolerance, otherwise the export is
rejected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from oilspill.evaluation.model_loading import load_model_from_checkpoint

# Names baked into the exported graph; kept in one place so the serving code and
# the parity check agree on them.
INPUT_NAME = "input"
OUTPUT_NAME = "logits"

# Batch (0) and spatial (2, 3) axes are dynamic; the channel axis (1) is fixed at
# 3 (SAR is replicated to 3 channels upstream to match the pretrained stem).
_DYNAMIC_AXES: dict[str, dict[int, str]] = {
    INPUT_NAME: {0: "batch", 2: "height", 3: "width"},
    OUTPUT_NAME: {0: "batch", 2: "height", 3: "width"},
}


def _as_hw(image_size: int | tuple[int, int]) -> tuple[int, int]:
    """Normalise an ``image_size`` argument to ``(height, width)``."""
    if isinstance(image_size, int):
        return (image_size, image_size)
    height, width = image_size
    return (int(height), int(width))


def export_to_onnx(
    checkpoint_path: Path | str,
    out_path: Path | str,
    *,
    image_size: int | tuple[int, int],
    opset: int = 17,
) -> Path:
    """Export a trainer checkpoint to an ONNX file.

    Parameters
    ----------
    checkpoint_path:
        Path to a trainer checkpoint (``best.pt``).
    out_path:
        Destination ``.onnx`` path. Parent directories are created.
    image_size:
        Spatial size of the dummy input used for tracing. A square ``int`` or an
        explicit ``(height, width)``. The exported graph still accepts other
        sizes thanks to the dynamic spatial axes; this only fixes the trace.
    opset:
        ONNX opset version (default 17).

    Returns
    -------
    Path
        The path the model was written to.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, _config = load_model_from_checkpoint(checkpoint_path, device="cpu")
    height, width = _as_hw(image_size)
    dummy = torch.randn(1, 3, height, width, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(out_path),
            dynamo=False,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes=_DYNAMIC_AXES,
            opset_version=opset,
        )
    return out_path


def verify_parity(
    checkpoint_path: Path | str,
    onnx_path: Path | str,
    *,
    image_size: int | tuple[int, int],
    n: int = 8,
    atol: float = 1e-4,
) -> float:
    """Check that the ONNX graph reproduces the torch model's logits.

    Runs both the torch model and an ONNX Runtime session on the *same* ``n``
    fixed random inputs (deterministic seed) and returns the maximum absolute
    difference over all elements. Asserts the difference is below ``atol``.

    Parameters
    ----------
    checkpoint_path:
        The checkpoint the ONNX file was exported from.
    onnx_path:
        The exported ONNX file.
    image_size:
        Spatial size of the parity inputs (square ``int`` or ``(height, width)``).
    n:
        Number of random inputs (a single batch of ``n``).
    atol:
        Maximum tolerated absolute difference.

    Returns
    -------
    float
        The maximum absolute difference between torch and ONNX logits.
    """
    height, width = _as_hw(image_size)
    model, _config = load_model_from_checkpoint(checkpoint_path, device="cpu")

    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(n, 3, height, width, generator=generator, dtype=torch.float32)

    with torch.no_grad():
        torch_logits = model(inputs).detach().cpu().numpy().astype(np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    (onnx_logits,) = session.run(None, {input_name: inputs.numpy().astype(np.float32)})
    onnx_logits = np.asarray(onnx_logits, dtype=np.float32)

    max_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
    assert max_diff < atol, (
        f"ONNX parity check failed: max abs diff {max_diff:.3e} >= atol {atol:.3e}"
    )
    return max_diff
