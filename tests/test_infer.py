"""Tests for tiled ONNX inference and stitching (:mod:`oilspill.pipeline.infer`).

These build a tiny real ONNX model in-process (a single 1x1 conv mapping
``(N, 3, H, W) -> (N, 5, H, W)``) and drive :func:`tiled_predict` through
:mod:`onnxruntime`, so they need no network and no large artifacts. A 1x1 conv has
no receptive-field edge effect, which lets us assert that tiled+stitched logits
equal a single full-image forward exactly on the (non-reflect-padded) interior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from oilspill.metrics import NUM_CLASSES, OIL_CLASS_INDEX
from oilspill.pipeline.infer import (
    INPUT_NAME,
    OUTPUT_NAME,
    load_session,
    tiled_predict,
)


class _PointwiseSeg(torch.nn.Module):
    """Trivial fully-pointwise segmentation head: a single 1x1 conv to 5 classes."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, NUM_CLASSES, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@pytest.fixture(scope="module")
def onnx_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Export a tiny deterministic 1x1-conv ONNX model with dynamic axes."""
    torch.manual_seed(0)
    model = _PointwiseSeg().eval()
    out_path = tmp_path_factory.mktemp("onnx") / "tiny.onnx"
    dummy = torch.randn(1, 3, 16, 16, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(out_path),
            dynamo=False,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                INPUT_NAME: {0: "batch", 2: "height", 3: "width"},
                OUTPUT_NAME: {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=17,
        )
    return out_path


@pytest.fixture(scope="module")
def session(onnx_model_path: Path) -> ort.InferenceSession:
    return load_session(onnx_model_path)


def _make_scene(height: int, width: int, *, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((3, height, width)).astype(np.float32)


def test_load_session_missing_path() -> None:
    with pytest.raises(FileNotFoundError):
        load_session("does/not/exist.onnx")


def test_load_session_returns_session(session: ort.InferenceSession) -> None:
    assert isinstance(session, ort.InferenceSession)


def test_output_shapes_and_ranges(session: ort.InferenceSession) -> None:
    height, width = 200, 160
    scene = _make_scene(height, width)
    class_mask, oil_prob = tiled_predict(scene, session, tile_size=64, overlap=16, batch_size=4)

    assert class_mask.shape == (height, width)
    assert oil_prob.shape == (height, width)
    assert class_mask.dtype == np.uint8
    assert oil_prob.dtype == np.float32
    assert class_mask.min() >= 0
    assert class_mask.max() < NUM_CLASSES
    assert float(oil_prob.min()) >= 0.0
    assert float(oil_prob.max()) <= 1.0


def test_non_divisible_scene_size(session: ort.InferenceSession) -> None:
    """A scene whose size is not a multiple of the stride exercises pad + crop."""
    height, width = 300, 250
    scene = _make_scene(height, width, seed=7)
    class_mask, oil_prob = tiled_predict(scene, session, tile_size=128, overlap=32, batch_size=3)
    assert class_mask.shape == (height, width)
    assert oil_prob.shape == (height, width)
    assert np.isfinite(oil_prob).all()


def test_scene_smaller_than_tile(session: ort.InferenceSession) -> None:
    """A scene smaller than one tile is reflect/edge-padded up to a single tile."""
    scene = _make_scene(40, 50, seed=3)
    class_mask, oil_prob = tiled_predict(scene, session, tile_size=128, overlap=32)
    assert class_mask.shape == (40, 50)
    assert oil_prob.shape == (40, 50)


def test_determinism(session: ort.InferenceSession) -> None:
    scene = _make_scene(180, 220, seed=5)
    mask1, prob1 = tiled_predict(scene, session, tile_size=96, overlap=24)
    mask2, prob2 = tiled_predict(scene, session, tile_size=96, overlap=24)
    np.testing.assert_array_equal(mask1, mask2)
    np.testing.assert_array_equal(prob1, prob2)


def test_default_tile_size_runs(session: ort.InferenceSession) -> None:
    """The 512/64 defaults must work on a scene smaller than one default tile."""
    scene = _make_scene(300, 300, seed=11)
    class_mask, oil_prob = tiled_predict(scene, session)
    assert class_mask.shape == (300, 300)
    assert oil_prob.shape == (300, 300)


def _full_forward(session: ort.InferenceSession, scene: np.ndarray) -> np.ndarray:
    """Single full-image ONNX forward, returning logits ``(C, H, W)`` float64."""
    out = session.run([OUTPUT_NAME], {INPUT_NAME: scene[np.newaxis].astype(np.float32)})[0]
    return np.asarray(out, dtype=np.float64)[0]


def test_interior_tiled_equals_full(session: ort.InferenceSession) -> None:
    """Tiled+stitched result equals a single full-image forward on the interior.

    With a 1x1 conv there is no receptive-field edge effect, so the only place
    tiling can differ from a full forward is where reflect padding leaks into
    border tiles. The interior (away from the scene border by the overlap width)
    must match the full-image softmax/argmax exactly within float tolerance.
    """
    height, width = 256, 224
    tile_size, overlap = 96, 24
    scene = _make_scene(height, width, seed=42)

    class_mask, oil_prob = tiled_predict(
        scene, session, tile_size=tile_size, overlap=overlap, batch_size=4
    )

    full_logits = _full_forward(session, scene)
    full_mask = np.argmax(full_logits, axis=0).astype(np.uint8)
    shifted = full_logits - full_logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    full_probs = exp / exp.sum(axis=0, keepdims=True)
    full_oil = full_probs[OIL_CLASS_INDEX].astype(np.float32)

    # Interior excludes a border band where reflect padding influences edge tiles.
    b = overlap
    sl = (slice(b, height - b), slice(b, width - b))
    np.testing.assert_allclose(oil_prob[sl], full_oil[sl], atol=1e-5)
    np.testing.assert_array_equal(class_mask[sl], full_mask[sl])


def test_invalid_arguments(session: ort.InferenceSession) -> None:
    scene = _make_scene(64, 64)
    with pytest.raises(ValueError):
        tiled_predict(scene[0], session)  # not (3, H, W)
    with pytest.raises(ValueError):
        tiled_predict(scene, session, tile_size=64, overlap=64)  # overlap >= tile
    with pytest.raises(ValueError):
        tiled_predict(scene, session, batch_size=0)
