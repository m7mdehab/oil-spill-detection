"""Tiled ONNX inference and logit-averaged stitching for full SAR scenes.

A full Sentinel-1 scene is far larger than the model's training resolution, so it
is processed in overlapping tiles and the per-tile predictions are stitched back
into a single full-scene result. The model operates on *logits*, so stitching is
done in logit space (the overlapping tile outputs are averaged before the final
softmax/argmax) rather than on hard class labels -- averaging soft evidence avoids
the blocky seams a label-vote stitcher produces and is the standard approach for
sliding-window semantic segmentation.

Pipeline
--------
``scene (3, H, W)``  -- already-normalised, ImageNet-space CHW from
:mod:`oilspill.pipeline.preprocess`; this module does **not** re-normalise --

    -> reflect-pad to a whole number of tiles
    -> extract overlapping ``tile_size`` tiles (stride ``tile_size - overlap``)
    -> batched ONNX forward -> per-tile logits ``(B, C, t, t)``
    -> accumulate logits into a full-scene buffer with a matching weight buffer
       (a cosine taper, so tile interiors dominate over their feathered edges)
    -> divide to get the blended mean logits, crop the padding
    -> softmax over classes -> ``oil_prob``; argmax -> ``class_mask``

Georeferencing
--------------
The returned ``class_mask`` / ``oil_prob`` are aligned 1:1 with the input scene
grid: every output pixel corresponds to the same input pixel (the reflect padding
is cropped away and there is **no** resampling). The caller's affine transform and
CRS for the input scene therefore apply to the outputs unchanged.

Tile size
---------
``tile_size`` defaults to 512 to match the resolution the model was trained at,
which gives the closest match to the training distribution and the best accuracy.
The model is fully resolution-flexible (its ONNX graph has dynamic spatial axes),
so the plan's smaller 256/32 setting is also valid -- pass ``tile_size=256,
overlap=32`` for lower peak memory at some accuracy cost. ``tile_size`` and
``overlap`` are both configurable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

from oilspill.metrics import NUM_CLASSES, OIL_CLASS_INDEX

if TYPE_CHECKING:
    from collections.abc import Iterator

# Names baked into the exported ONNX graph (see oilspill.packaging.onnx_export).
INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def load_session(onnx_path: str | Path) -> ort.InferenceSession:
    """Open an ONNX model as a CPU :class:`onnxruntime.InferenceSession`.

    Parameters
    ----------
    onnx_path:
        Path to the exported ``.onnx`` model (e.g. ``artifacts/exports/model.onnx``).

    Returns
    -------
    onnxruntime.InferenceSession
        Session pinned to the CPU execution provider.
    """
    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _cosine_window(tile_size: int) -> np.ndarray:
    """Return a 2-D cosine (Hann) taper ``(tile_size, tile_size)`` in ``(0, 1]``.

    The taper is strictly positive (a small floor keeps every pixel's weight above
    zero, so a pixel covered by a single tile is never divided by zero) and peaks
    at the tile centre, so where tiles overlap the better-supported tile interiors
    are weighted above their feathered edges.
    """
    if tile_size == 1:
        return np.ones((1, 1), dtype=np.float64)
    # 1-D Hann window scaled into (eps, 1], then outer-product to 2-D.
    n = np.arange(tile_size, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (tile_size - 1))
    hann = hann * (1.0 - 1e-3) + 1e-3
    return np.outer(hann, hann)


def _tile_origins(extent: int, tile_size: int, stride: int) -> list[int]:
    """Top-left tile origins along one axis covering ``[0, extent)``.

    ``extent`` is assumed already padded to a whole number of strides, so the
    origins tile it exactly with no tile running past the edge.
    """
    if extent <= tile_size:
        return [0]
    origins = list(range(0, extent - tile_size + 1, stride))
    if origins[-1] != extent - tile_size:
        origins.append(extent - tile_size)
    return origins


def _iter_batches(items: list[tuple[int, int]], batch_size: int) -> Iterator[list[tuple[int, int]]]:
    """Yield ``items`` (tile origin pairs) in chunks of at most ``batch_size``."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def tiled_predict(
    scene_chw: np.ndarray,
    session: ort.InferenceSession,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
    num_classes: int = NUM_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """Run tiled ONNX inference over a full scene and stitch the logits.

    The scene is reflect-padded to a whole number of tiles, cut into overlapping
    ``tile_size`` x ``tile_size`` tiles (stride ``tile_size - overlap``), run
    through ``session`` in batches, and the per-tile logits are accumulated into a
    full-scene buffer weighted by a cosine taper. Overlapping regions are *blended*
    (weighted-averaged in logit space), never overwritten. The padding is then
    cropped so the result lines up 1:1 with the input grid (no resampling), and a
    softmax/argmax produces the outputs.

    Parameters
    ----------
    scene_chw:
        Model-ready, already-normalised scene of shape ``(3, H, W)`` float32 (the
        ImageNet-normalised CHW tensor produced upstream). It is **not**
        re-normalised here.
    session:
        An :class:`onnxruntime.InferenceSession` for the segmentation model, whose
        graph maps input ``(N, 3, h, w)`` to logits ``(N, num_classes, h, w)`` with
        dynamic batch and spatial axes.
    tile_size:
        Side length of the square inference tiles. Defaults to 512 to match the
        model's training resolution; the model is resolution-flexible, so smaller
        settings (e.g. 256 with ``overlap=32``) are also valid.
    overlap:
        Pixels of overlap between adjacent tiles. The stride is ``tile_size -
        overlap``. Must satisfy ``0 <= overlap < tile_size``.
    batch_size:
        Number of tiles per ONNX forward pass.
    num_classes:
        Number of output classes (the model's logit channel count).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(class_mask, oil_prob)`` where ``class_mask`` is ``(H, W)`` uint8 holding
        the per-pixel argmax class, and ``oil_prob`` is ``(H, W)`` float32 holding
        the softmax probability of the oil-spill class. Both share the input ``H, W``
        and grid; the caller's geotransform/CRS apply unchanged.
    """
    if scene_chw.ndim != 3 or scene_chw.shape[0] != 3:
        raise ValueError(f"scene_chw must have shape (3, H, W), got {scene_chw.shape}")
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")
    if not 0 <= overlap < tile_size:
        raise ValueError(f"require 0 <= overlap < tile_size, got overlap={overlap}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    scene = np.ascontiguousarray(scene_chw, dtype=np.float32)
    _, height, width = scene.shape
    stride = tile_size - overlap

    # Reflect-pad so the (possibly tile-smaller) scene tiles into whole strides.
    # Padded extent is the smallest multiple of `stride` that is >= max(extent,
    # tile_size) and at least one full tile.
    def _padded_extent(extent: int) -> int:
        target = max(extent, tile_size)
        n_strides = (target - tile_size + stride - 1) // stride
        return tile_size + n_strides * stride

    pad_h = _padded_extent(height)
    pad_w = _padded_extent(width)
    # np.pad reflect mode requires pad < dim; fall back to edge padding when the
    # scene is too small for a reflection of the needed width.
    pad_mode_h = "reflect" if (pad_h - height) < height else "edge"
    pad_mode_w = "reflect" if (pad_w - width) < width else "edge"
    if pad_mode_h != pad_mode_w:
        mode = "edge"  # mixed: use edge for both to keep np.pad simple and safe
        padded = np.pad(scene, ((0, 0), (0, pad_h - height), (0, pad_w - width)), mode=mode)
    else:
        padded = np.pad(
            scene,
            ((0, 0), (0, pad_h - height), (0, pad_w - width)),
            mode=pad_mode_h,  # type: ignore[arg-type]
        )

    origins_y = _tile_origins(pad_h, tile_size, stride)
    origins_x = _tile_origins(pad_w, tile_size, stride)
    tile_origins = [(y, x) for y in origins_y for x in origins_x]

    window = _cosine_window(tile_size)  # (tile_size, tile_size)
    logit_acc = np.zeros((num_classes, pad_h, pad_w), dtype=np.float64)
    weight_acc = np.zeros((pad_h, pad_w), dtype=np.float64)

    for batch in _iter_batches(tile_origins, batch_size):
        tiles = np.stack(
            [padded[:, y : y + tile_size, x : x + tile_size] for (y, x) in batch],
            axis=0,
        ).astype(np.float32)
        outputs = session.run([OUTPUT_NAME], {INPUT_NAME: tiles})[0]
        logits = np.asarray(outputs, dtype=np.float64)  # (B, C, t, t)
        if logits.shape[1] != num_classes:
            raise ValueError(
                f"model produced {logits.shape[1]} classes, expected num_classes={num_classes}"
            )
        for tile_logits, (y, x) in zip(logits, batch, strict=True):
            logit_acc[:, y : y + tile_size, x : x + tile_size] += tile_logits * window
            weight_acc[y : y + tile_size, x : x + tile_size] += window

    # Weighted-average the overlapping logits, then crop the padding.
    mean_logits = logit_acc / weight_acc[np.newaxis, :, :]
    mean_logits = mean_logits[:, :height, :width]

    # Numerically stable softmax over the class axis.
    shifted = mean_logits - mean_logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=0, keepdims=True)

    class_mask = np.argmax(mean_logits, axis=0).astype(np.uint8)
    oil_prob = probs[OIL_CLASS_INDEX].astype(np.float32)
    return class_mask, oil_prob
