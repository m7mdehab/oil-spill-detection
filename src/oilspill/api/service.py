"""Service layer: model registry, ONNX caching, prediction and scene jobs.

This module holds all the non-HTTP logic the route handlers delegate to, so the
FastAPI layer (``app.py``) stays a thin adapter:

* :class:`ModelRegistry` reads the committed evaluation JSONs in ``results_dir``
  to expose each model's real metrics, lazily loads each model's ONNX session on
  first use and caches it (load once, reuse), and reports availability by probing
  for a loadable export.
* :func:`predict_image` runs the upload-image path: PIL -> ImageNet-normalised
  CHW -> :func:`tiled_predict` -> colourised mask + blended overlay (base64 PNGs)
  + per-class pixel percentages.
* :class:`JobStore` is an in-process registry of background scene jobs (a plain
  dict guarded by a lock); jobs run on FastAPI ``BackgroundTasks`` threads. No
  external broker is involved.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from oilspill.api.models import (
    JobResult,
    JobStatusResponse,
    ModelInfo,
    PerClassMetrics,
)
from oilspill.data.colors import CLASS_COLORS, colorize_mask
from oilspill.metrics import CLASS_NAMES, NUM_CLASSES
from oilspill.pipeline.infer import load_session, tiled_predict

if TYPE_CHECKING:
    import onnxruntime as ort

    from oilspill.api.settings import Settings

# ImageNet normalisation constants. Re-declared here (rather than importing the
# albumentations-based transforms module) to keep the API import-light; these are
# the exact values used in training -- see oilspill.data.transforms.IMAGENET_MEAN.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default model size for plain uploaded images (matches the training chip size).
_PREDICT_SIZE = 512

# Smoke/debug runs are excluded from the public model list by tag.
_SKIP_TAGS = frozenset({"smoke"})


# --- model registry ----------------------------------------------------------


@dataclass
class _ModelRecord:
    """Internal per-model state derived from one evaluation JSON."""

    model_id: str
    name: str
    info: ModelInfo
    onnx_path: Path | None


class ModelRegistry:
    """Reads evaluation JSONs and lazily loads + caches ONNX sessions.

    Parameters
    ----------
    settings:
        Resolved API settings (provides ``results_dir``, ``onnx_dir`` and
        ``default_onnx``).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._lock = threading.Lock()
        self._records: dict[str, _ModelRecord] = self._load_records()

    def _candidate_onnx(self, model_id: str) -> Path | None:
        """Return a loadable ONNX path for ``model_id`` or ``None``.

        Prefers a per-model export (``<onnx_dir>/<model_id>.onnx``) and falls
        back to the configured default model. Returns ``None`` when neither
        exists on disk.
        """
        per_model = self._settings.onnx_dir / f"{model_id}.onnx"
        if per_model.exists():
            return per_model
        if self._settings.default_onnx.exists():
            return self._settings.default_onnx
        return None

    def _load_records(self) -> dict[str, _ModelRecord]:
        """Build the model records from every non-smoke JSON in ``results_dir``."""
        records: dict[str, _ModelRecord] = {}
        results_dir = self._settings.results_dir
        if not results_dir.exists():
            return records
        for json_path in sorted(results_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            meta = data.get("meta", {})
            if meta.get("tag") in _SKIP_TAGS:
                continue
            model_id = meta.get("run_name") or json_path.stem
            onnx_path = self._candidate_onnx(model_id)
            info = self._build_info(model_id, data, available=onnx_path is not None)
            records[model_id] = _ModelRecord(
                model_id=model_id,
                name=info.name,
                info=info,
                onnx_path=onnx_path,
            )
        return records

    @staticmethod
    def _build_info(model_id: str, data: dict[str, Any], *, available: bool) -> ModelInfo:
        """Map one evaluation JSON to a :class:`ModelInfo`."""
        metrics = data.get("metrics", {})
        aggregate = metrics.get("aggregate", {})
        per_class_raw = metrics.get("per_class", {})
        class_names = metrics.get("class_names", list(CLASS_NAMES))

        iou = per_class_raw.get("iou", {})
        precision = per_class_raw.get("precision", {})
        recall = per_class_raw.get("recall", {})
        f1 = per_class_raw.get("f1", {})
        per_class = {
            name: PerClassMetrics(
                iou=iou.get(name),
                precision=precision.get(name),
                recall=recall.get(name),
                f1=f1.get(name),
            )
            for name in class_names
        }

        return ModelInfo(
            id=model_id,
            name=model_id,
            oil_iou=aggregate.get("oil_iou"),
            oil_recall=aggregate.get("oil_recall"),
            mean_iou=aggregate.get("mean_iou"),
            macro_f1=aggregate.get("macro_f1"),
            pixel_accuracy=aggregate.get("pixel_accuracy"),
            per_class=per_class,
            available=available,
        )

    def list_models(self) -> list[ModelInfo]:
        """Return the public model list (real metrics, availability flags)."""
        return [record.info for record in self._records.values()]

    def resolve_id(self, model_id: str | None) -> str:
        """Resolve a requested model id to a known one.

        Falls back to the first available model, then the first known model, so a
        caller that omits ``model`` (or names an unknown one) still gets a usable
        result rather than an error.
        """
        if model_id and model_id in self._records:
            return model_id
        for record in self._records.values():
            if record.info.available:
                return record.model_id
        if self._records:
            return next(iter(self._records))
        return model_id or "model"

    def get_session(self, model_id: str | None) -> tuple[str, ort.InferenceSession]:
        """Return ``(resolved_id, session)``, loading + caching on first use.

        Raises
        ------
        FileNotFoundError
            If no ONNX export is available for the resolved model.
        """
        resolved = self.resolve_id(model_id)
        with self._lock:
            cached = self._sessions.get(resolved)
            if cached is not None:
                return resolved, cached
            record = self._records.get(resolved)
            onnx_path = record.onnx_path if record else self._candidate_onnx(resolved)
            if onnx_path is None:
                raise FileNotFoundError(
                    f"No ONNX export available for model '{resolved}'. "
                    "Export one to the configured onnx_dir or set OILSPILL_API_DEFAULT_ONNX."
                )
            session = load_session(onnx_path)
            self._sessions[resolved] = session
            return resolved, session


# --- prediction --------------------------------------------------------------


def _image_to_chw(image: Image.Image, size: int = _PREDICT_SIZE) -> np.ndarray:
    """Load a PIL image as a model-ready ImageNet-normalised CHW float32 tensor.

    Uploaded demo images are JPG/PNG chips like the training data, so this matches
    the training preprocessing: RGB -> resize to ``size`` -> scale to [0, 1] ->
    ImageNet-normalise -> CHW.
    """
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)


def _png_data_uri(rgb: np.ndarray) -> str:
    """Encode an ``HxWx3`` uint8 RGB array as a base64 PNG ``data:`` URI."""
    buf = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def predict_image(
    registry: ModelRegistry,
    image: Image.Image,
    model_id: str | None,
    settings: Settings,
) -> dict[str, Any]:
    """Run segmentation on one uploaded image and build the response payload.

    Returns a plain dict matching :class:`oilspill.api.models.PredictResponse`.
    """
    resolved, session = registry.get_session(model_id)

    orig_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = orig_rgb.shape[:2]

    scene_chw = _image_to_chw(image, size=_PREDICT_SIZE)
    class_mask, _oil_prob = tiled_predict(
        scene_chw,
        session,
        tile_size=settings.tile_size,
        overlap=settings.overlap,
        batch_size=settings.batch_size,
    )

    # Colourise at model resolution, then resize back to the original image size
    # (nearest-neighbour to preserve discrete class colours) for the overlay.
    mask_rgb_small = colorize_mask(class_mask)
    mask_img = Image.fromarray(mask_rgb_small, mode="RGB").resize(
        (width, height), Image.Resampling.NEAREST
    )
    mask_rgb = np.asarray(mask_img, dtype=np.uint8)

    overlay = (0.5 * orig_rgb.astype(np.float32) + 0.5 * mask_rgb.astype(np.float32)).astype(
        np.uint8
    )

    # Per-class pixel percentages computed at model resolution.
    counts = np.bincount(class_mask.ravel(), minlength=NUM_CLASSES).astype(np.float64)
    total = float(counts.sum()) or 1.0
    class_percentages = {
        CLASS_NAMES[i]: round(100.0 * counts[i] / total, 4) for i in range(NUM_CLASSES)
    }
    legend = {CLASS_NAMES[i]: list(CLASS_COLORS[i]) for i in range(NUM_CLASSES)}

    return {
        "model": resolved,
        "width": width,
        "height": height,
        "class_percentages": class_percentages,
        "legend": legend,
        "mask_png": _png_data_uri(mask_rgb),
        "overlay_png": _png_data_uri(overlay),
    }


# --- scene jobs --------------------------------------------------------------


@dataclass
class _Job:
    """Internal mutable state of one scene-detection job."""

    job_id: str
    status: str = "queued"
    detail: str | None = None
    result: JobResult | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


# Type of the callable that performs the actual detection. Injected so tests can
# substitute a fast mock for the network-bound real pipeline.
SceneRunner = Any


class JobStore:
    """In-process registry of background scene jobs (dict + lock, no broker).

    Jobs are created in the ``queued`` state, moved to ``running`` when the
    background task starts, and end in ``done`` or ``error``. State lives only in
    this process for the life of the server, which is what the single-container
    Spaces deployment needs.
    """

    def __init__(self, runner: SceneRunner | None = None) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self._runner: SceneRunner = runner if runner is not None else _run_scene_detection

    def create(self) -> str:
        """Register a new queued job and return its id."""
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = _Job(job_id=job_id)
        return job_id

    def _get(self, job_id: str) -> _Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def status(self, job_id: str) -> JobStatusResponse | None:
        """Return the current status of ``job_id`` (or ``None`` if unknown)."""
        job = self._get(job_id)
        if job is None:
            return None
        with job._lock:
            return JobStatusResponse(
                job_id=job.job_id,
                status=job.status,  # type: ignore[arg-type]
                detail=job.detail,
                result=job.result,
            )

    def run(
        self,
        job_id: str,
        aoi: dict[str, Any],
        start: str,
        end: str,
        model_id: str | None,
        registry: ModelRegistry,
    ) -> None:
        """Execute one job to completion (intended to run on a background thread).

        Any failure is captured into the job's ``error`` state with a message,
        rather than propagated, so a failing job never crashes the server.
        """
        job = self._get(job_id)
        if job is None:
            return
        with job._lock:
            job.status = "running"
        try:
            result = self._runner(aoi, start, end, model_id, registry)
            with job._lock:
                job.status = "done"
                job.result = result
        except Exception as exc:  # surface any failure as job error, never crash
            with job._lock:
                job.status = "error"
                job.detail = str(exc)


def _run_scene_detection(
    aoi: dict[str, Any],
    start: str,
    end: str,
    model_id: str | None,
    registry: ModelRegistry,
) -> JobResult:
    """Default scene runner: search/download/detect over an AOI via the pipeline.

    Writes products to a temporary directory and returns the summary + inline
    GeoJSON. This is the only path that touches the network; tests inject a mock
    runner instead.
    """
    import tempfile

    from oilspill.pipeline.detect import detect_from_aoi

    resolved = registry.resolve_id(model_id)
    _record_id, _session = registry.get_session(resolved)
    onnx_path = registry._candidate_onnx(resolved)
    if onnx_path is None:
        raise FileNotFoundError(f"No ONNX export available for model '{resolved}'.")

    with tempfile.TemporaryDirectory(prefix="oilspill-scene-") as tmp:
        out_dir = Path(tmp)
        aoi_path = out_dir / "aoi.geojson"
        aoi_path.write_text(json.dumps(aoi), encoding="utf-8")
        result = detect_from_aoi(aoi_path, start, end, onnx_path, out_dir)
        geojson = json.loads(Path(result.geojson_path).read_text(encoding="utf-8"))
        return JobResult(
            num_oil_polygons=result.num_oil_polygons,
            total_oil_area_km2=result.total_oil_area_km2,
            geojson=geojson,
        )


__all__ = [
    "JobStore",
    "ModelRegistry",
    "SceneRunner",
    "predict_image",
]
