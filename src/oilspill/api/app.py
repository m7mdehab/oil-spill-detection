"""FastAPI application factory for the oil-spill inference service.

Exposes a small JSON API plus static hosting for the built frontend:

* ``GET  /healthz``            -- liveness probe.
* ``GET  /models``            -- model registry with real evaluation metrics.
* ``GET  /samples``           -- preloaded sample images (list + bytes).
* ``GET  /samples/{name}``    -- raw sample image bytes.
* ``POST /predict``           -- segment one uploaded image.
* ``POST /jobs/scene``        -- queue a full-scene AOI detection job.
* ``GET  /jobs/{job_id}``     -- poll a scene job.

The app degrades gracefully: a missing ONNX export yields a clear 503 instead of
a crash, and the static frontend is only mounted when ``web/dist`` exists.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from oilspill.api.models import (
    HealthResponse,
    JobStatusResponse,
    ModelsResponse,
    PredictResponse,
    SampleInfo,
    SamplesResponse,
    SceneJobRequest,
    SceneJobResponse,
)
from oilspill.api.service import JobStore, ModelRegistry, predict_image
from oilspill.api.settings import Settings, get_settings

if TYPE_CHECKING:
    from oilspill.api.service import SceneRunner

# Image extensions exposed by the /samples endpoint.
_SAMPLE_EXTS = (".jpg", ".jpeg", ".png")


def _get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry  # type: ignore[no-any-return]


def _get_jobs(request: Request) -> JobStore:
    return request.app.state.jobs  # type: ignore[no-any-return]


def _get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


# Annotated dependency aliases (avoids `Depends()` in plain argument defaults).
RegistryDep = Annotated[ModelRegistry, Depends(_get_registry)]
JobsDep = Annotated[JobStore, Depends(_get_jobs)]
SettingsDep = Annotated[Settings, Depends(_get_settings_dep)]
UploadDep = Annotated[UploadFile, File(...)]
ModelFormDep = Annotated[str | None, Form()]


def create_app(
    settings: Settings | None = None,
    *,
    scene_runner: SceneRunner | None = None,
) -> FastAPI:
    """Build and return the configured FastAPI application.

    Parameters
    ----------
    settings:
        Optional pre-built settings; defaults to environment-resolved settings.
    scene_runner:
        Optional override for the scene-job runner (tests inject a fast mock to
        avoid network access and a real model).
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="Oil Spill Detection API",
        version="0.1.0",
        description="SAR oil-spill segmentation and full-scene detection.",
    )
    app.state.settings = settings
    app.state.registry = ModelRegistry(settings)
    app.state.jobs = JobStore(runner=scene_runner)

    @app.get("/healthz", response_model=HealthResponse, tags=["system"])
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/models", response_model=ModelsResponse, tags=["models"])
    def list_models(registry: RegistryDep) -> ModelsResponse:
        return ModelsResponse(models=registry.list_models())

    @app.get("/samples", response_model=SamplesResponse, tags=["samples"])
    def list_samples(cfg: SettingsDep) -> SamplesResponse:
        samples: list[SampleInfo] = []
        if cfg.samples_dir.exists():
            for path in sorted(cfg.samples_dir.iterdir()):
                if path.suffix.lower() in _SAMPLE_EXTS:
                    samples.append(SampleInfo(id=path.stem, url=f"/samples/{path.name}"))
        return SamplesResponse(samples=samples)

    @app.get("/samples/{name}", tags=["samples"])
    def get_sample(name: str, cfg: SettingsDep) -> FileResponse:
        # Guard against path traversal: only serve plain filenames from the dir.
        if "/" in name or "\\" in name or name in {"", ".", ".."}:
            raise HTTPException(status_code=400, detail="Invalid sample name.")
        path = cfg.samples_dir / name
        if not path.is_file() or path.suffix.lower() not in _SAMPLE_EXTS:
            raise HTTPException(status_code=404, detail=f"Sample not found: {name}")
        return FileResponse(path)

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    async def predict(
        file: UploadDep,
        registry: RegistryDep,
        cfg: SettingsDep,
        model: ModelFormDep = None,
    ) -> PredictResponse:
        raw = await file.read()
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail="Could not read image.") from exc

        try:
            payload = predict_image(registry, image, model, cfg)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return PredictResponse(**payload)

    @app.post("/jobs/scene", response_model=SceneJobResponse, tags=["jobs"])
    def create_scene_job(
        body: SceneJobRequest,
        background_tasks: BackgroundTasks,
        registry: RegistryDep,
        jobs: JobsDep,
    ) -> SceneJobResponse:
        job_id = jobs.create()
        background_tasks.add_task(
            jobs.run,
            job_id,
            body.aoi,
            body.start,
            body.end,
            body.model,
            registry,
        )
        return SceneJobResponse(job_id=job_id, status="queued")

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
    def get_job(job_id: str, jobs: JobsDep) -> JobStatusResponse:
        status = jobs.status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return status

    # Mount the built frontend last so the API routes above take precedence; only
    # if it exists, so the API still serves standalone.
    if settings.web_dist.exists():
        app.mount("/", StaticFiles(directory=settings.web_dist, html=True), name="web")

    return app


__all__ = ["create_app"]
