"""Tests for the FastAPI inference service (:mod:`oilspill.api`).

The fast suite uses a tiny in-process 1x1-conv ONNX model (no 110MB download) and
a mocked scene runner (no network), so the whole module runs in a couple of
seconds. The real-model predict path is covered by a single ``slow`` test that is
skipped unless an exported model exists.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from oilspill.api.app import create_app
from oilspill.api.models import JobResult
from oilspill.api.service import ModelRegistry
from oilspill.api.settings import Settings
from oilspill.metrics import CLASS_NAMES, NUM_CLASSES
from oilspill.pipeline.infer import INPUT_NAME, OUTPUT_NAME

if TYPE_CHECKING:
    from collections.abc import Iterator


# --- fixtures ----------------------------------------------------------------


class _PointwiseSeg(torch.nn.Module):
    """Trivial fully-pointwise segmentation head: a 1x1 conv to NUM_CLASSES."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, NUM_CLASSES, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _export_tiny_onnx(out_path: Path) -> None:
    torch.manual_seed(0)
    model = _PointwiseSeg().eval()
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


def _write_results_json(path: Path, run_name: str, tag: str) -> None:
    data = {
        "meta": {"run_name": run_name, "tag": tag, "source_type": "checkpoint"},
        "metrics": {
            "class_names": list(CLASS_NAMES),
            "per_class": {
                "iou": {name: 0.5 for name in CLASS_NAMES},
                "precision": {name: 0.6 for name in CLASS_NAMES},
                "recall": {name: 0.7 for name in CLASS_NAMES},
                "f1": {name: 0.65 for name in CLASS_NAMES},
            },
            "aggregate": {
                "mean_iou": 0.5,
                "macro_f1": 0.65,
                "pixel_accuracy": 0.9,
                "oil_iou": 0.5,
                "oil_recall": 0.7,
            },
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """A self-contained Settings pointing at temp results/onnx/samples dirs."""
    root = tmp_path_factory.mktemp("api-env")
    results_dir = root / "results"
    onnx_dir = root / "exports"
    samples_dir = root / "samples"
    for d in (results_dir, onnx_dir, samples_dir):
        d.mkdir()

    # Two real models + one smoke run that must be excluded from /models.
    _write_results_json(results_dir / "tiny-a.json", "tiny-a", "tiny-a")
    _write_results_json(results_dir / "tiny-b.json", "tiny-b", "tiny-b")
    _write_results_json(results_dir / "smoke.json", "run-smoke", "smoke")

    _export_tiny_onnx(onnx_dir / "model.onnx")

    # A couple of small sample images.
    for i in (1, 2):
        img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB")
        img.save(samples_dir / f"sample_0{i}.png")

    return Settings(
        onnx_dir=onnx_dir,
        default_onnx=onnx_dir / "model.onnx",
        results_dir=results_dir,
        samples_dir=samples_dir,
        web_dist=root / "no-web",  # absent -> static mount skipped
        tile_size=64,
        overlap=16,
        batch_size=4,
    )


def _mock_runner(
    aoi: dict[str, Any],
    start: str,
    end: str,
    model_id: str | None,
    registry: ModelRegistry,
) -> JobResult:
    return JobResult(
        num_oil_polygons=2,
        total_oil_area_km2=1.25,
        geojson={"type": "FeatureCollection", "features": []},
    )


@pytest.fixture
def client(env: Settings) -> Iterator[TestClient]:
    app = create_app(env, scene_runner=_mock_runner)
    with TestClient(app) as test_client:
        yield test_client


# --- /healthz ----------------------------------------------------------------


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- /models -----------------------------------------------------------------


def test_models_lists_real_metrics_excludes_smoke(client: TestClient) -> None:
    resp = client.get("/models")
    assert resp.status_code == 200
    models = resp.json()["models"]
    ids = {m["id"] for m in models}
    assert ids == {"tiny-a", "tiny-b"}  # smoke run excluded

    model = next(m for m in models if m["id"] == "tiny-a")
    assert model["oil_iou"] == 0.5
    assert model["oil_recall"] == 0.7
    assert model["mean_iou"] == 0.5
    assert model["macro_f1"] == 0.65
    assert model["pixel_accuracy"] == 0.9
    assert model["available"] is True  # default model.onnx is loadable
    assert set(model["per_class"]) == set(CLASS_NAMES)
    assert model["per_class"]["Oil Spill"]["iou"] == 0.5


def test_models_available_false_without_onnx(env: Settings, tmp_path: Path) -> None:
    cfg = env.model_copy(
        update={
            "onnx_dir": tmp_path / "empty",
            "default_onnx": tmp_path / "empty" / "model.onnx",
        }
    )
    app = create_app(cfg)
    with TestClient(app) as c:
        models = c.get("/models").json()["models"]
    assert all(m["available"] is False for m in models)


# --- /samples ----------------------------------------------------------------


def test_samples_list_and_serve(client: TestClient) -> None:
    resp = client.get("/samples")
    assert resp.status_code == 200
    samples = resp.json()["samples"]
    assert {s["id"] for s in samples} == {"sample_01", "sample_02"}

    url = samples[0]["url"]
    assert url.startswith("/samples/")
    img_resp = client.get(url)
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"].startswith("image/")


def test_sample_not_found(client: TestClient) -> None:
    assert client.get("/samples/nope.png").status_code == 404


def test_sample_traversal_rejected(client: TestClient) -> None:
    # Encoded traversal resolves to a name with a slash -> rejected as 400/404.
    assert client.get("/samples/..%2Fsecret.txt").status_code in {400, 404}


# --- /predict ----------------------------------------------------------------


def _png_upload(size: int = 256) -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _decode_data_uri_png(uri: str) -> Image.Image:
    assert uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    return Image.open(io.BytesIO(raw))


def test_predict_returns_valid_pngs_and_percentages(client: TestClient) -> None:
    files = {"file": ("test.png", _png_upload(256), "image/png")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["width"] == 256
    assert body["height"] == 256
    assert body["model"] in {"tiny-a", "tiny-b"}

    # Class percentages cover every class and sum to ~100%.
    assert set(body["class_percentages"]) == set(CLASS_NAMES)
    assert sum(body["class_percentages"].values()) == pytest.approx(100.0, abs=0.5)

    # Legend present for every class as [r, g, b].
    assert set(body["legend"]) == set(CLASS_NAMES)
    assert all(len(rgb) == 3 for rgb in body["legend"].values())

    # Both PNGs decode and the overlay matches the original image size.
    mask_img = _decode_data_uri_png(body["mask_png"])
    overlay_img = _decode_data_uri_png(body["overlay_png"])
    assert mask_img.size == (256, 256)
    assert overlay_img.size == (256, 256)


def test_predict_with_model_field(client: TestClient) -> None:
    files = {"file": ("test.png", _png_upload(128), "image/png")}
    resp = client.post("/predict", files=files, data={"model": "tiny-b"})
    assert resp.status_code == 200
    assert resp.json()["model"] == "tiny-b"


def test_predict_bad_image(client: TestClient) -> None:
    files = {"file": ("bad.png", b"not an image", "image/png")}
    assert client.post("/predict", files=files).status_code == 400


def test_predict_503_without_model(env: Settings, tmp_path: Path) -> None:
    cfg = env.model_copy(
        update={
            "onnx_dir": tmp_path / "none",
            "default_onnx": tmp_path / "none" / "model.onnx",
        }
    )
    app = create_app(cfg)
    with TestClient(app) as c:
        files = {"file": ("test.png", _png_upload(64), "image/png")}
        assert c.post("/predict", files=files).status_code == 503


# --- /jobs -------------------------------------------------------------------


def test_scene_job_lifecycle(client: TestClient) -> None:
    body = {
        "aoi": {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]},
        "start": "2024-01-01",
        "end": "2024-01-31",
        "model": "tiny-a",
    }
    resp = client.post("/jobs/scene", json=body)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "queued"
    job_id = payload["job_id"]

    # TestClient runs BackgroundTasks synchronously after the response, so by the
    # time we poll the job has reached a terminal state.
    status = client.get(f"/jobs/{job_id}").json()
    assert status["job_id"] == job_id
    assert status["status"] == "done"
    assert status["result"]["num_oil_polygons"] == 2
    assert status["result"]["total_oil_area_km2"] == 1.25


def test_scene_job_error_captured() -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> JobResult:
        raise RuntimeError("no scene found")

    cfg = Settings(results_dir=Path("docs/results"))
    app = create_app(cfg, scene_runner=_boom)
    with TestClient(app) as c:
        body = {"aoi": {}, "start": "2024-01-01", "end": "2024-01-31"}
        job_id = c.post("/jobs/scene", json=body).json()["job_id"]
        status = c.get(f"/jobs/{job_id}").json()
    assert status["status"] == "error"
    assert "no scene found" in status["detail"]


def test_unknown_job_404(client: TestClient) -> None:
    assert client.get("/jobs/does-not-exist").status_code == 404


# --- app factory + real model ------------------------------------------------


def test_create_app_title() -> None:
    assert create_app(Settings(results_dir=Path("docs/results"))).title == "Oil Spill Detection API"


def test_models_reads_real_results_dir() -> None:
    """The real docs/results JSONs parse and exclude the smoke run."""
    app = create_app(Settings(results_dir=Path("docs/results")))
    with TestClient(app) as c:
        ids = {m["id"] for m in c.get("/models").json()["models"]}
    assert "segformer-mit-b2" in ids
    assert not any("smoke" in i for i in ids)


@pytest.mark.slow
def test_predict_with_real_onnx() -> None:
    onnx = Path("artifacts/exports/model.onnx")
    if not onnx.exists():
        pytest.skip("real ONNX model not present")
    app = create_app(Settings())
    with TestClient(app) as c:
        files = {"file": ("test.png", _png_upload(256), "image/png")}
        resp = c.post("/predict", files=files)
    assert resp.status_code == 200
    assert sum(resp.json()["class_percentages"].values()) == pytest.approx(100.0, abs=0.5)
