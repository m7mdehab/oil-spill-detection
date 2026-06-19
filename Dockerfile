# syntax=docker/dockerfile:1

# ---- Stage 1: build the web frontend ----
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
# `npm install` (not `npm ci`): the committed lock may have been generated on a
# different OS and omit the host's platform-specific optional deps (e.g. rollup's
# native binary), which breaks the Vite build on linux. install resolves them.
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build   # -> /web/dist

# ---- Stage 2: python runtime serving the API + static frontend ----
# Official uv image (uv + a system Python 3.11 preinstalled) -- the documented,
# reliable base for uv-managed projects.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS runtime

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PORT=7860 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching). The serving dependency set
# (no training/eval extras) is all prebuilt wheels, so no compiler is needed. torch resolves to the CPU
# wheel index pinned in pyproject; rasterio/geopandas/onnxruntime ship manylinux
# wheels that bundle their native libs, so no system GDAL is required.
# LICENSE is required: pyproject sets license = { file = "LICENSE" }, which
# hatchling validates while building the project during uv sync.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Application code, the built frontend, and the small committed assets the API
# serves: preloaded sample images (/samples) and the per-model metrics JSONs that
# back /models.
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/samples/ ./data/samples/
COPY docs/results/ ./docs/results/
COPY --from=web /web/dist ./web/dist

# A trained ONNX model is not baked into the image (weights live on the Hugging
# Face Hub, not in git). The entrypoint fetches it at startup when
# OILSPILL_MODEL_HF_REPO is set; without it the API still serves and returns a
# clear 503 from /predict until a model is available.
# Put the synced virtualenv on PATH and use it directly. We invoke the venv's
# python rather than `uv run` so the container never tries to re-sync (which would
# pull the dev/ml groups) at startup.
ENV PATH="/app/.venv/bin:$PATH" \
    OILSPILL_API_ONNX_DIR=/app/artifacts/exports \
    OILSPILL_API_WEB_DIST=/app/web/dist
RUN mkdir -p /app/artifacts/exports

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/healthz').status==200 else 1)"

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "scripts/serve.py"]
