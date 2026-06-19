# syntax=docker/dockerfile:1

# ---- Stage 1: build the web frontend ----
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build   # -> /web/dist

# ---- Stage 2: python runtime serving the API + static frontend ----
FROM python:3.11-slim-bookworm AS runtime

# uv for fast, locked installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PORT=7860 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching). torch resolves to the CPU
# wheel index pinned in pyproject; rasterio/geopandas/onnxruntime ship manylinux
# wheels that bundle their native libs, so no system GDAL is required.
COPY pyproject.toml uv.lock README.md ./
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
ENV OILSPILL_API_ONNX_DIR=/app/artifacts/exports \
    OILSPILL_API_WEB_DIST=/app/web/dist
RUN mkdir -p /app/artifacts/exports

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/healthz').status==200 else 1)"

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uv", "run", "python", "scripts/serve.py"]
