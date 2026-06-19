#!/usr/bin/env bash
# Optionally fetch the trained ONNX model from the Hugging Face Hub before
# starting the server. Set OILSPILL_MODEL_HF_REPO (e.g. "m7mdehab/oil-spill-segmentation")
# and optionally OILSPILL_MODEL_FILE (default model.onnx). Without these the API
# still starts and serves; /predict returns 503 until a model is present.
set -euo pipefail

ONNX_DIR="${OILSPILL_API_ONNX_DIR:-/app/artifacts/exports}"
MODEL_FILE="${OILSPILL_MODEL_FILE:-model.onnx}"
mkdir -p "$ONNX_DIR"

if [ -n "${OILSPILL_MODEL_HF_REPO:-}" ] && [ ! -f "$ONNX_DIR/$MODEL_FILE" ]; then
  echo "fetching $MODEL_FILE from HF repo $OILSPILL_MODEL_HF_REPO ..."
  python - <<'PY' || echo "model fetch failed; the API will serve without a model"
import os
from huggingface_hub import hf_hub_download
repo = os.environ["OILSPILL_MODEL_HF_REPO"]
fname = os.environ.get("OILSPILL_MODEL_FILE", "model.onnx")
dest = os.environ.get("OILSPILL_API_ONNX_DIR", "/app/artifacts/exports")
path = hf_hub_download(repo_id=repo, filename=fname, local_dir=dest)
print("downloaded", path)
PY
fi

exec "$@"
