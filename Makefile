SHELL := sh

DATA_ROOT ?= data/datasets/oil_spill
SMOKE_CKPT = $$(ls -t artifacts/checkpoints/*/best.pt 2>/dev/null | head -1)

# Best model selected by oil-class IoU (see _pm/DECISIONS.md D029): SegFormer mit-b2.
BEST_CKPT ?= artifacts/checkpoints/run-20260619-090026/best.pt
BEST_RUN_JSON ?= docs/results/segformer-mit-b2.json
ONNX_OUT ?= artifacts/exports/model.onnx
HF_REPO ?= m7mdehab/oil-spill-segmentation

.PHONY: check fmt test lock data train-smoke evaluate-smoke export-onnx publish

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	uv run pytest -m "not slow"

fmt:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

lock:
	uv lock

# --- data & smoke reproduction ---
# Verify the dataset archive checksum, extract it, and (re)generate the data report.
data:
	uv run python scripts/make_data.py

# Fast CPU sanity training: 2 epochs on ~10% of the data; writes a checkpoint + MLflow run.
train-smoke:
	uv run python scripts/train.py --config configs/unet_smoke.yaml --smoke

# Evaluate the most recent checkpoint on a small slice of the test split; writes
# metrics/plots/gallery and updates docs/results.md.
evaluate-smoke:
	uv run python scripts/evaluate.py --checkpoint $(SMOKE_CKPT) --split test --max-images 30 --tag smoke

# --- packaging & publish ---
# Export the selected best checkpoint to ONNX with a parity check against PyTorch.
export-onnx:
	uv run python scripts/export_onnx.py --checkpoint $(BEST_CKPT) --out $(ONNX_OUT)

# Export then publish the best model to the Hugging Face Hub with an honest model
# card. Defaults to --dry-run (no network); pass DRY_RUN= to actually push:
#   make publish DRY_RUN=
DRY_RUN ?= --dry-run
publish: export-onnx
	uv run python scripts/publish_hf.py --onnx $(ONNX_OUT) --checkpoint $(BEST_CKPT) \
		--run-json $(BEST_RUN_JSON) --repo-id $(HF_REPO) $(DRY_RUN)
