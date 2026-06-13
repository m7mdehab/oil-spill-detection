SHELL := sh

DATA_ROOT ?= data/datasets/oil_spill
SMOKE_CKPT = $$(ls -t artifacts/checkpoints/*/best.pt 2>/dev/null | head -1)

.PHONY: check fmt test lock data train-smoke evaluate-smoke

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
