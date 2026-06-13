SHELL := sh

.PHONY: check fmt test lock

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
