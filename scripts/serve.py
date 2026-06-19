"""Uvicorn launcher for the oil-spill inference API.

Run with ``uv run python scripts/serve.py``. Host and port come from the
environment (``HOST`` / ``PORT``), defaulting to ``0.0.0.0:7860`` -- the port
Hugging Face Spaces expects.
"""

from __future__ import annotations

import os

import uvicorn

from oilspill.api.app import create_app

# Built once at import so a single worker reuses the cached ONNX sessions and the
# in-process job store across requests.
app = create_app()


def main() -> None:
    """Start the ASGI server."""
    # Bind all interfaces so the container is reachable; the deployment fronts it.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
