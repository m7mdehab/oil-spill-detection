"""Publish an exported model to the Hugging Face Hub.

Creates (or reuses) a Hugging Face model repo and uploads the ONNX file, a
generated model card (``README.md``), and the metrics JSON. The model card is
built from the committed metrics JSON so its numbers are traceable.

The token is read from ``HF_TOKEN`` in ``.env`` (never printed). A ``--dry-run``
mode builds the card and lists exactly what would be uploaded **without any
network calls**; it is the default so tests and CI never hit the network.

Examples
--------
Dry run (default, no network)::

    python scripts/publish_hf.py --onnx artifacts/exports/model.onnx \\
        --checkpoint <ckpt> --run-json docs/results/unet-r34-baseline.json \\
        --repo-id m7mdehab/oil-spill-segmentation --dry-run

Actually publish (requires HF_TOKEN with write access)::

    python scripts/publish_hf.py --onnx artifacts/exports/model.onnx \\
        --checkpoint <ckpt> --run-json docs/results/unet-r34-baseline.json \\
        --repo-id m7mdehab/oil-spill-segmentation --no-dry-run --private
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oilspill.evaluation.model_loading import load_model_from_checkpoint
from oilspill.packaging.model_card import build_model_card

# Name the README/model card and metrics file take inside the repo.
CARD_FILENAME = "README.md"
METRICS_FILENAME = "metrics.json"


def _arch_label(config: dict[str, Any]) -> str:
    """Human-readable architecture label from a checkpoint config block."""
    model = config.get("model", {})
    arch = model.get("arch", "model")
    encoder = model.get("encoder_name") or model.get("encoder")
    if encoder:
        return f"{arch} ({encoder} encoder)"
    return str(arch)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish an ONNX model to the Hugging Face Hub.")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to the exported ONNX file.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint the ONNX was exported from (for the architecture label).",
    )
    parser.add_argument(
        "--run-json",
        type=Path,
        required=True,
        help="Committed metrics JSON (docs/results/<run>.json) for the model card.",
    )
    parser.add_argument(
        "--repo-id", required=True, help="Target HF repo id, e.g. user/oil-spill-segmentation."
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private", dest="private", action="store_true", help="Create a private repo (default)."
    )
    visibility.add_argument(
        "--public", dest="private", action="store_false", help="Create a public repo."
    )
    parser.set_defaults(private=True)
    dry = parser.add_mutually_exclusive_group()
    dry.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Build the card and list files without any network calls (default).",
    )
    dry.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Actually create the repo and upload.",
    )
    parser.set_defaults(dry_run=True)
    return parser.parse_args(argv)


def _build_card(args: argparse.Namespace) -> str:
    """Build the model card markdown from the checkpoint + run JSON."""
    _model, config = load_model_from_checkpoint(args.checkpoint, device="cpu")
    return build_model_card(
        args.run_json,
        repo_id=args.repo_id,
        arch=_arch_label(config),
        onnx_filename=args.onnx.name,
    )


def _planned_uploads(args: argparse.Namespace) -> list[tuple[str, str]]:
    """The ``(local_source, path_in_repo)`` pairs that would be uploaded."""
    return [
        (str(args.onnx), args.onnx.name),
        ("<generated model card>", CARD_FILENAME),
        (str(args.run_json), METRICS_FILENAME),
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.onnx.exists():
        print(f"error: ONNX file not found: {args.onnx}", file=sys.stderr)
        return 1
    if not args.run_json.exists():
        print(f"error: run JSON not found: {args.run_json}", file=sys.stderr)
        return 1

    card = _build_card(args)
    uploads = _planned_uploads(args)
    visibility = "private" if args.private else "public"

    print(f"repo id    : {args.repo_id} ({visibility})")
    print(f"model card : {len(card)} chars built from {args.run_json}")
    print("files to upload:")
    for source, dest in uploads:
        print(f"  - {dest}  <-  {source}")

    if args.dry_run:
        print("\n[dry-run] no network calls made; nothing was uploaded.")
        return 0

    # --- real publish path (only reached with --no-dry-run) ---
    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("error: HF_TOKEN not set (put it in .env)", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(args.onnx),
        path_in_repo=args.onnx.name,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add ONNX model",
    )
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo=CARD_FILENAME,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add model card",
    )
    api.upload_file(
        path_or_fileobj=str(args.run_json),
        path_in_repo=METRICS_FILENAME,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add evaluation metrics",
    )
    print(f"\npublished to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
