"""Command-line entry point for exporting a checkpoint to ONNX.

Exports a trainer checkpoint to an ONNX graph (legacy ``torch.onnx.export``,
dynamic batch and spatial axes) and then verifies that the ONNX graph reproduces
the torch model's logits to within a tolerance, printing the max absolute
difference and PASS/FAIL.

Examples
--------
Export the most recent checkpoint::

    python scripts/export_onnx.py --checkpoint artifacts/checkpoints/<run>/best.pt

Export to an explicit path at a chosen opset::

    python scripts/export_onnx.py --checkpoint <ckpt> --out artifacts/exports/model.onnx --opset 17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Allow running as a plain script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oilspill.evaluation.model_loading import load_model_from_checkpoint
from oilspill.packaging.onnx_export import export_to_onnx, verify_parity

# Parity tolerance: tracing through the legacy exporter should match the torch
# model to well within this on CPU float32.
PARITY_ATOL = 1e-4


def _resolve_image_size(arg_size: int | None, config: dict[str, Any]) -> int:
    """Resolve a square image size, defaulting to the checkpoint's data.image_size."""
    if arg_size is not None:
        return arg_size
    cfg_size = config.get("data", {}).get("image_size")
    if isinstance(cfg_size, int):
        return cfg_size
    if isinstance(cfg_size, (list, tuple)) and len(cfg_size) == 2:
        return int(cfg_size[0])
    return 256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX and verify parity.")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to a trainer checkpoint (best.pt)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .onnx path. Defaults to artifacts/exports/<run>.onnx.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Square trace size. Defaults to the checkpoint's data.image_size.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    return parser.parse_args(argv)


def _default_out(checkpoint: Path) -> Path:
    """Default ONNX path derived from the checkpoint's run directory name."""
    stem = (
        checkpoint.parent.name if checkpoint.stem in {"best", "last", "model"} else checkpoint.stem
    )
    return Path("artifacts/exports") / f"{stem}.onnx"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Load the config once to resolve the default image size honestly.
    _model, config = load_model_from_checkpoint(args.checkpoint, device="cpu")
    image_size = _resolve_image_size(args.image_size, config)
    out_path: Path = args.out or _default_out(args.checkpoint)

    print(f"checkpoint : {args.checkpoint}")
    print(f"arch       : {config['model'].get('arch', '?')}")
    print(f"image size : {image_size}x{image_size} (dynamic batch + H/W in graph)")
    print(f"opset      : {args.opset}")

    written = export_to_onnx(args.checkpoint, out_path, image_size=image_size, opset=args.opset)
    print(f"exported   : {written}")

    max_diff = verify_parity(args.checkpoint, written, image_size=image_size, n=8, atol=PARITY_ATOL)
    passed = max_diff < PARITY_ATOL
    print(f"parity max abs diff: {max_diff:.3e} (atol {PARITY_ATOL:.0e})")
    print("parity: PASS" if passed else "parity: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
