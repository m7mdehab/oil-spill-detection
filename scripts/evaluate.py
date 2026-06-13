"""Command-line entry point for evaluating a checkpoint or ONNX model.

Runs a model over a dataset split and produces the project's reporting artifacts:
``metrics.json``, ``confusion_matrix.png``, ``pr_curves.png``, a ``gallery/`` of
best/worst predictions, and an updated results table in ``docs/results.md``.

Examples
--------
Evaluate a checkpoint on the test split (fast smoke subset)::

    python scripts/evaluate.py --checkpoint artifacts/checkpoints/<run>/best.pt \\
        --split test --max-images 30 --tag smoke

Evaluate an exported ONNX model on the full test split::

    python scripts/evaluate.py --onnx artifacts/exports/model.onnx --tag unet-baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Allow running as a plain script (``python scripts/evaluate.py``) without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from oilspill.data import OilSpillDataset, build_transforms
from oilspill.evaluation import (
    evaluate_model,
    load_model_from_checkpoint,
    load_onnx_session,
    plot_confusion_matrix,
    plot_pr_curves,
    save_prediction_gallery,
    update_results_markdown,
    write_results_json,
)
from oilspill.evaluation.model_loading import ModelOrSession

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint or ONNX model.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Path to a trainer checkpoint (best.pt).")
    source.add_argument("--onnx", type=Path, help="Path to an exported ONNX model.")
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="test", help="Dataset split."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/datasets/oil_spill"),
        help="Dataset root (contains train/ and test/).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Square eval image size. Defaults to the checkpoint's data.image_size.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to artifacts/eval/<name>.",
    )
    parser.add_argument(
        "--results-md", type=Path, default=Path("docs/results.md"), help="Results table to update."
    )
    parser.add_argument("--tag", default="", help="Run tag, e.g. 'smoke' or 'unet-baseline'.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--n-best", type=int, default=4, help="Best gallery panels.")
    parser.add_argument("--n-worst", type=int, default=4, help="Worst gallery panels.")
    parser.add_argument(
        "--max-images", type=int, default=None, help="Limit images (fast smoke eval)."
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name used for the out-dir and results row. Defaults to the model stem.",
    )
    return parser.parse_args(argv)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


def _resolve_image_size(arg_size: int | None, config: dict[str, Any] | None) -> tuple[int, int]:
    """Resolve the eval (height, width). Config ``image_size`` may be int or list."""
    if arg_size is not None:
        return (arg_size, arg_size)
    if config is not None:
        cfg_size = config.get("data", {}).get("image_size")
        if isinstance(cfg_size, int):
            return (cfg_size, cfg_size)
        if isinstance(cfg_size, (list, tuple)) and len(cfg_size) == 2:
            return (int(cfg_size[0]), int(cfg_size[1]))
    return (256, 256)


def _default_name(args: argparse.Namespace) -> str:
    if args.name:
        return args.name
    model_path: Path = args.checkpoint or args.onnx
    # Use the parent run dir name when the file is a generic 'best.pt'.
    if model_path.stem in {"best", "last", "model"} and model_path.parent.name:
        return model_path.parent.name
    return model_path.stem


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    device = _resolve_device(args.device)

    model: ModelOrSession
    config: dict[str, Any] | None = None
    if args.checkpoint is not None:
        model, config = load_model_from_checkpoint(args.checkpoint, device=device)
        logger.info("loaded checkpoint %s (arch=%s)", args.checkpoint, config["model"]["arch"])
    else:
        model = load_onnx_session(args.onnx)
        logger.info("loaded ONNX model %s", args.onnx)

    image_size = _resolve_image_size(args.image_size, config)
    run_name = _default_name(args)
    out_dir: Path = args.out_dir or Path("artifacts/eval") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transforms(train=False, image_size=image_size)
    dataset = OilSpillDataset(args.data_root, args.split, transform=transform)
    logger.info("evaluating split=%s (%d images) at size=%s", args.split, len(dataset), image_size)

    output = evaluate_model(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        max_images=args.max_images,
        collect_pr=True,
    )
    result = output.result

    # Metrics JSON -- the single source for every number in the docs.
    json_path = out_dir / "metrics.json"
    meta = {
        "run_name": run_name,
        "tag": args.tag,
        "source": str(args.checkpoint or args.onnx).replace("\\", "/"),
        "source_type": "checkpoint" if args.checkpoint else "onnx",
        "split": args.split,
        "image_size": list(image_size),
        "num_images": output.num_images,
        "device": str(device),
    }
    write_results_json(result, json_path, meta)

    # Figures.
    cm_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(result, cm_path)
    pr_path = out_dir / "pr_curves.png"
    if output.pr_probs is not None and output.pr_targets is not None:
        plot_pr_curves(output.pr_probs, output.pr_targets, result.class_names, pr_path)

    # Gallery.
    gallery_dir = out_dir / "gallery"
    panels = save_prediction_gallery(
        model,
        dataset,
        output.per_image_oil_iou,
        gallery_dir,
        n_best=args.n_best,
        n_worst=args.n_worst,
    )

    # Results table.
    update_results_markdown(
        args.results_md,
        run_name,
        result,
        json_path,
        tag=args.tag,
        num_images=output.num_images,
    )

    print("\n=== evaluation complete ===")
    print(f"run name      : {run_name}{f'  [{args.tag}]' if args.tag else ''}")
    print(f"images        : {output.num_images} ({args.split} split)")
    print(f"oil IoU       : {result.oil_iou:.4f}")
    print(f"oil recall    : {result.oil_recall:.4f}")
    print(f"mean IoU      : {result.mean_iou:.4f}")
    print(f"macro F1      : {result.macro_f1:.4f}")
    print(f"pixel accuracy: {result.pixel_accuracy:.4f}")
    print(f"metrics JSON  : {json_path}")
    print(f"confusion png : {cm_path}")
    print(f"pr curves png : {pr_path}")
    print(f"gallery       : {len(panels)} panels in {gallery_dir}")
    print(f"results table : {args.results_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
