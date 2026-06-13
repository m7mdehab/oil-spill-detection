"""Command-line entry point for training a segmentation model.

Examples
--------
Full run from a config::

    python scripts/train.py --config configs/unet.yaml

Fast CPU smoke run (2 epochs, ~10% data, no AMP)::

    python scripts/train.py --config configs/unet_smoke.yaml --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as a plain script (``python scripts/train.py``) without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oilspill.training import (
    SMOKE_SUBSET_FRACTION,
    TrainConfig,
    fit,
    make_smoke_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML train config.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast CPU sanity run: 2 epochs, ~10%% of data, tiny batches, no AMP.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override runtime.device from the config.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override optim.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override optim.lr.")
    parser.add_argument("--seed", type=int, default=None, help="Override runtime.seed.")
    return parser.parse_args(argv)


def _apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.device is not None:
        cfg = cfg.model_copy(
            update={"runtime": cfg.runtime.model_copy(update={"device": args.device})}
        )
    if args.epochs is not None:
        cfg = cfg.model_copy(update={"optim": cfg.optim.model_copy(update={"epochs": args.epochs})})
    if args.batch_size is not None:
        cfg = cfg.model_copy(
            update={"data": cfg.data.model_copy(update={"batch_size": args.batch_size})}
        )
    if args.lr is not None:
        cfg = cfg.model_copy(update={"optim": cfg.optim.model_copy(update={"lr": args.lr})})
    if args.seed is not None:
        cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"seed": args.seed})})
    return cfg


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    cfg = TrainConfig.from_yaml(args.config)

    subset_fraction: float | None = None
    if args.smoke:
        cfg = make_smoke_config(cfg)
        subset_fraction = SMOKE_SUBSET_FRACTION
    cfg = _apply_overrides(cfg, args)

    result = fit(cfg, subset_fraction=subset_fraction)

    print("\n=== training complete ===")
    print(f"mlflow run id : {result.mlflow_run_id}")
    print(f"best checkpoint: {result.best_checkpoint}")
    print(f"last checkpoint: {result.last_checkpoint}")
    print(f"best {cfg.checkpoint.monitor}: {result.best_metric:.4f}")
    print("final metrics :")
    for key in sorted(result.metrics):
        print(f"  {key}: {result.metrics[key]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
