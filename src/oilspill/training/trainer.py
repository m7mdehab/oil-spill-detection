"""Config-driven training engine for the segmentation task.

A minimal, dependency-light custom training loop (no PyTorch Lightning -- it is
not a project dependency). It supports AMP (a no-op on CPU), cosine/step
schedulers, early stopping, best/last checkpointing under ``artifacts/`` and
MLflow logging of the flattened config, per-epoch metrics, and a few sample
prediction images. All metric computation routes through
:class:`oilspill.metrics.SegmentationMetrics`.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from oilspill.metrics import (
    CLASS_NAMES,
    NUM_CLASSES,
    SegmentationMetrics,
    logits_to_labels,
)
from oilspill.models import build_model as registry_build_model
from oilspill.training.config import TrainConfig
from oilspill.training.datasets import (
    SyntheticSegmentationDataset,
    load_real_datasets,
)
from oilspill.training.losses import build_loss, compute_auto_class_weights
from oilspill.training.seed import seed_everything

logger = logging.getLogger(__name__)

SegDataset = Dataset[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class FitResult:
    """Outcome of a training run."""

    best_checkpoint: Path
    last_checkpoint: Path
    best_metric: float
    metrics: dict[str, float]
    mlflow_run_id: str


def resolve_device(requested: str) -> torch.device:
    """Resolve ``"auto"``/``"cpu"``/``"cuda"`` to an available device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


def build_model(cfg: TrainConfig) -> nn.Module:
    """Instantiate the segmentation model from config via the model registry."""
    return registry_build_model(cfg.model, pretrained=True)


def build_optimizer(cfg: TrainConfig, params: object) -> torch.optim.Optimizer:
    name = cfg.optim.optimizer
    kwargs = {"lr": cfg.optim.lr, "weight_decay": cfg.optim.weight_decay}
    if name == "adam":
        return torch.optim.Adam(params, **kwargs)  # type: ignore[arg-type]
    if name == "adamw":
        return torch.optim.AdamW(params, **kwargs)  # type: ignore[arg-type]
    if name == "sgd":
        return torch.optim.SGD(params, momentum=0.9, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown optimizer: {name!r}")


def build_scheduler(
    cfg: TrainConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler | None:
    sched = cfg.optim.scheduler
    if sched == "none":
        return None
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optim.epochs)
    if sched == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.optim.step_size, gamma=cfg.optim.gamma
        )
    raise ValueError(f"unknown scheduler: {sched!r}")


def _subset(dataset: SegDataset, fraction: float, seed: int) -> SegDataset:
    """Deterministically take roughly ``fraction`` of a dataset (>= 1 item)."""
    n = len(dataset)  # type: ignore[arg-type]
    k = max(1, round(n * fraction))
    if k >= n:
        return dataset
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=gen)[:k].tolist()
    return Subset(dataset, indices)


def _flatten_metrics(prefix: str, result: object) -> dict[str, float]:
    """Pull scalar metrics out of a :class:`MetricResult` with a name prefix."""
    from oilspill.metrics import MetricResult

    assert isinstance(result, MetricResult)
    out = {
        f"{prefix}_mean_iou": result.mean_iou,
        f"{prefix}_oil_iou": result.oil_iou,
        f"{prefix}_oil_recall": result.oil_recall,
        f"{prefix}_macro_f1": result.macro_f1,
        f"{prefix}_macro_precision": result.macro_precision,
        f"{prefix}_macro_recall": result.macro_recall,
        f"{prefix}_pixel_accuracy": result.pixel_accuracy,
    }
    return {k: v for k, v in out.items() if not np.isnan(v)}


def _save_checkpoint(path: Path, model: nn.Module, cfg: TrainConfig, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.model_dump(mode="json"),
            "epoch": epoch,
            "class_names": list(CLASS_NAMES),
        },
        path,
    )


def _log_sample_predictions(
    model: nn.Module,
    dataset: SegDataset,
    device: torch.device,
    n: int,
    epoch: int,
) -> None:
    """Log up to ``n`` (input, ground-truth, prediction) panels to MLflow."""
    if n <= 0:
        return
    model.eval()
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        for i in range(min(n, len(dataset))):  # type: ignore[arg-type]
            image, mask = dataset[i]
            logits = model(image.unsqueeze(0).to(device))
            pred = logits_to_labels(logits)[0].cpu()
            panel = _make_panel(image, mask, pred)
            out_path = Path(tmp) / f"epoch{epoch:03d}_sample{i}.png"
            _save_png(panel, out_path)
            mlflow.log_artifact(str(out_path), artifact_path="samples")


def _make_panel(image: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor) -> np.ndarray:
    """Build an RGB strip: input | ground truth | prediction (uint8)."""
    img = image[0] if image.ndim == 3 else image
    img = img.cpu().float()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_rgb = (img.numpy() * 255).astype(np.uint8)
    img_rgb = np.stack([img_rgb] * 3, axis=-1)

    scale = 255 // max(1, NUM_CLASSES - 1)
    gt_rgb = np.stack([(mask.cpu().numpy() * scale).astype(np.uint8)] * 3, axis=-1)
    pred_rgb = np.stack([(pred.cpu().numpy() * scale).astype(np.uint8)] * 3, axis=-1)
    return np.concatenate([img_rgb, gt_rgb, pred_rgb], axis=1)


def _save_png(array: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(array).save(path)


def fit(
    cfg: TrainConfig,
    *,
    train_dataset: SegDataset | None = None,
    val_dataset: SegDataset | None = None,
    subset_fraction: float | None = None,
) -> FitResult:
    """Train a model according to ``cfg`` and return the run outcome.

    Datasets may be injected (used by tests and the synthetic fallback);
    otherwise they are built from :mod:`oilspill.data`. ``subset_fraction``,
    when set, keeps roughly that fraction of each split (smoke mode).
    """
    seed_everything(cfg.runtime.seed, deterministic=cfg.runtime.deterministic)
    device = resolve_device(cfg.runtime.device)

    train_ds, val_ds = _resolve_datasets(cfg, train_dataset, val_dataset)
    if subset_fraction is not None:
        train_ds = _subset(train_ds, subset_fraction, cfg.data.split_seed)
        val_ds = _subset(val_ds, subset_fraction, cfg.data.split_seed + 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    model = build_model(cfg).to(device)
    loss_fn = _build_loss_fn(cfg, train_ds, device)
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, optimizer)

    use_amp = cfg.runtime.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if cfg.runtime.precision == "bf16" else torch.float16

    # Recent MLflow gates the local file store behind an opt-in env var. We log
    # to a local, gitignored ./mlruns by design (no DB dependency available), so
    # enable it explicitly unless the user already configured a backend.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    run_name = cfg.mlflow.run_name or datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
    ckpt_dir = cfg.checkpoint.dir / run_name

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(cfg.flatten())
        run_id = run.info.run_id

        best_metric = -float("inf") if cfg.checkpoint.mode == "max" else float("inf")
        best_epoch = 0
        epochs_no_improve = 0
        last_metrics: dict[str, float] = {}
        best_ckpt = ckpt_dir / "best.pt"
        last_ckpt = ckpt_dir / "last.pt"

        for epoch in range(1, cfg.optim.epochs + 1):
            train_loss = _train_one_epoch(
                model, train_loader, loss_fn, optimizer, scaler, device, amp_dtype, use_amp, epoch
            )
            if scheduler is not None:
                scheduler.step()

            val_result = _validate(model, val_loader, device, cfg.model.num_classes)
            metrics = _flatten_metrics("val", val_result)
            metrics["train_loss"] = train_loss
            metrics["lr"] = optimizer.param_groups[0]["lr"]
            mlflow.log_metrics(metrics, step=epoch)
            last_metrics = metrics

            current = metrics.get(cfg.checkpoint.monitor)
            improved = False
            if current is not None:
                improved = _is_improvement(
                    current, best_metric, cfg.checkpoint.mode, cfg.early_stopping.min_delta
                )
            logger.info(
                "epoch %d/%d train_loss=%.4f %s=%s%s",
                epoch,
                cfg.optim.epochs,
                train_loss,
                cfg.checkpoint.monitor,
                "n/a" if current is None else f"{current:.4f}",
                " *" if improved else "",
            )

            if cfg.checkpoint.save_last:
                _save_checkpoint(last_ckpt, model, cfg, epoch)
            if improved and current is not None:
                best_metric = current
                best_epoch = epoch
                epochs_no_improve = 0
                if cfg.checkpoint.save_best:
                    _save_checkpoint(best_ckpt, model, cfg, epoch)
            else:
                epochs_no_improve += 1

            if cfg.early_stopping.enabled and epochs_no_improve >= cfg.early_stopping.patience:
                logger.info("early stopping at epoch %d (no improvement)", epoch)
                break

        # Guarantee a best checkpoint exists even if the monitored metric was
        # never produced (e.g. monitored class absent from a tiny synthetic val).
        if not best_ckpt.exists():
            _save_checkpoint(best_ckpt, model, cfg, best_epoch or 1)

        _log_sample_predictions(model, val_ds, device, cfg.mlflow.log_samples, best_epoch)
        mlflow.log_artifacts(str(ckpt_dir), artifact_path="checkpoints")
        mlflow.set_tag("best_epoch", str(best_epoch))

    return FitResult(
        best_checkpoint=best_ckpt,
        last_checkpoint=last_ckpt,
        best_metric=float(best_metric) if np.isfinite(best_metric) else float("nan"),
        metrics=last_metrics,
        mlflow_run_id=run_id,
    )


def _resolve_datasets(
    cfg: TrainConfig,
    train_dataset: SegDataset | None,
    val_dataset: SegDataset | None,
) -> tuple[SegDataset, SegDataset]:
    if train_dataset is not None and val_dataset is not None:
        return train_dataset, val_dataset
    if cfg.data.root.exists():
        real = load_real_datasets(
            cfg.data.root,
            image_size=cfg.data.image_size,
            val_fraction=cfg.data.val_split,
            seed=cfg.data.split_seed,
        )
        if real is not None:
            return real
    else:
        logger.warning("data root %s does not exist; using synthetic data", cfg.data.root)
    logger.warning("falling back to synthetic datasets")
    train = SyntheticSegmentationDataset(
        length=max(cfg.data.batch_size * 2, 8),
        image_size=min(cfg.data.image_size, 64),
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        seed=cfg.data.split_seed,
    )
    val = SyntheticSegmentationDataset(
        length=max(cfg.data.batch_size, 4),
        image_size=min(cfg.data.image_size, 64),
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        seed=cfg.data.split_seed + 100,
    )
    return train, val


def _build_loss_fn(cfg: TrainConfig, train_ds: SegDataset, device: torch.device) -> nn.Module:
    class_weights: torch.Tensor | None = None
    if cfg.loss.class_weights == "auto":
        logger.info("computing inverse-frequency class weights from training masks")
        targets = (train_ds[i][1] for i in range(len(train_ds)))  # type: ignore[arg-type]
        class_weights = compute_auto_class_weights(targets, cfg.model.num_classes)
    loss_fn = build_loss(cfg.loss, num_classes=cfg.model.num_classes, class_weights=class_weights)
    return loss_fn.to(device)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    epoch: int,
) -> float:
    model.train()
    running = 0.0
    seen = 0
    pbar = tqdm(loader, desc=f"train e{epoch}", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = images.size(0)
        running += loss.item() * bs
        seen += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running / max(seen, 1)


def _validate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    num_classes: int,
) -> object:
    model.eval()
    metric = SegmentationMetrics(num_classes=num_classes)
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            masks = masks.to(device).long()
            logits = model(images)
            preds = logits_to_labels(logits)
            metric.update(preds.cpu(), masks.cpu())
    return metric.compute()


def _is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "max":
        return current > best + min_delta
    return current < best - min_delta


def make_smoke_config(cfg: TrainConfig) -> TrainConfig:
    """Return a CPU-safe smoke variant: 2 epochs, tiny batches, no AMP."""
    overrides = {
        "optim": {**cfg.optim.model_dump(), "epochs": 2},
        "data": {
            **cfg.data.model_dump(),
            "batch_size": min(cfg.data.batch_size, 2),
            "num_workers": 0,
        },
        "runtime": {**cfg.runtime.model_dump(), "amp": False, "device": "cpu"},
        "early_stopping": {**cfg.early_stopping.model_dump(), "enabled": False},
        "mlflow": {**cfg.mlflow.model_dump(), "log_samples": min(cfg.mlflow.log_samples, 2)},
    }
    return cfg.model_copy(update={k: type(getattr(cfg, k))(**v) for k, v in overrides.items()})


SMOKE_SUBSET_FRACTION: float = 0.1
