# configs

YAML configuration files for training runs and the Sentinel-1 processing pipeline
(model architecture and hyperparameters, dataset paths and splits, augmentation
settings, pipeline stage options). Code reads these configs rather than hardcoding
parameters, so experiments stay reproducible and diffable.

## Training configs

- `unet.yaml` — full U-Net run (ResNet-34/ImageNet, Dice+Focal, cosine schedule,
  early stopping on `val_oil_iou`). Schema: `oilspill.training.config.TrainConfig`.
- `unet_smoke.yaml` — fast CPU sanity config (from-scratch encoder, tiny batches).
  Use with `python scripts/train.py --config configs/unet_smoke.yaml --smoke`.
