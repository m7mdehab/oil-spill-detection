"""Second CNN architecture: DeepLabV3+ (atrous spatial pyramid pooling).

This module documents the project's second-architecture choice; no custom code is
required because ``segmentation_models_pytorch`` provides ``DeepLabV3Plus``
directly, so it is built by the registry's smp fallback from a config with
``arch: DeepLabV3Plus`` (see ``configs/deeplabv3plus.yaml``).

Why DeepLabV3+ and not Mask2Former
----------------------------------
DeepLabV3+ is a strong, distinct CNN family (ResNet backbone + ASPP +
encoder-decoder) that contrasts well with the U-Net baseline and the
transformer-based SegFormer, giving the comparison three genuinely different
inductive biases.

Mask2Former was considered (it is the "harder, preferred" option) but rejected
for this project: it is a *mask-classification* model trained with a
set-prediction objective (Hungarian matching between predicted and ground-truth
masks, plus per-mask classification and mask/dice losses). That objective is
fundamentally incompatible with this repo's shared per-pixel training path, whose
losses take ``(logits[N,C,H,W], target[N,H,W])`` and whose metrics are per-pixel.
Wrapping Mask2Former to emit per-pixel logits would discard the very mechanism
that makes it strong, while a faithful integration would require a parallel
trainer with its own loss plumbing — exactly the "loss plumbing fights the
trainer" condition under which the plan prescribes the DeepLabV3+ fallback.
"""

from __future__ import annotations

# The architecture name resolved by the registry's smp fallback.
ARCH_NAME = "DeepLabV3Plus"
