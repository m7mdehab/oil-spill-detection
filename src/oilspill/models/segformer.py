"""Hugging Face SegFormer architecture for SAR oil-spill segmentation.

SegFormer (Xie et al., 2021) pairs a hierarchical Mix-Transformer (MiT) encoder
with a lightweight all-MLP decode head. We wrap
:class:`transformers.SegformerForSemanticSegmentation` so the project's trainer
and evaluation harness can use it through the shared model registry with no other
changes.

Key integration details (see comments in :class:`SegFormer` below):

* **Output resolution.** HF SegFormer emits logits at ``H/4 x W/4``. The trainer
  and metrics compare logits against full-resolution masks, so :meth:`forward`
  bilinearly upsamples the logits back to the input ``(H, W)``.
* **Normalisation.** SegFormer expects ImageNet-normalised 3-channel input. The
  data pipeline (``oilspill.data.build_transforms``) already applies ImageNet
  mean/std to the 3-channel SAR image, so inputs are consistent and we add **no**
  second normalisation here.
* **Channels.** SegFormer's pretrained patch-embedding stem is 3-channel. We keep
  3 channels (SAR replicated to RGB), consistent with the rest of the project.
* **Input size.** SegFormer downsamples by a factor of 32, so the input spatial
  size should be a multiple of 32; ``image_size`` 256 and 512 both satisfy this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from oilspill.models.registry import register_model

if TYPE_CHECKING:
    from oilspill.training.config import ModelConfig

# Default Hugging Face checkpoint id when ``model_cfg.params["hf_model"]`` is unset.
_DEFAULT_HF_MODEL = "nvidia/mit-b2"

# Per-variant MiT encoder hyper-parameters (depths and embedding widths) for the
# published ``nvidia/mit-b*`` checkpoints. Used to construct a ``SegformerConfig``
# *offline* (no network round-trip) when ``pretrained`` is False, e.g. during
# evaluation where the trained checkpoint weights are loaded afterwards. All other
# fields use the HF defaults. Source: the SegFormer paper / HF model configs.
_MIT_VARIANTS: dict[str, dict[str, list[int]]] = {
    "mit-b0": {"depths": [2, 2, 2, 2], "hidden_sizes": [32, 64, 160, 256]},
    "mit-b1": {"depths": [2, 2, 2, 2], "hidden_sizes": [64, 128, 320, 512]},
    "mit-b2": {"depths": [3, 4, 6, 3], "hidden_sizes": [64, 128, 320, 512]},
    "mit-b3": {"depths": [3, 4, 18, 3], "hidden_sizes": [64, 128, 320, 512]},
    "mit-b4": {"depths": [3, 8, 27, 3], "hidden_sizes": [64, 128, 320, 512]},
    "mit-b5": {"depths": [3, 6, 40, 3], "hidden_sizes": [64, 128, 320, 512]},
}
# The decode-head width differs between b0 (256) and the larger variants (768).
_DECODE_HEAD_DIM: dict[str, int] = {"mit-b0": 256}


def _offline_config(hf_model: str, num_classes: int) -> SegformerConfig:
    """Build a ``SegformerConfig`` without network access for a known MiT variant.

    Falls back to :meth:`SegformerConfig.from_pretrained` (which may download the
    config JSON) for unrecognised checkpoint ids.
    """
    variant = hf_model.rsplit("/", maxsplit=1)[-1]
    spec = _MIT_VARIANTS.get(variant)
    if spec is None:
        config = SegformerConfig.from_pretrained(hf_model)
        assert isinstance(config, SegformerConfig)
    else:
        config = SegformerConfig(
            depths=spec["depths"],
            hidden_sizes=spec["hidden_sizes"],
            decoder_hidden_size=_DECODE_HEAD_DIM.get(variant, 768),
        )
    # ``num_labels`` is a PretrainedConfig property (kept out of the typed __init__
    # signature); set it explicitly so the decode-head classifier matches our task.
    config.num_labels = num_classes
    return config


class SegFormer(nn.Module):
    """SegFormer wrapper emitting input-resolution ``(N, num_classes, H, W)`` logits.

    The wrapped :class:`~transformers.SegformerForSemanticSegmentation` produces
    logits at a quarter of the input resolution; :meth:`forward` upsamples them
    back to the original spatial size so they line up with full-resolution masks.
    """

    def __init__(self, backbone: SegformerForSemanticSegmentation) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_channels, H, W). Inputs are already ImageNet-normalised by the
        # data pipeline, so we feed them straight to the backbone (no re-norm).
        input_size = x.shape[-2:]
        # HF returns logits at H/4 x W/4 in ``outputs.logits``.
        logits = self.backbone(pixel_values=x).logits
        # Upsample back to the input resolution; the trainer/metrics compare these
        # logits against full-resolution masks.
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


@register_model("segformer")
def build_segformer(model_cfg: ModelConfig, pretrained: bool) -> nn.Module:
    """Build a :class:`SegFormer` from ``model_cfg``.

    When ``pretrained`` is ``True`` (training) the HF base weights are downloaded
    and loaded, with the classification head resized to ``num_classes``. When
    ``False`` (evaluation) the model is constructed from a config **without any
    network access**, so the trained checkpoint can be loaded over it offline.
    """
    hf_model = str(model_cfg.params.get("hf_model", _DEFAULT_HF_MODEL))
    num_classes = model_cfg.num_classes

    if model_cfg.in_channels != 3:
        # SegFormer's pretrained stem is 3-channel; the project keeps SAR as 3
        # replicated channels. Reject other widths rather than silently misbehave.
        raise ValueError(
            f"SegFormer expects in_channels=3 (SAR replicated to RGB), got {model_cfg.in_channels}"
        )

    if pretrained:
        backbone = SegformerForSemanticSegmentation.from_pretrained(
            hf_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        # Offline construction: derive the architecture from a built-in variant
        # table (no weight download), then instantiate fresh randomly-initialised
        # modules. The trained checkpoint is loaded over these weights afterwards.
        config = _offline_config(hf_model, num_classes)
        backbone = SegformerForSemanticSegmentation(config)

    return SegFormer(backbone)
