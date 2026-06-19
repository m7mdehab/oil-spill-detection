"""Earth-observation foundation-model backbone with a segmentation decode head.

This module fine-tunes a self-supervised foundation backbone for SAR oil-spill
segmentation. The model wraps a Vision-Transformer foundation encoder (DINOv2 by
default) and attaches a lightweight convolutional decode head that turns the
patch-token feature map into full-resolution per-pixel logits.

Foundation-model selection (the decision trail)
-----------------------------------------------
The brief was to fine-tune *one* EO foundation model, trying candidates in order
and stopping at the first that works **dep-light** (no heavy unavailable package),
is **CPU-smoke-testable**, and is reachable via Hugging Face / ``timm``:

1. **Prithvi-EO-2.0** (``ibm-nasa-geospatial/Prithvi-EO-2.0-300M``) -- a ViT-MAE
   EO backbone. *Blocked*: it is not exposed as a ``transformers`` model class
   (no ``Prithvi*`` in ``transformers``) and is not in ``timm``. The supported
   loader is IBM's ``terratorch`` (plus ``prithvi_mae``), a heavy dependency that
   is **not installed and may not be added** here. Adapting its 6-band HLS
   patch-embed to 3 channels is feasible in principle, but the loader dependency
   is the hard blocker, so we fall through.
2. **Clay** -- *Blocked*: no ``clay`` package installed; the official loader is a
   bespoke Lightning/checkpoint stack, not dep-light via HF/timm. Fall through.
3. **DOFA** (wavelength-aware) -- *Blocked*: no ``dofa`` / ``torchgeo`` package
   installed; reachable dep-light only through ``torchgeo``, which is not present.
   Fall through.
4. **SAM2** mask-decoder fine-tune -- *Blocked*: no ``sam2`` package installed.
   Fall through.
5. **DINOv2 + conv decode head** -- *Landed*. ``facebook/dinov2-small`` loads via
   the already-installed ``transformers`` (:class:`~transformers.Dinov2Model`),
   constructs **offline** from a config (no download) for evaluation, and runs on
   CPU. This is the dep-light, guaranteed option and is what ships here.

Integration details
--------------------
* **Channels.** DINOv2's pretrained patch-embed stem is 3-channel; the project
  feeds SAR replicated to 3 channels, so we keep ``in_channels=3``.
* **Normalisation.** Inputs are already ImageNet-normalised by
  ``oilspill.data.build_transforms``; DINOv2 uses ImageNet mean/std too, so we add
  **no** second normalisation.
* **Input size.** DINOv2 uses a patch size of 14, so the input spatial size must
  be a multiple of 14 (e.g. 224 or 448). The decode head upsamples the
  ``H/14 x W/14`` token grid back to the input resolution.
* **Output resolution.** :meth:`forward` always returns ``(N, num_classes, H, W)``
  logits at the **input** resolution, matching the trainer/metrics contract.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Dinov2Config, Dinov2Model

from oilspill.models.registry import register_model

if TYPE_CHECKING:
    from oilspill.training.config import ModelConfig

# Default Hugging Face backbone when ``model_cfg.params["backbone"]`` is unset.
_DEFAULT_BACKBONE = "facebook/dinov2-small"

# Per-variant DINOv2 ViT hyper-parameters for the published ``facebook/dinov2-*``
# checkpoints, used to construct a :class:`~transformers.Dinov2Config` *offline*
# (no network round-trip) when ``pretrained`` is False (e.g. evaluation, where the
# trained checkpoint weights are loaded afterwards). Source: HF DINOv2 configs.
_DINOV2_VARIANTS: dict[str, dict[str, int]] = {
    "dinov2-small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
    "dinov2-base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    "dinov2-large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
    "dinov2-giant": {"hidden_size": 1536, "num_hidden_layers": 40, "num_attention_heads": 24},
}

# DINOv2 patch size (all published variants use 14).
_PATCH_SIZE = 14


def _offline_config(backbone: str) -> Dinov2Config:
    """Build a ``Dinov2Config`` without network access for a known DINOv2 variant.

    Falls back to :meth:`Dinov2Config.from_pretrained` (which may fetch the config
    JSON) for unrecognised checkpoint ids.
    """
    variant = backbone.rsplit("/", maxsplit=1)[-1]
    spec = _DINOV2_VARIANTS.get(variant)
    if spec is None:
        config = Dinov2Config.from_pretrained(backbone)
        assert isinstance(config, Dinov2Config)
        return config
    return Dinov2Config(
        hidden_size=spec["hidden_size"],
        num_hidden_layers=spec["num_hidden_layers"],
        num_attention_heads=spec["num_attention_heads"],
        patch_size=_PATCH_SIZE,
    )


class _ConvDecodeHead(nn.Module):
    """Lightweight conv decode head: reshape patch tokens then progressively upsample.

    The DINOv2 backbone emits a sequence of patch tokens for an ``H/14 x W/14``
    grid. We drop the CLS token, reshape the patch tokens to a spatial feature map
    ``(N, C, H/14, W/14)``, then apply two conv blocks with 2x bilinear upsampling
    each, and a final 1x1 classifier. The output is bilinearly resized to the exact
    input resolution in :meth:`Foundation.forward`.
    """

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(hidden // 2, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (N, C, h, w) at the patch-grid resolution.
        x = self.block1(feat)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.block2(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.classifier(x)


class Foundation(nn.Module):
    """DINOv2 foundation backbone + conv decode head emitting input-res logits.

    The wrapped :class:`~transformers.Dinov2Model` produces patch tokens for an
    ``H/14 x W/14`` grid; :meth:`forward` reshapes them to a feature map, runs the
    decode head, and upsamples the result back to the input ``(H, W)`` so the
    logits line up with full-resolution masks.

    Parameters
    ----------
    backbone:
        A constructed :class:`~transformers.Dinov2Model`.
    num_classes:
        Number of segmentation classes (output channels).
    freeze_backbone:
        When True, the backbone parameters are frozen and only the decode head is
        trained (linear-probe style); when False the whole model is fine-tuned.
    head_hidden:
        Width of the first decode-head conv block.
    """

    def __init__(
        self,
        backbone: Dinov2Model,
        num_classes: int,
        *,
        freeze_backbone: bool = False,
        head_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        # ``patch_size`` is typed loosely on the HF config; DINOv2 uses a single
        # square patch, so coerce to a plain int for the grid-size arithmetic.
        patch_size = backbone.config.patch_size
        self.patch_size = int(
            patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size
        )
        self.head = _ConvDecodeHead(backbone.config.hidden_size, num_classes, hidden=head_hidden)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _tokens_to_feature_map(
        self, tokens: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        # tokens: (N, 1 + grid_h*grid_w, C). Drop the leading CLS token, then
        # reshape the patch tokens to a (N, C, grid_h, grid_w) spatial map.
        patch_tokens = tokens[:, 1:, :]
        n, num_patches, channels = patch_tokens.shape
        expected = grid_h * grid_w
        if num_patches != expected:
            # Some DINOv2 builds append register tokens; keep the last grid tokens.
            patch_tokens = patch_tokens[:, num_patches - expected :, :]
        feat = patch_tokens.transpose(1, 2).reshape(n, channels, grid_h, grid_w)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_channels, H, W). Inputs are already ImageNet-normalised by the
        # data pipeline, so we feed them straight to the backbone (no re-norm).
        input_size = x.shape[-2:]
        height, width = int(input_size[0]), int(input_size[1])
        grid_h = math.ceil(height / self.patch_size)
        grid_w = math.ceil(width / self.patch_size)

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=x)
        else:
            outputs = self.backbone(pixel_values=x)
        tokens = outputs.last_hidden_state

        feat = self._tokens_to_feature_map(tokens, grid_h, grid_w)
        logits = self.head(feat)
        # Resize to the exact input resolution (the head's two 2x upsamples land
        # near but not exactly on H, W for arbitrary sizes).
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


@register_model("foundation")
def build_foundation(model_cfg: ModelConfig, pretrained: bool) -> nn.Module:
    """Build a :class:`Foundation` (DINOv2 + conv head) from ``model_cfg``.

    When ``pretrained`` is ``True`` (training) the DINOv2 base weights are
    downloaded and loaded into the backbone. When ``False`` (evaluation) the
    backbone is constructed from a config **without any network access**, so a
    trained checkpoint can be loaded over it offline.

    Recognised ``model_cfg.params`` keys:

    * ``backbone`` -- HF DINOv2 id (default ``facebook/dinov2-small``).
    * ``freeze_backbone`` -- bool, train only the decode head (default False).
    * ``head_hidden`` -- int, width of the first decode-head conv block (default 256).
    """
    backbone_id = str(model_cfg.params.get("backbone", _DEFAULT_BACKBONE))
    num_classes = model_cfg.num_classes
    freeze_backbone = bool(model_cfg.params.get("freeze_backbone", False))
    head_hidden = int(model_cfg.params.get("head_hidden", 256))

    if model_cfg.in_channels != 3:
        # DINOv2's pretrained stem is 3-channel; the project keeps SAR as 3
        # replicated channels. Reject other widths rather than silently misbehave.
        raise ValueError(
            f"Foundation (DINOv2) expects in_channels=3 (SAR replicated to RGB), "
            f"got {model_cfg.in_channels}"
        )

    if pretrained:
        backbone = Dinov2Model.from_pretrained(backbone_id)
        assert isinstance(backbone, Dinov2Model)
    else:
        # Offline construction from a built-in variant table (no weight download);
        # the trained checkpoint is loaded over these weights afterwards.
        config = _offline_config(backbone_id)
        backbone = Dinov2Model(config)

    return Foundation(
        backbone,
        num_classes,
        freeze_backbone=freeze_backbone,
        head_hidden=head_hidden,
    )
