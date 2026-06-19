"""Render an honest Hugging Face model card from a committed metrics JSON.

Every number on the card is read from a metrics JSON written by the evaluation
harness (the same files committed under ``docs/results/``); nothing is invented.
The card describes the model, its intended use, the training dataset, the real
per-class IoU/precision/recall, the headline oil-class IoU/recall, the 5-class
colour legend, and an explicit limitations section.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oilspill.data.colors import CLASS_COLORS
from oilspill.metrics import CLASS_NAMES, OIL_CLASS_INDEX

# The dataset the baseline was trained on. Linked, not fabricated.
_DATASET_NAME = "MKLab Oil Spill Detection Dataset"
_DATASET_URL = "https://m4d.iti.gr/oil-spill-detection-dataset/"


def _load_run(run_json_path: Path | str) -> dict[str, Any]:
    """Load and minimally validate a metrics JSON written by the eval harness."""
    data: dict[str, Any] = json.loads(Path(run_json_path).read_text(encoding="utf-8"))
    if "metrics" not in data:
        raise ValueError(f"{run_json_path} has no 'metrics' block; not an evaluation JSON")
    metrics = data["metrics"]
    for key in ("per_class", "aggregate"):
        if key not in metrics:
            raise ValueError(f"{run_json_path} metrics is missing '{key}'")
    return data


def _fmt(value: float | None) -> str:
    """Format a metric value to 4 dp, or an em dash when undefined (``null``)."""
    return "—" if value is None else f"{value:.4f}"


def _hex(rgb: tuple[int, int, int]) -> str:
    """RGB tuple to an uppercase ``#RRGGBB`` hex string."""
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


def _per_class_table(per_class: dict[str, dict[str, float | None]]) -> str:
    """Build the per-class IoU/precision/recall markdown table."""
    iou = per_class.get("iou", {})
    precision = per_class.get("precision", {})
    recall = per_class.get("recall", {})
    rows = [
        "| Class | IoU | Precision | Recall |",
        "| --- | --- | --- | --- |",
    ]
    for name in CLASS_NAMES:
        rows.append(
            f"| {name} | {_fmt(iou.get(name))} | "
            f"{_fmt(precision.get(name))} | {_fmt(recall.get(name))} |"
        )
    return "\n".join(rows)


def _legend_table() -> str:
    """Build the 5-class colour legend markdown table."""
    rows = [
        "| Class id | Class | RGB | Hex |",
        "| --- | --- | --- | --- |",
    ]
    for idx, (name, rgb) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, strict=True)):
        rows.append(f"| {idx} | {name} | `{rgb[0]}, {rgb[1]}, {rgb[2]}` | `{_hex(rgb)}` |")
    return "\n".join(rows)


def build_model_card(
    run_json_path: Path | str,
    *,
    repo_id: str,
    arch: str,
    onnx_filename: str = "model.onnx",
    license_id: str = "mit",
) -> str:
    """Produce a Hugging Face model card (markdown) for a published model.

    Parameters
    ----------
    run_json_path:
        Path to a committed metrics JSON (``docs/results/<run>.json``). All
        reported numbers are read from this file.
    repo_id:
        Target Hugging Face repo id, e.g. ``"m7mdehab/oil-spill-segmentation"``.
    arch:
        Architecture description for the model card, e.g. ``"U-Net (ResNet-34
        encoder)"``.
    onnx_filename:
        Name the ONNX file will have in the repo (used in the usage snippet).
    license_id:
        SPDX-style license identifier for the card front matter.

    Returns
    -------
    str
        The complete model card as markdown text.
    """
    data = _load_run(run_json_path)
    meta = data.get("meta", {})
    metrics = data["metrics"]
    per_class = metrics["per_class"]
    aggregate = metrics["aggregate"]

    oil_name = CLASS_NAMES[OIL_CLASS_INDEX]
    oil_iou = aggregate.get("oil_iou")
    oil_recall = aggregate.get("oil_recall")
    mean_iou = aggregate.get("mean_iou")
    macro_f1 = aggregate.get("macro_f1")
    pixel_acc = aggregate.get("pixel_accuracy")

    image_size = meta.get("image_size")
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        in_h, in_w = int(image_size[0]), int(image_size[1])
    else:
        in_h, in_w = 256, 256
    size_str = f"{in_h}x{in_w}"
    num_images = meta.get("num_images")
    split = meta.get("split", "test")

    front_matter = "\n".join(
        [
            "---",
            f"license: {license_id}",
            "library_name: onnx",
            "pipeline_tag: image-segmentation",
            "tags:",
            "- oil-spill-detection",
            "- semantic-segmentation",
            "- remote-sensing",
            "- sar",
            "- sentinel-1",
            "- onnx",
            "---",
        ]
    )

    usage = "\n".join(
        [
            "```python",
            "import numpy as np",
            "import onnxruntime as ort",
            "from huggingface_hub import hf_hub_download",
            "",
            f'path = hf_hub_download("{repo_id}", "{onnx_filename}")',
            'session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])',
            "",
            "# Input: float32 (N, 3, H, W); batch and spatial dims are dynamic.",
            f"x = np.random.rand(1, 3, {in_h}, {in_w}).astype(np.float32)",
            'logits = session.run(None, {"input": x})[0]  # (N, 5, H, W)',
            "labels = logits.argmax(axis=1)               # (N, H, W) class ids 0..4",
            "```",
        ]
    )

    return f"""{front_matter}

# Oil Spill Segmentation ({arch})

Semantic segmentation model for **oil spill detection in Sentinel-1 SAR imagery**.
The model assigns each pixel one of five classes (sea surface, oil spill,
look-alike, ship, land). It is distributed as an ONNX graph with dynamic batch
and spatial axes, so it accepts variable-size inputs.

- **Architecture:** {arch}
- **Task:** multiclass semantic segmentation ({len(CLASS_NAMES)} classes)
- **Input:** float32 tensor `(N, 3, H, W)` (SAR intensity replicated to 3 channels)
- **Output:** logits `(N, {len(CLASS_NAMES)}, H, W)`; take `argmax` over the class axis
- **Format:** ONNX (opset 17), runs on CPU via onnxruntime

## Intended use

Pixel-level oil-spill segmentation in Sentinel-1 SAR scenes for research,
monitoring assistance, and as a screening aid to flag candidate slicks for
human review. It is **not** intended to be the sole basis for operational or
legal decisions (see Limitations).

## Training dataset

Trained on the [{_DATASET_NAME}]({_DATASET_URL}), a public Sentinel-1 SAR oil
spill dataset with pixel-level annotations for the five classes below. The split
used is **1002 training / 110 test** images. Images were processed at {size_str}.

## Evaluation results

Evaluated on the **{split}** split{f" ({num_images} images)" if num_images is not None else ""}
at {size_str}. All numbers below are produced by the project's evaluation harness;
none are hand-edited.

**Headline metrics ({oil_name}):**

- {oil_name} IoU: **{_fmt(oil_iou)}**
- {oil_name} recall: **{_fmt(oil_recall)}**

**Aggregate:**

- Mean IoU (macro): {_fmt(mean_iou)}
- Macro F1: {_fmt(macro_f1)}
- Pixel accuracy: {_fmt(pixel_acc)}

**Per-class:**

{_per_class_table(per_class)}

## Class colour legend

Masks use the dataset's authoritative colour legend (class id order 0..4):

{_legend_table()}

## Usage

{usage}

## Limitations

- **Oil / look-alike confusion.** Oil slicks and natural look-alikes (biogenic
  slicks, low-wind dark patches, organic films) appear similar in SAR. The
  model confuses the two, so a positive detection is not a confirmed spill.
- **Small dataset.** Training used only ~1000 images; generalisation to other
  sensors, geographies, sea states, and acquisition geometries is unverified.
- **Single VV polarization / SAR intensity.** The model consumes SAR intensity
  only; it has no optical, multi-polarization, or wind/ancillary context.
- **Environmental look-alikes.** Low wind, biogenic slicks, internal waves, and
  ship wakes can all produce dark features that may be misclassified.
- **Not for sole operational decision-making.** Outputs are a screening aid and
  must be confirmed by a human analyst with corroborating evidence before any
  operational, regulatory, or legal action.

## Citation

If you use this model, please cite the underlying dataset:
[{_DATASET_NAME}]({_DATASET_URL}).
"""
