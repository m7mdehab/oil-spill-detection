# Metrics: how this project measures segmentation quality

This project reports a deliberately specific set of segmentation metrics, and
computes all of them in exactly one place: [`src/oilspill/metrics.py`](../src/oilspill/metrics.py).
This document explains what is reported, why, and how it differs from the
original 2024 report.

## What is reported

For each of the five classes (Sea Surface, **Oil Spill**, Look-alike, Ship,
Land):

- **IoU** (Jaccard index) — intersection over union of predicted vs. true pixels.
- **Precision**, **Recall**, **F1**.

And as aggregates:

- **mean IoU** and **macro precision / recall / F1** — unweighted means over
  classes, so a rare class counts as much as a common one.
- **pixel accuracy** — reported *only* alongside the per-class numbers, never on
  its own (see below for why it is misleading in isolation).

The two headline numbers the project is selected and compared on are
**oil-class IoU** and **oil-class recall**. Detecting the oil is the task;
overall pixel accuracy is dominated by the sea-surface background and says
almost nothing about that task.

All metrics are derived from a single confusion matrix (truth × prediction)
accumulated with `torchmetrics`. The derivations (TP/FP/FN per class → IoU,
precision, recall, F1) are written out in the module docstring.

## Why the original 2024 numbers were uninformative

The original report published, per model, a table of the form:

| Model | Test Accuracy | Precision | Recall | IoU |
|---|---|---|---|---|
| DeepLabV3+ | 96.25% | 96.25% | 96.25% | 92.77% |
| U-Net | 93.55% | 94.07% | 92.99% | 40.00% |
| FCN | 93.23% | 93.23% | 93.23% | 87.32% |
| SegNet | 92.87% | 92.87% | 92.87% | 86.70% |

Two problems make most of these numbers hard to interpret:

1. **Accuracy = precision = recall to the decimal.** When precision, recall, and
   accuracy come out *identically* (e.g. SegNet's 92.87% across all three), the
   metrics were almost certainly **micro-averaged over all pixels**. Under
   micro-averaging in a single-label, every-pixel-gets-one-class setting, the
   total false positives and total false negatives are equal by construction
   (every misclassified pixel is simultaneously an FP for the class it was
   assigned and an FN for its true class), so micro precision ≡ micro recall ≡
   accuracy. The three columns therefore carry **one** piece of information, not
   three, and that one piece is pixel accuracy.

2. **Pixel accuracy is dominated by the background.** In SAR oil-spill scenes the
   sea-surface class occupies the overwhelming majority of pixels. A model that
   predicted "sea surface" almost everywhere would still score very high pixel
   accuracy while finding little or no oil. A 92–96% accuracy headline is
   consistent with both an excellent detector and a near-useless one; it does
   not distinguish them.

The U-Net row is the tell: 92.99% recall but **40.00% IoU**. Those are not
reconcilable under a single consistent averaging scheme — the recall/precision
figures are micro-averaged (background-dominated) while the IoU is evidently
computed differently (plausibly a macro or oil-class IoU). Mixing averaging
schemes within one table is exactly what makes the comparison unreliable.

None of this means the original work was wrong to build the models; it means the
*reporting* could not support the conclusions drawn from it (e.g. ranking the
four models). This project fixes the reporting, not the intent.

## How the new reporting differs

- **One implementation.** Every number in every document, the API, and the
  README is produced by `oilspill.metrics`. A repository-wide check enforces
  that there is no second metric implementation.
- **Per-class first.** Per-class IoU/precision/recall/F1 are always shown.
  Aggregates are *macro* (class-balanced), so rare-but-important classes such as
  oil are not averaged away by the background.
- **Honest undefined values.** A class absent from a split has undefined
  IoU/precision/recall; it is reported as `null`/`nan` and excluded from the
  macro mean, never silently counted as 0 or 1.
- **Oil-class headline.** Model selection and the top-line comparison use
  oil-class IoU and recall. Pixel accuracy may appear only next to the per-class
  table, for context, never as the summary figure.
- **Traceable.** `scripts/evaluate.py` emits a metrics JSON; every figure quoted
  elsewhere traces back to one of those JSON files.

The original 2024 figures are retained verbatim in
[`docs/legacy_content.md`](legacy_content.md) and are only ever referenced as
"original 2024 results", with the caveats above.
