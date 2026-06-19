# Loss ablation: imbalance-aware loss vs. headline oil metrics

This experiment isolates the effect of the **segmentation loss** on the rare
oil-spill class. The dataset is heavily imbalanced — the `Oil Spill` class is
~0.99% of pixels, a ~89:1 majority-to-oil ratio (see
[`data_report.md`](data_report.md)). A loss minimised by predicting mostly
background scores well on pixel accuracy while under-segmenting oil, so the
headline metrics here are **oil-class IoU** and **oil-class recall**, not pixel
accuracy. Pixel accuracy appears in the table only as a secondary column,
alongside the per-class numbers. See [`metrics.md`](metrics.md) for the exact
metric definitions and why selection is on oil IoU/recall.

## The matrix

`{architecture} × {loss}` with everything else held fixed:

- **Architectures:** `Unet`, `DeepLabV3Plus`, `segformer`, `foundation`. Each
  cell is derived from that architecture's base config (`configs/<arch>.yaml`);
  architectures without a base config are skipped by the runner.
- **Loss variants:**
  - `ce_weighted` — class-weighted cross-entropy
    (`type: ce`, `class_weights: auto`, inverse-frequency weights from the train
    masks).
  - `dice_focal` — Dice + Focal combo
    (`type: dice_focal`, `class_weights: auto`, `focal_gamma: 2.0`).
- **Fixed for fairness:** a single seed (42) and a single epoch budget across
  every cell, so the only thing that varies within an architecture is the loss.

The full matrix is 4 architectures × 2 losses = 8 cells (fewer if an
architecture has no base config). Generated configs live in
[`../configs/ablations/`](../configs/ablations/), one YAML per cell named
`<arch>__<loss>.yaml`, each a complete and validated `TrainConfig`.

## Running it

The runner ([`../scripts/run_ablations.py`](../scripts/run_ablations.py)) only
orchestrates the existing training/evaluation entry points; it never trains a
model itself.

```sh
# 1. Generate one config per cell (fixed seed/budget).
uv run python scripts/run_ablations.py generate --epochs 40

# 2. Emit the GPU training + evaluation commands for the operator to run.
uv run python scripts/run_ablations.py commands --gpu L4

# 3. After the runs land (checkpoints under artifacts/checkpoints/<run>/best.pt
#    and committed metrics under docs/results/<run>.json), build the table.
uv run python scripts/run_ablations.py aggregate
```

Each cell trains on Modal
(`modal run --detach scripts/modal_train.py::main --config <cfg> --gpu L4`) and
is then evaluated on the **test** split
(`scripts/evaluate.py --checkpoint <best.pt> --split test`), which writes the
committed `docs/results/<run>.json` that `aggregate` reads.

## Budget

A single Modal L4 run is roughly **$0.30–0.60**, so the full 8-cell matrix is
about **$2.40–$4.80**. To stay within free credits, reduce the matrix:

- `--max-cells N` caps the number of cells run.
- `--arch Unet --arch segformer` restricts to specific architectures.
- `--epochs` lowers the per-run budget.

`commands` prints a cost estimate scaled to the cells it emits.

## Results

Headline ordering is by **oil IoU** (descending); the winning cell is marked.
`pixel acc*` is secondary and shown only alongside the per-class numbers. This
table is populated by `run_ablations.py aggregate` after the runs complete.

<!-- ABLATIONS_TABLE_BEGIN -->

| arch | loss | oil IoU | oil recall | mean IoU | macro F1 | pixel acc* |
| --- | --- | --- | --- | --- | --- | --- |
| _(populated by `run_ablations.py aggregate` after runs)_ |  |  |  |  |  |  |

<!-- ABLATIONS_TABLE_END -->
