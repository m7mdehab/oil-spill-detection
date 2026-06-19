# Results

This file is **auto-generated** by `scripts/evaluate.py`. Do not edit the table by
hand: re-running the evaluation harness updates the row for a given run in place.

Every number in the table below is read directly from a metrics JSON committed at
`docs/results/<run>.json` (a copy is also written under `artifacts/eval/<run>/` with
the figures), so each figure is traceable from a fresh clone. "Pixel acc." is the
overall pixel accuracy (all classes); it
is reported for completeness but the project is selected and judged on **Oil IoU**
and **Oil recall**, not pixel accuracy. Runs tagged `smoke` come from short
CPU sanity runs on a tiny subset and are **not** representative of model quality.

To (re)generate a row:

```bash
python scripts/evaluate.py --checkpoint artifacts/checkpoints/<run>/best.pt --tag <tag>
```

<!-- RESULTS_TABLE_BEGIN -->

| Run | Tag | Oil IoU | Oil recall | Mean IoU | Macro F1 | Pixel acc. | Images | Date | Metrics JSON |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run-20260613-101720 | smoke | 0.0014 | 0.0014 | 0.2731 | 0.3470 | 0.8609 | 30 | 2026-06-13 | `docs/results/run-20260613-101720.json` |
| segformer-mit-b2 | segformer-mit-b2 | 0.5662 | 0.7644 | 0.6964 | 0.8017 | 0.9668 | 110 | 2026-06-19 | `docs/results/segformer-mit-b2.json` |
| unet-r34-baseline | unet-baseline | 0.5419 | 0.6853 | 0.6353 | 0.7467 | 0.9533 | 110 | 2026-06-13 | `docs/results/unet-r34-baseline.json` |

<!-- RESULTS_TABLE_END -->

## Per-class metrics

<!-- RESULTS_DETAIL_BEGIN -->

#### run-20260613-101720 (smoke)

| Class | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Sea Surface | 0.8569 | 0.9628 | 0.8862 | 0.9229 |
| Oil Spill | 0.0014 | 0.1163 | 0.0014 | 0.0028 |
| Look-alike | 0.2559 | 0.3540 | 0.4800 | 0.4075 |
| Ship | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Land | 0.2514 | 0.2648 | 0.8324 | 0.4017 |

#### segformer-mit-b2 (segformer-mit-b2)

| Class | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Sea Surface | 0.9657 | 0.9821 | 0.9830 | 0.9825 |
| Oil Spill | 0.5662 | 0.6859 | 0.7644 | 0.7231 |
| Look-alike | 0.5814 | 0.7606 | 0.7116 | 0.7353 |
| Ship | 0.4246 | 0.5393 | 0.6664 | 0.5961 |
| Land | 0.9442 | 0.9529 | 0.9903 | 0.9713 |


#### unet-r34-baseline (unet-baseline)

| Class | IoU | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Sea Surface | 0.9514 | 0.9775 | 0.9727 | 0.9751 |
| Oil Spill | 0.5419 | 0.7214 | 0.6853 | 0.7029 |
| Look-alike | 0.4623 | 0.6168 | 0.6486 | 0.6323 |
| Ship | 0.3046 | 0.3588 | 0.6686 | 0.4670 |
| Land | 0.9166 | 0.9255 | 0.9896 | 0.9565 |

<!-- RESULTS_DETAIL_END -->
