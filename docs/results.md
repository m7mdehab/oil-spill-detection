# Results

This file is **auto-generated** by `scripts/evaluate.py`. Do not edit the table by
hand: re-running the evaluation harness updates the row for a given run in place.

Every number in the table below is read directly from a metrics JSON written under
`artifacts/eval/<run>/metrics.json` by the same run, so each figure is traceable to
a committed artifact. "Pixel acc." is the overall pixel accuracy (all classes); it
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
| run-20260613-101720 | smoke | 0.0014 | 0.0014 | 0.2731 | 0.3470 | 0.8609 | 30 | 2026-06-13 | `artifacts/eval/run-20260613-101720/metrics.json` |

<!-- RESULTS_TABLE_END -->
