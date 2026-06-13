"""Write the metrics JSON and maintain the results table in ``docs/results.md``.

The JSON produced by :func:`write_results_json` is the *single source* of every
number that appears in the docs/README: :func:`update_results_markdown` reads the
same :class:`~oilspill.metrics.MetricResult` and emits a table row, and records the
path of the JSON so any figure in the docs is traceable back to a committed file.

The results table lives between stable HTML-comment markers so re-running the
harness for a given run name *updates* that run's row in place instead of
appending a duplicate.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oilspill.metrics import MetricResult

TABLE_BEGIN = "<!-- RESULTS_TABLE_BEGIN -->"
TABLE_END = "<!-- RESULTS_TABLE_END -->"

_TABLE_HEADER = (
    "| Run | Tag | Oil IoU | Oil recall | Mean IoU | Macro F1 | "
    "Pixel acc. | Images | Date | Metrics JSON |"
)
_TABLE_DIVIDER = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

_DOC_HEADER = """# Results

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
"""


def write_results_json(
    result: MetricResult,
    out_path: Path | str,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write the full metrics (plus run metadata) as JSON; return the path.

    The metric payload is ``MetricResult.to_dict()`` verbatim under a ``"metrics"``
    key, with caller-supplied ``meta`` (run name, checkpoint path, tag, image
    count, ...) alongside it.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "meta": meta or {},
        "metrics": result.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def _format_row(
    run_name: str,
    tag: str,
    result: MetricResult,
    num_images: int,
    json_path: str,
    date: str,
) -> str:
    return (
        f"| {run_name} | {tag} | {result.oil_iou:.4f} | {result.oil_recall:.4f} | "
        f"{result.mean_iou:.4f} | {result.macro_f1:.4f} | {result.pixel_accuracy:.4f} | "
        f"{num_images} | {date} | `{json_path}` |"
    )


def _empty_table() -> list[str]:
    return [TABLE_BEGIN, "", _TABLE_HEADER, _TABLE_DIVIDER, TABLE_END]


def _parse_rows(lines: list[str]) -> dict[str, str]:
    """Map run name -> existing row, for rows between the table markers."""
    rows: dict[str, str] = {}
    inside = False
    for line in lines:
        if line.strip() == TABLE_BEGIN:
            inside = True
            continue
        if line.strip() == TABLE_END:
            inside = False
            continue
        if not inside:
            continue
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if stripped in (_TABLE_HEADER, _TABLE_DIVIDER):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells:
            rows[cells[0]] = stripped
    return rows


def update_results_markdown(
    results_md_path: Path | str,
    run_name: str,
    result: MetricResult,
    json_path: Path | str,
    *,
    tag: str = "",
    num_images: int = 0,
    date: str | None = None,
) -> Path:
    """Insert or update the row for ``run_name`` in the results table.

    Creates ``results_md_path`` (with the auto-generated header and an empty table)
    if it does not exist. The marker-delimited table is rewritten so that the row
    for ``run_name`` is added or replaced and rows are sorted by run name. Every
    cell is derived from ``result`` (a :class:`MetricResult`), and ``json_path`` is
    recorded so the numbers trace back to that JSON.
    """
    results_md_path = Path(results_md_path)
    date = date or datetime.now(UTC).strftime("%Y-%m-%d")
    json_str = str(json_path).replace("\\", "/")

    if results_md_path.exists():
        text = results_md_path.read_text(encoding="utf-8")
    else:
        results_md_path.parent.mkdir(parents=True, exist_ok=True)
        text = _DOC_HEADER + "\n" + "\n".join(_empty_table()) + "\n"

    if TABLE_BEGIN not in text or TABLE_END not in text:
        # Append a fresh table if the markers are missing.
        text = text.rstrip() + "\n\n" + "\n".join(_empty_table()) + "\n"

    lines = text.splitlines()
    rows = _parse_rows(lines)
    rows[run_name] = _format_row(run_name, tag, result, num_images, json_str, date)

    sorted_rows = [rows[name] for name in sorted(rows)]
    new_table = [TABLE_BEGIN, "", _TABLE_HEADER, _TABLE_DIVIDER, *sorted_rows, "", TABLE_END]

    # Replace the existing marker block with the rebuilt table.
    out_lines: list[str] = []
    inside = False
    replaced = False
    for line in lines:
        if line.strip() == TABLE_BEGIN:
            inside = True
            out_lines.extend(new_table)
            replaced = True
            continue
        if line.strip() == TABLE_END:
            inside = False
            continue
        if inside:
            continue
        out_lines.append(line)
    if not replaced:
        out_lines.extend(["", *new_table])

    results_md_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    return results_md_path


__all__ = [
    "TABLE_BEGIN",
    "TABLE_END",
    "update_results_markdown",
    "write_results_json",
]
