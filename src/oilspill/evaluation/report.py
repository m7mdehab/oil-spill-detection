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
DETAIL_BEGIN = "<!-- RESULTS_DETAIL_BEGIN -->"
DETAIL_END = "<!-- RESULTS_DETAIL_END -->"

_PER_CLASS_HEADER = "| Class | IoU | Precision | Recall | F1 |"
_PER_CLASS_DIVIDER = "| --- | --- | --- | --- | --- |"

_TABLE_HEADER = (
    "| Run | Tag | Oil IoU | Oil recall | Mean IoU | Macro F1 | "
    "Pixel acc. | Images | Date | Metrics JSON |"
)
_TABLE_DIVIDER = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"

_DOC_HEADER = """# Results

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
    return [
        TABLE_BEGIN,
        "",
        _TABLE_HEADER,
        _TABLE_DIVIDER,
        TABLE_END,
        "",
        "## Per-class metrics",
        "",
        DETAIL_BEGIN,
        DETAIL_END,
    ]


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


def _format_per_class_block(run_name: str, tag: str, result: MetricResult) -> list[str]:
    """A per-class IoU/precision/recall/F1 table for one run (nan rendered '-')."""

    def fmt(value: float) -> str:
        return "-" if value != value else f"{value:.4f}"  # NaN != NaN

    heading = f"#### {run_name}" + (f" ({tag})" if tag else "")
    lines = [heading, "", _PER_CLASS_HEADER, _PER_CLASS_DIVIDER]
    for i, name in enumerate(result.class_names):
        lines.append(
            f"| {name} | {fmt(float(result.iou[i]))} | {fmt(float(result.precision[i]))} | "
            f"{fmt(float(result.recall[i]))} | {fmt(float(result.f1[i]))} |"
        )
    lines.append("")
    return lines


def _parse_detail_blocks(lines: list[str]) -> dict[str, list[str]]:
    """Map run name -> its per-class block lines, between the detail markers."""
    blocks: dict[str, list[str]] = {}
    inside = False
    current: str | None = None
    for line in lines:
        if line.strip() == DETAIL_BEGIN:
            inside = True
            continue
        if line.strip() == DETAIL_END:
            inside = False
            current = None
            continue
        if not inside:
            continue
        if line.startswith("#### "):
            current = line[len("#### ") :].split(" (")[0].strip()
            blocks[current] = [line]
        elif current is not None:
            blocks[current].append(line)
    # Trim trailing blank lines in each block.
    for name, block in blocks.items():
        while block and not block[-1].strip():
            block.pop()
        blocks[name] = block
    return blocks


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
    if DETAIL_BEGIN not in text or DETAIL_END not in text:
        detail_scaffold = f"## Per-class metrics\n\n{DETAIL_BEGIN}\n{DETAIL_END}\n"
        text = text.rstrip() + "\n\n" + detail_scaffold

    lines = text.splitlines()

    rows = _parse_rows(lines)
    rows[run_name] = _format_row(run_name, tag, result, num_images, json_str, date)
    sorted_rows = [rows[name] for name in sorted(rows)]
    new_table = [TABLE_BEGIN, "", _TABLE_HEADER, _TABLE_DIVIDER, *sorted_rows, "", TABLE_END]

    blocks = _parse_detail_blocks(lines)
    blocks[run_name] = _format_per_class_block(run_name, tag, result)
    detail_body: list[str] = []
    for name in sorted(blocks):
        detail_body.extend(blocks[name])
        detail_body.append("")
    new_detail = [DETAIL_BEGIN, "", *detail_body, DETAIL_END]

    # Replace the table and detail marker blocks with the rebuilt versions.
    out_lines: list[str] = []
    skip_until: str | None = None
    for line in lines:
        if skip_until is not None:
            if line.strip() == skip_until:
                skip_until = None
            continue
        if line.strip() == TABLE_BEGIN:
            out_lines.extend(new_table)
            skip_until = TABLE_END
            continue
        if line.strip() == DETAIL_BEGIN:
            out_lines.extend(new_detail)
            skip_until = DETAIL_END
            continue
        out_lines.append(line)

    results_md_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    return results_md_path


__all__ = [
    "TABLE_BEGIN",
    "TABLE_END",
    "update_results_markdown",
    "write_results_json",
]
