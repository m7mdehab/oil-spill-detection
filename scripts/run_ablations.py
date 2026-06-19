"""Imbalance & loss ablation runner.

Defines and drives the ablation matrix {architecture} x {loss}, where the only
thing that varies within an architecture is the segmentation loss:

* ``ce_weighted`` -- class-weighted cross-entropy (inverse-frequency weights).
* ``dice_focal``  -- Dice + Focal combo (also class-weighted on the CE/Focal term).

Everything else (seed, epoch budget, data, optimiser) is held fixed across the
two loss cells of an architecture so the comparison is fair. The dataset is
heavily imbalanced (oil class ~0.99% of pixels, ~89:1 majority:oil -- see
``docs/data_report.md``), so the headline metrics are **oil-class IoU** and
**oil-class recall**, never pixel accuracy. See ``docs/ablations.md`` for the
full protocol.

Modes
-----
``generate``
    Write one complete ``TrainConfig`` YAML per matrix cell into
    ``configs/ablations/<arch>__<loss>.yaml``, derived from the per-arch base
    config (``configs/<arch>.yaml``) with the loss block and budget overridden.

``commands``
    Print, for every cell, the exact ``modal run`` training command followed by
    the ``evaluate.py`` command the operator runs once the checkpoint lands.
    Also prints a Modal cost estimate.

``aggregate``
    Read every ``docs/results/<run>.json`` that matches an ablation run name and
    write a comparison table into ``docs/ablations.md`` (sorted by oil IoU,
    descending; winning cell marked).

This script only *orchestrates* the existing training/evaluation entry points;
it never trains a model itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Allow running as a plain script (``python scripts/run_ablations.py``) without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from oilspill.training.config import TrainConfig

# --- Matrix definition --------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Architectures in the canonical matrix. Each must have a base config at
# ``configs/<arch>.yaml``; cells whose base config is missing are skipped (with a
# warning) so the matrix can be run with whatever architectures are wired up.
ARCHITECTURES: tuple[str, ...] = ("Unet", "DeepLabV3Plus", "segformer", "foundation")

# Loss cells: name -> the ``loss`` block override applied on top of the base
# config. ``class_weights: auto`` derives inverse-frequency weights from the
# training masks, up-weighting the rare oil class.
LOSS_VARIANTS: dict[str, dict[str, Any]] = {
    "ce_weighted": {"type": "ce", "class_weights": "auto"},
    "dice_focal": {"type": "dice_focal", "class_weights": "auto", "focal_gamma": 2.0},
}

# Fixed fairness knobs shared by every cell.
DEFAULT_SEED = 42
DEFAULT_EPOCHS = 40

# Modal L4 per-run cost band (rough), used for the printed estimate.
COST_PER_RUN_LOW = 0.30
COST_PER_RUN_HIGH = 0.60

# Markers in docs/ablations.md that ``aggregate`` rewrites in place.
TABLE_BEGIN = "<!-- ABLATIONS_TABLE_BEGIN -->"
TABLE_END = "<!-- ABLATIONS_TABLE_END -->"

CONFIGS_DIR = REPO_ROOT / "configs"
ABLATIONS_DIR = CONFIGS_DIR / "ablations"
RESULTS_DIR = REPO_ROOT / "docs" / "results"
ABLATIONS_DOC = REPO_ROOT / "docs" / "ablations.md"


@dataclass(frozen=True)
class Cell:
    """A single ablation matrix cell."""

    arch: str
    loss: str

    @property
    def run_name(self) -> str:
        """Stable identifier used for config filename, mlflow run and eval tag."""
        return f"abl-{self.arch.lower()}-{self.loss}"

    @property
    def config_filename(self) -> str:
        return f"{self.arch}__{self.loss}.yaml"


def matrix(architectures: tuple[str, ...] = ARCHITECTURES) -> list[Cell]:
    """All cells of the {arch} x {loss} matrix, in a stable order."""
    return [Cell(arch=arch, loss=loss) for arch in architectures for loss in LOSS_VARIANTS]


def _apply_max_cells(cells: list[Cell], max_cells: int | None) -> list[Cell]:
    if max_cells is not None and max_cells >= 0:
        return cells[:max_cells]
    return cells


# --- generate -----------------------------------------------------------------


def build_config(
    cell: Cell,
    *,
    base_dir: Path = CONFIGS_DIR,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build the config dict for ``cell`` from its per-arch base config.

    Reads ``<base_dir>/<arch>.yaml``, overrides the loss block, pins the epoch
    budget / seed for fairness, and sets a stable mlflow run name.
    """
    base_path = base_dir / f"{cell.arch}.yaml"
    raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"base config {base_path} must be a mapping")
    config: dict[str, Any] = raw

    # Loss override -- the only intentional difference within an architecture.
    config["loss"] = dict(LOSS_VARIANTS[cell.loss])

    # Fixed budget / seed across all cells for a fair comparison.
    config.setdefault("optim", {})
    config["optim"]["epochs"] = epochs
    config.setdefault("runtime", {})
    config["runtime"]["seed"] = seed
    config.setdefault("data", {})
    config["data"]["split_seed"] = seed

    # Stable, identifiable run name for MLflow and downstream eval.
    config.setdefault("mlflow", {})
    config["mlflow"]["run_name"] = cell.run_name

    return config


def generate(
    architectures: tuple[str, ...],
    *,
    epochs: int,
    seed: int,
    max_cells: int | None,
    out_dir: Path = ABLATIONS_DIR,
    base_dir: Path = CONFIGS_DIR,
) -> list[Path]:
    """Write one validated YAML config per matrix cell. Returns written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = _apply_max_cells(matrix(architectures), max_cells)

    written: list[Path] = []
    for cell in cells:
        base_path = base_dir / f"{cell.arch}.yaml"
        if not base_path.exists():
            print(
                f"skip {cell.arch}: no base config at {base_path} (architecture not wired up yet)",
                file=sys.stderr,
            )
            continue

        config = build_config(cell, base_dir=base_dir, epochs=epochs, seed=seed)
        # Validate before writing so a generated file is never an invalid config.
        TrainConfig.model_validate(config)

        dest = out_dir / cell.config_filename
        header = (
            f"# Ablation cell: arch={cell.arch}, loss={cell.loss}.\n"
            f"# Generated by scripts/run_ablations.py -- edit the matrix there, not here.\n"
            f"# Fixed for fairness: seed={seed}, epochs={epochs}. "
            f"Headline metrics: oil IoU & oil recall.\n"
        )
        body = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
        dest.write_text(header + body, encoding="utf-8")
        written.append(dest)

    return written


# --- commands -----------------------------------------------------------------


def _modal_command(config_path: Path, gpu: str) -> str:
    rel = config_path.relative_to(REPO_ROOT).as_posix()
    return f"uv run modal run --detach scripts/modal_train.py::main --config {rel} --gpu {gpu}"


def _evaluate_command(cell: Cell) -> str:
    # The checkpoint lands under artifacts/checkpoints/<run>/best.pt once the
    # Modal job is pulled back locally (see scripts/modal_train.py).
    ckpt = f"artifacts/checkpoints/{cell.run_name}/best.pt"
    return (
        f"uv run python scripts/evaluate.py --checkpoint {ckpt} "
        f"--split test --tag {cell.loss} --name {cell.run_name}"
    )


def commands(
    architectures: tuple[str, ...],
    *,
    gpu: str,
    max_cells: int | None,
    out_dir: Path = ABLATIONS_DIR,
) -> list[str]:
    """Print the modal + evaluate commands per cell. Returns the lines printed."""
    cells = _apply_max_cells(matrix(architectures), max_cells)
    lines: list[str] = []

    for cell in cells:
        config_path = out_dir / cell.config_filename
        lines.append(f"# {cell.arch} x {cell.loss}  (run: {cell.run_name})")
        lines.append(_modal_command(config_path, gpu))
        lines.append(_evaluate_command(cell))
        lines.append("")

    n = len(cells)
    cost_low = n * COST_PER_RUN_LOW
    cost_high = n * COST_PER_RUN_HIGH
    lines.append(
        f"# cost estimate: {n} run(s) on Modal {gpu} "
        f"~= ${cost_low:.2f}-${cost_high:.2f} "
        f"(@ ${COST_PER_RUN_LOW:.2f}-${COST_PER_RUN_HIGH:.2f}/run)."
    )
    lines.append(
        "# to stay within free credits: reduce --max-cells, drop architectures, or lower --epochs."
    )

    for line in lines:
        print(line)
    return lines


# --- aggregate ----------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    """The headline numbers pulled from one ablation result JSON."""

    arch: str
    loss: str
    oil_iou: float
    oil_recall: float
    mean_iou: float
    macro_f1: float
    pixel_accuracy: float


def _parse_run_name(run_name: str) -> tuple[str, str] | None:
    """Recover (arch, loss) from an ``abl-<arch>-<loss>`` run name, if it is one."""
    if not run_name.startswith("abl-"):
        return None
    for loss in LOSS_VARIANTS:
        suffix = f"-{loss}"
        if run_name.endswith(suffix):
            arch_slug = run_name[len("abl-") : -len(suffix)]
            # Map the lowercase slug back to the canonical arch spelling.
            for arch in ARCHITECTURES:
                if arch.lower() == arch_slug:
                    return arch, loss
            return arch_slug, loss
    return None


def collect_results(results_dir: Path = RESULTS_DIR) -> list[CellResult]:
    """Read every ablation result JSON under ``results_dir``."""
    out: list[CellResult] = []
    if not results_dir.exists():
        return out

    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        run_name = data.get("meta", {}).get("run_name", path.stem)
        parsed = _parse_run_name(run_name)
        if parsed is None:
            continue
        arch, loss = parsed
        agg = data["metrics"]["aggregate"]
        out.append(
            CellResult(
                arch=arch,
                loss=loss,
                oil_iou=float(agg["oil_iou"]),
                oil_recall=float(agg["oil_recall"]),
                mean_iou=float(agg["mean_iou"]),
                macro_f1=float(agg["macro_f1"]),
                pixel_accuracy=float(agg["pixel_accuracy"]),
            )
        )
    return out


def _build_table(results: list[CellResult]) -> list[str]:
    """Render the comparison table (oil IoU desc), marking the winner."""
    header = "| arch | loss | oil IoU | oil recall | mean IoU | macro F1 | pixel acc* |"
    divider = "| --- | --- | --- | --- | --- | --- | --- |"
    if not results:
        return [
            header,
            divider,
            "| _(populated by `run_ablations.py aggregate` after runs)_ |  |  |  |  |  |  |",
        ]

    ordered = sorted(results, key=lambda r: r.oil_iou, reverse=True)
    best_oil_iou = ordered[0].oil_iou
    rows: list[str] = [header, divider]
    for r in ordered:
        win = " **(best oil IoU)**" if r.oil_iou == best_oil_iou else ""
        rows.append(
            f"| {r.arch} | {r.loss} | {r.oil_iou:.4f}{win} | {r.oil_recall:.4f} | "
            f"{r.mean_iou:.4f} | {r.macro_f1:.4f} | {r.pixel_accuracy:.4f} |"
        )
    return rows


def aggregate(
    *,
    results_dir: Path = RESULTS_DIR,
    doc_path: Path = ABLATIONS_DOC,
) -> list[CellResult]:
    """Rebuild the results table inside ``doc_path`` from the result JSONs."""
    results = collect_results(results_dir)
    table = _build_table(results)

    text = doc_path.read_text(encoding="utf-8")
    if TABLE_BEGIN not in text or TABLE_END not in text:
        raise ValueError(f"{doc_path} is missing the {TABLE_BEGIN}/{TABLE_END} markers")

    before, rest = text.split(TABLE_BEGIN, 1)
    _, after = rest.split(TABLE_END, 1)
    new = "".join([before, TABLE_BEGIN, "\n\n", "\n".join(table), "\n\n", TABLE_END, after])
    doc_path.write_text(new, encoding="utf-8")
    return results


# --- CLI ----------------------------------------------------------------------


def _arch_tuple(values: list[str] | None) -> tuple[str, ...]:
    return tuple(values) if values else ARCHITECTURES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    arch_help = "restrict architectures (repeatable)"
    cells_help = "cap the number of cells (budget)"

    gen = sub.add_parser("generate", help="write a TrainConfig YAML per matrix cell")
    gen.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="epoch budget for every cell"
    )
    gen.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed shared by every cell")
    gen.add_argument("--arch", action="append", choices=list(ARCHITECTURES), help=arch_help)
    gen.add_argument("--max-cells", type=int, default=None, help=cells_help)

    cmd = sub.add_parser("commands", help="print modal + evaluate commands per cell")
    cmd.add_argument("--gpu", default="L4", help="Modal GPU type passthrough")
    cmd.add_argument("--arch", action="append", choices=list(ARCHITECTURES), help=arch_help)
    cmd.add_argument("--max-cells", type=int, default=None, help=cells_help)

    sub.add_parser("aggregate", help="rebuild the results table in docs/ablations.md")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.mode == "generate":
        written = generate(
            _arch_tuple(args.arch),
            epochs=args.epochs,
            seed=args.seed,
            max_cells=args.max_cells,
        )
        print(f"generated {len(written)} config(s):")
        for path in written:
            print(f"  {path.relative_to(REPO_ROOT).as_posix()}")
        return 0

    if args.mode == "commands":
        commands(_arch_tuple(args.arch), gpu=args.gpu, max_cells=args.max_cells)
        return 0

    if args.mode == "aggregate":
        results = aggregate()
        doc_rel = ABLATIONS_DOC.relative_to(REPO_ROOT).as_posix()
        print(f"aggregated {len(results)} ablation result(s) into {doc_rel}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
