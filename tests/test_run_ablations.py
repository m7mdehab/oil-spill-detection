"""Fast tests for the ablation runner.

These exercise the real ``generate`` / ``commands`` / ``aggregate`` code paths
against tiny on-disk fixtures in ``tmp_path``. No training, no network, no Modal.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from oilspill.training.config import TrainConfig

# Import the script module by path (it lives in scripts/, not an installed pkg).
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_ablations.py"
_spec = importlib.util.spec_from_file_location("run_ablations", _SCRIPT)
assert _spec is not None and _spec.loader is not None
run_ablations = importlib.util.module_from_spec(_spec)
sys.modules["run_ablations"] = run_ablations
_spec.loader.exec_module(run_ablations)


def _base_config(arch: str) -> dict[str, Any]:
    """A minimal but valid per-arch base config (validates as TrainConfig)."""
    return {
        "model": {"arch": arch, "encoder": "resnet34", "num_classes": 5},
        "data": {"image_size": 256, "batch_size": 4},
        "optim": {"lr": 0.0003, "epochs": 50},
        "loss": {"type": "dice_focal", "class_weights": "auto"},
        "runtime": {"seed": 7},
        "mlflow": {"experiment_name": "oil-spill-segmentation"},
    }


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """A configs/ dir containing a base config for every matrix architecture."""
    d = tmp_path / "configs"
    d.mkdir()
    for arch in run_ablations.ARCHITECTURES:
        (d / f"{arch}.yaml").write_text(yaml.safe_dump(_base_config(arch)), encoding="utf-8")
    return d


# --- generate -----------------------------------------------------------------


def test_generate_writes_one_valid_config_per_cell(tmp_path: Path, base_dir: Path) -> None:
    out_dir = tmp_path / "ablations"
    written = run_ablations.generate(
        run_ablations.ARCHITECTURES,
        epochs=40,
        seed=42,
        max_cells=None,
        out_dir=out_dir,
        base_dir=base_dir,
    )

    expected = len(run_ablations.ARCHITECTURES) * len(run_ablations.LOSS_VARIANTS)
    assert len(written) == expected

    for path in written:
        # Every generated file parses and validates as a real TrainConfig.
        cfg = TrainConfig.from_yaml(path)
        assert cfg.optim.epochs == 40
        assert cfg.runtime.seed == 42
        assert cfg.mlflow.run_name is not None
        assert cfg.mlflow.run_name.startswith("abl-")


def test_generate_applies_loss_overrides(tmp_path: Path, base_dir: Path) -> None:
    out_dir = tmp_path / "ablations"
    run_ablations.generate(
        ("Unet",),
        epochs=10,
        seed=1,
        max_cells=None,
        out_dir=out_dir,
        base_dir=base_dir,
    )

    ce = TrainConfig.from_yaml(out_dir / "Unet__ce_weighted.yaml")
    assert ce.loss.type == "ce"
    assert ce.loss.class_weights == "auto"

    df = TrainConfig.from_yaml(out_dir / "Unet__dice_focal.yaml")
    assert df.loss.type == "dice_focal"
    assert df.loss.class_weights == "auto"


def test_generate_skips_missing_base_config(tmp_path: Path, base_dir: Path) -> None:
    # Remove one base config; its two cells should be skipped, not error.
    (base_dir / "foundation.yaml").unlink()
    out_dir = tmp_path / "ablations"
    written = run_ablations.generate(
        run_ablations.ARCHITECTURES,
        epochs=40,
        seed=42,
        max_cells=None,
        out_dir=out_dir,
        base_dir=base_dir,
    )
    names = {p.name for p in written}
    assert "foundation__ce_weighted.yaml" not in names
    assert "Unet__ce_weighted.yaml" in names


def test_generate_max_cells(tmp_path: Path, base_dir: Path) -> None:
    out_dir = tmp_path / "ablations"
    written = run_ablations.generate(
        run_ablations.ARCHITECTURES,
        epochs=40,
        seed=42,
        max_cells=3,
        out_dir=out_dir,
        base_dir=base_dir,
    )
    assert len(written) == 3


# --- commands -----------------------------------------------------------------


def test_commands_emits_two_commands_per_cell(capsys: pytest.CaptureFixture[str]) -> None:
    lines = run_ablations.commands(("Unet", "segformer"), gpu="L4", max_cells=None)
    out = capsys.readouterr().out

    modal_lines = [ln for ln in lines if ln.startswith("uv run modal run")]
    eval_lines = [ln for ln in lines if "scripts/evaluate.py" in ln]

    n_cells = 2 * len(run_ablations.LOSS_VARIANTS)
    assert len(modal_lines) == n_cells
    assert len(eval_lines) == n_cells
    # Cost estimate and GPU passthrough are present in the printed output.
    assert "--gpu L4" in out
    assert "cost estimate" in out


def test_commands_respects_max_cells(capsys: pytest.CaptureFixture[str]) -> None:
    run_ablations.commands(run_ablations.ARCHITECTURES, gpu="A10G", max_cells=1)
    out = capsys.readouterr().out
    assert out.count("uv run modal run") == 1
    assert "--gpu A10G" in out


# --- aggregate ----------------------------------------------------------------


def _write_result(results_dir: Path, run_name: str, oil_iou: float) -> None:
    payload = {
        "meta": {"run_name": run_name, "split": "test"},
        "metrics": {
            "aggregate": {
                "oil_iou": oil_iou,
                "oil_recall": oil_iou + 0.1,
                "mean_iou": 0.6,
                "macro_f1": 0.7,
                "pixel_accuracy": 0.95,
            }
        },
    }
    (results_dir / f"{run_name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_builds_sorted_table(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_result(results_dir, "abl-unet-ce_weighted", oil_iou=0.40)
    _write_result(results_dir, "abl-unet-dice_focal", oil_iou=0.55)
    # A non-ablation result that must be ignored.
    _write_result(results_dir, "unet-r34-baseline", oil_iou=0.99)

    doc = tmp_path / "ablations.md"
    doc.write_text(
        f"intro\n\n{run_ablations.TABLE_BEGIN}\n\nplaceholder\n\n{run_ablations.TABLE_END}\nend\n",
        encoding="utf-8",
    )

    results = run_ablations.aggregate(results_dir=results_dir, doc_path=doc)
    assert len(results) == 2  # baseline ignored

    text = doc.read_text(encoding="utf-8")
    # Winner (higher oil IoU) is dice_focal and is marked + sorted first.
    body = text.split(run_ablations.TABLE_BEGIN, 1)[1].split(run_ablations.TABLE_END, 1)[0]
    rows = [ln for ln in body.splitlines() if ln.startswith("| Unet")]
    assert rows[0].startswith("| Unet | dice_focal")
    assert "best oil IoU" in rows[0]
    assert "best oil IoU" not in rows[1]
    # Original surrounding text is preserved.
    assert "intro" in text
    assert "end" in text


def test_aggregate_requires_markers(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    doc = tmp_path / "ablations.md"
    doc.write_text("no markers here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="markers"):
        run_ablations.aggregate(results_dir=results_dir, doc_path=doc)


def test_parse_run_name_roundtrip() -> None:
    for cell in run_ablations.matrix():
        parsed = run_ablations._parse_run_name(cell.run_name)
        assert parsed == (cell.arch, cell.loss)
