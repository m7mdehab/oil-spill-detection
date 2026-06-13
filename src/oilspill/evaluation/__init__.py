"""Evaluation and reporting harness for trained segmentation models.

This subpackage turns a checkpoint (or exported ONNX model) into the project's
reporting artifacts: a full metrics JSON, a confusion-matrix figure, per-class
precision-recall curves, a best/worst prediction gallery, and an auto-updated
results table in ``docs/results.md``. All metrics route through
:mod:`oilspill.metrics`, the single metric source for the repository.
"""

from __future__ import annotations

from oilspill.evaluation.evaluate import (
    EvaluationOutput,
    evaluate_model,
    rank_by_oil_iou,
)
from oilspill.evaluation.gallery import save_prediction_gallery
from oilspill.evaluation.model_loading import (
    build_model_from_config,
    load_model_from_checkpoint,
    load_onnx_session,
    predict_logits,
)
from oilspill.evaluation.plots import plot_confusion_matrix, plot_pr_curves
from oilspill.evaluation.report import (
    update_results_markdown,
    write_results_json,
)

__all__ = [
    "EvaluationOutput",
    "build_model_from_config",
    "evaluate_model",
    "load_model_from_checkpoint",
    "load_onnx_session",
    "plot_confusion_matrix",
    "plot_pr_curves",
    "predict_logits",
    "rank_by_oil_iou",
    "save_prediction_gallery",
    "update_results_markdown",
    "write_results_json",
]
