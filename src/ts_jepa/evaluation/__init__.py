"""Evaluation metrics and closed-loop reports."""

from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.evaluation.evaluate import (
    baseline_report,
    evaluate_closed_loop,
    evaluate_embedding_tsne,
    evaluate_prediction_horizon_nmae,
    evaluate_with_scheduler,
    validate_baseline,
    write_evaluation_artifacts,
)
from ts_jepa.evaluation.metrics import (
    communication_bits_embedding,
    communication_bits_rgb,
    control_score,
    nmae,
)

__all__ = [
    "communication_bits_embedding",
    "communication_bits_rgb",
    "control_score",
    "nmae",
    "baseline_report",
    "evaluate_closed_loop",
    "evaluate_embedding_tsne",
    "evaluate_prediction_horizon_nmae",
    "evaluate_with_scheduler",
    "validate_baseline",
    "write_evaluation_artifacts",
    "resolve_run_checkpoint",
]
