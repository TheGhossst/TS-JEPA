"""Evaluation metrics and closed-loop reports."""

from ts_jepa.evaluation.checkpoints import resolve_jepa_checkpoint_from_actor, resolve_run_checkpoint
from ts_jepa.evaluation.evaluate import (
    baseline_report,
    evaluate_actor_nmae,
    evaluate_closed_loop,
    evaluate_closed_loop_full_information_diagnostic,
    evaluate_closed_loop_stability,
    evaluate_consecutive_frame_mape,
    evaluate_embedding_tsne,
    evaluate_fig4_sampling_rate_mape,
    evaluate_prediction_horizon_nmae,
    evaluate_with_scheduler,
    validate_baseline,
    write_evaluation_artifacts,
)
from ts_jepa.evaluation.metrics import (
    communication_bits_embedding,
    communication_bits_rgb,
    communication_reduction_report,
    consecutive_frame_mape,
    control_score,
    latent_cosine_by_horizon,
    nmae,
    physical_force_range_n,
)

__all__ = [
    "communication_bits_embedding",
    "communication_bits_rgb",
    "communication_reduction_report",
    "consecutive_frame_mape",
    "control_score",
    "nmae",
    "latent_cosine_by_horizon",
    "physical_force_range_n",
    "audit_actor_training",
    "audit_actor_training_from_runs",
    "baseline_report",
    "evaluate_actor_nmae",
    "evaluate_closed_loop",
    "evaluate_closed_loop_full_information_diagnostic",
    "evaluate_closed_loop_stability",
    "evaluate_consecutive_frame_mape",
    "evaluate_fig4_sampling_rate_mape",
    "evaluate_embedding_tsne",
    "evaluate_prediction_horizon_nmae",
    "evaluate_with_scheduler",
    "validate_baseline",
    "write_evaluation_artifacts",
    "resolve_jepa_checkpoint_from_actor",
    "resolve_run_checkpoint",
]


def __getattr__(name: str):
    if name == "audit_actor_training":
        from ts_jepa.evaluation.actor_training_audit import audit_actor_training

        return audit_actor_training
    if name == "audit_actor_training_from_runs":
        from ts_jepa.evaluation.actor_training_audit import audit_actor_training_from_runs

        return audit_actor_training_from_runs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
