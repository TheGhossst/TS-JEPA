"""Backward-compatible re-exports. Prefer ``ts_jepa.plan.preprocessing``."""

from ts_jepa.plan.preprocessing import (
    EVAL_PIPELINE_STAGES,
    PLAN_PREPROCESSING,
    TRAINING_PIPELINE_STAGES,
    assert_plan_preprocessing_config,
)

__all__ = [
    "EVAL_PIPELINE_STAGES",
    "PLAN_PREPROCESSING",
    "TRAINING_PIPELINE_STAGES",
    "assert_plan_preprocessing_config",
]
