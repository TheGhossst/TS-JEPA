"""Backward-compatible re-exports. Prefer ``ts_jepa.plan.training``."""

from ts_jepa.plan.training import PLAN_JEPA_TRAINING, assert_plan_jepa_training_config

__all__ = ["PLAN_JEPA_TRAINING", "assert_plan_jepa_training_config"]
