"""Backward-compatible re-exports. Prefer ``ts_jepa.plan.predictor``."""

from ts_jepa.plan.predictor import PLAN_PREDICTOR, assert_plan_predictor_config

__all__ = ["PLAN_PREDICTOR", "assert_plan_predictor_config"]
