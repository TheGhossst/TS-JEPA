"""Backward-compatible re-exports. Prefer ``ts_jepa.plan.loss``."""

from ts_jepa.plan.loss import PLAN_JEPA_LOSS, assert_plan_jepa_loss_config

__all__ = ["PLAN_JEPA_LOSS", "assert_plan_jepa_loss_config"]
