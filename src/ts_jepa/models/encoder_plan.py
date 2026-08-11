"""Backward-compatible re-exports. Prefer ``ts_jepa.plan.encoder``."""

from ts_jepa.plan.encoder import PLAN_ENCODER, PLAN_TARGET_ENCODER, assert_plan_encoder_config

__all__ = ["PLAN_ENCODER", "PLAN_TARGET_ENCODER", "assert_plan_encoder_config"]
