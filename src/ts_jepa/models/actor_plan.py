"""Backward-compatible re-exports. Prefer ``ts_jepa.plan.actor``."""

from ts_jepa.plan.actor import (
    PLAN_SEMANTIC_ACTOR,
    PLAN_SEMANTIC_ACTOR_TRAINING,
    assert_plan_semantic_actor_config,
    assert_plan_semantic_actor_training_config,
)

__all__ = [
    "PLAN_SEMANTIC_ACTOR",
    "PLAN_SEMANTIC_ACTOR_TRAINING",
    "assert_plan_semantic_actor_config",
    "assert_plan_semantic_actor_training_config",
]
