"""Backward-compatible re-exports. Prefer ``ts_jepa.plan.temporal`` and ``ts_jepa.data.temporal``."""

from ts_jepa.data.temporal import (
    command_indices,
    context_frame_indices,
    describe_temporal_sample,
    max_valid_time_index,
    predicted_command_indices,
    target_end_indices,
)
from ts_jepa.plan.temporal import PLAN_TEMPORAL, assert_plan_temporal_config

__all__ = [
    "PLAN_TEMPORAL",
    "assert_plan_temporal_config",
    "command_indices",
    "context_frame_indices",
    "describe_temporal_sample",
    "max_valid_time_index",
    "predicted_command_indices",
    "target_end_indices",
]
