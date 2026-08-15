"""Working-overlay plan enforcement.

Paper-faithful configs omit ``plan.enforce`` (default True). The working
overlay sets ``plan.enforce: false`` so ``assert_plan_*`` become no-ops.
"""

from __future__ import annotations

from typing import Any


def plan_enforced(config: dict[str, Any] | None) -> bool:
    """True unless the config explicitly disables paper-plan asserts."""
    if not isinstance(config, dict):
        return True
    plan = config.get("plan") or {}
    if "enforce" in plan:
        return bool(plan["enforce"])
    return True


def is_working_mode(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    return str((config.get("plan") or {}).get("mode", "")).strip().lower() == "working"


def jepa_uses_kappa_stack(config: dict[str, Any] | None) -> bool:
    """Working overlay stacks κ frames into Ψ; paper Algorithm 1 uses one RGB frame."""
    if not isinstance(config, dict):
        return False
    env = config.get("environment") or {}
    obs = str(env.get("jepa_observation", "current_frame")).strip().lower()
    return obs in {"kappa_stack", "kappa", "stacked", "channel_concat"}


def jepa_in_channels(config: dict[str, Any]) -> int:
    ch = int(config["input"]["channels_per_rgb_frame"])
    if jepa_uses_kappa_stack(config):
        return ch * int(config["input"]["kappa"])
    return ch
