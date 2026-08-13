"""Plan §16 control/scheduling baselines and §17 experiment names.

Numeric grids that plan.md does not list belong in YAML as implementation
choices (plan §18). Do not put them in this module.
"""

from __future__ import annotations

from typing import Any

# Plan §16 paper-specified baseline *names* (not architectures).
PLAN_CONTROL_BASELINES = (
    "optimal_nonlinear_dp",
    "supervised_state_to_command",
    "generative_autoencoder",
)

# Plan §16 item 2: supervised κ=2 and κ=4 in Fig. 6.
PLAN_SUPERVISED_KAPPA = (2, 4)

PLAN_SCHEDULING_BASELINES = (
    "round_robin",
    "opportunistic",
)

# Plan §16 items 4–5: unscheduled → most recently received command.
PLAN_UNSCHEDULED_ACTION = "hold_last_command"

# IC keys that must exist so Figs. 7–8 and 10–11 can run (plan §18: existence ≠ paper number).
PLAN_SECTION_17_IC_KEYS = (
    "fig7_embedding_dims",
    "fig8_train_trajectories",
    "fig10_device_counts",
    "fig11_packet_loss",
)


def allow_non_baseline_embedding_dim(config: dict[str, Any]) -> bool:
    return bool(config.get("experiments", {}).get("allow_non_baseline_embedding_dim", False))


def assert_plan_section_16_17_config(config: dict[str, Any]) -> None:
    """
    Paper checks only: supervised κ set, SNR {5,10,20}, IC *keys* present.

    Does not freeze Fig. 7/8/10/11 numeric grids (plan §18).
    """
    errors: list[str] = []
    snr = list(config.get("wireless", {}).get("snr_targets_db", []))
    if snr != [5, 10, 20]:
        errors.append(f"wireless.snr_targets_db: plan §17 Fig. 9 requires [5, 10, 20], got {snr}")
    exp = config.get("experiments", {})
    for key in PLAN_SECTION_17_IC_KEYS:
        if key not in exp:
            errors.append(f"experiments.{key} must exist as an implementation choice (plan §17/§18)")
    if errors:
        raise ValueError("Plan §16–§17 config mismatch:\n  - " + "\n  - ".join(errors))
