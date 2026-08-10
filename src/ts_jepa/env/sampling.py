"""Observation sampling cadence helpers (ablation / diagnostics).

Physics integration timestep (ODE dt) is independent of observation cadence.
`sampling_interval_ms` in YAML was historically documentation-only; these helpers
make the intended stride explicit for controlled experiments.
"""

from __future__ import annotations


def physics_dt_ms(dt_s: float) -> float:
    return float(dt_s) * 1000.0


def observation_stride_steps(sampling_interval_ms: float, dt_s: float) -> int:
    """
    Number of ODE steps between stored/rendered observations.

    Requires sampling_interval_ms to be an integer multiple of dt_ms.
    """
    dt_ms = physics_dt_ms(dt_s)
    if dt_ms <= 0:
        raise ValueError(f"dt must be > 0, got {dt_s}")
    ratio = float(sampling_interval_ms) / dt_ms
    stride = int(round(ratio))
    if stride < 1:
        raise ValueError(
            f"observation stride must be >= 1; got sampling_interval_ms={sampling_interval_ms}, dt_s={dt_s}"
        )
    if abs(ratio - stride) > 1e-9:
        raise ValueError(
            f"sampling_interval_ms={sampling_interval_ms} is not an integer multiple of "
            f"physics dt_ms={dt_ms} (ratio={ratio})"
        )
    return stride


def num_observations_for_fixed_physics_budget(
    total_physics_steps: int,
    observation_stride: int,
) -> int:
    """Keep total integrated physical time fixed: observations = budget // stride."""
    stride = int(observation_stride)
    budget = int(total_physics_steps)
    if stride < 1:
        raise ValueError(f"observation_stride must be >= 1, got {stride}")
    if budget < stride:
        raise ValueError(f"total_physics_steps={budget} < observation_stride={stride}")
    if budget % stride != 0:
        raise ValueError(
            f"total_physics_steps={budget} must be divisible by observation_stride={stride} "
            "to keep physical duration exact"
        )
    return budget // stride


def resolve_observation_cadence(
    sampling_interval_ms: float,
    dt_s: float,
    total_physics_steps: int,
    *,
    allow_truncated_budget: bool = False,
) -> dict[str, float | int | bool]:
    """
    Map sampling interval -> stride, observation count, and effective physics budget.

    When ``allow_truncated_budget`` is True and nominal budget is not divisible by stride,
    uses floor(budget / stride) observations and an effective budget of num_obs * stride.
    """
    stride = observation_stride_steps(sampling_interval_ms, dt_s)
    budget = int(total_physics_steps)
    truncated = False
    try:
        num_obs = num_observations_for_fixed_physics_budget(budget, stride)
        effective_budget = budget
    except ValueError:
        if not allow_truncated_budget:
            raise
        num_obs = budget // stride
        if num_obs < 1:
            raise ValueError(
                f"total_physics_steps={budget} too small for observation_stride={stride}"
            )
        effective_budget = num_obs * stride
        truncated = True
    return {
        "sampling_interval_ms": int(sampling_interval_ms),
        "observation_stride": stride,
        "num_observations": num_obs,
        "physics_budget_nominal": budget,
        "physics_budget_effective": effective_budget,
        "physical_duration_s": float(effective_budget) * float(dt_s),
        "budget_truncated": truncated,
    }


def resolve_observation_cadence_from_config(
    config: dict,
    *,
    allow_truncated_budget: bool = False,
) -> dict[str, float | int | bool]:
    """Resolve cadence from ``config['simulation']`` (trajectory_steps = physics budget)."""
    sim = config["simulation"]
    return resolve_observation_cadence(
        float(sim.get("sampling_interval_ms", 1.0)),
        float(sim["dt"]),
        int(sim["trajectory_steps"]),
        allow_truncated_budget=allow_truncated_budget,
    )


def jepa_training_windows_per_trajectory(num_observations: int, kp: int) -> int:
    """
    Valid TrajectoryDataset (context, Kp-target) windows per trajectory.

    Matches ``TrajectoryDataset`` index construction:
      max_start = N - Kp - 1
      time_index in range(max_start + 1)  =>  N - Kp windows when N >= Kp.
    """
    n = int(num_observations)
    k = int(kp)
    if n < k:
        return 0
    return n - k


def actor_training_samples_per_trajectory(num_observations: int) -> int:
    """Actor datasets use one sample per stored observation."""
    return int(num_observations)
