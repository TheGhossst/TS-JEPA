"""Plan §2–§3 inverted cart-pole environment specification (docs/plan.md)."""

from __future__ import annotations

from typing import Any

from ts_jepa.plan.enforce import plan_enforced

# PAPER-SPECIFIED only. Native camera size, Gym vs custom backend, and how
# κ frames are stacked are NOT SPECIFIED — see docs/IMPLEMENTATION_CHOICES.md.
PLAN_ENVIRONMENT: dict[str, Any] = {
    "name": "inverted_cart_pole",
    "observation": "RGB",
    "encoder_height": 64,  # resize destination (Section IV.A)
    "encoder_width": 128,
    "channels": 3,
    "kappa": 2,
    "force_min_N": -20.0,
    "force_max_N": 20.0,
    "sampling_interval_ms": 1.0,
    "dt": 0.001,
    "trajectory_steps": 100,
    # Stored observations are at τ_o. Fig. 4 does not change this (plan §3).
    "observation_stride_steps": 1,
}


def assert_plan_environment_config(config: dict[str, Any]) -> None:
    """Raise ValueError when environment settings deviate from plan §2–§3."""
    if not plan_enforced(config):
        return
    errors: list[str] = []
    sim = config.get("simulation", {})
    inp = config.get("input", {})

    if str(sim.get("environment")) != PLAN_ENVIRONMENT["name"]:
        errors.append(
            f"simulation.environment: expected {PLAN_ENVIRONMENT['name']!r}, got {sim.get('environment')!r}"
        )
    if str(sim.get("state_representation")) != PLAN_ENVIRONMENT["observation"]:
        errors.append(
            f"simulation.state_representation: expected {PLAN_ENVIRONMENT['observation']!r}, "
            f"got {sim.get('state_representation')!r}"
        )

    tau_ms = float(sim.get("sampling_interval_ms", -1))
    dt = float(sim.get("dt", -1))
    if abs(tau_ms - PLAN_ENVIRONMENT["sampling_interval_ms"]) > 1e-9:
        errors.append(
            f"simulation.sampling_interval_ms: expected {PLAN_ENVIRONMENT['sampling_interval_ms']}, got {tau_ms}"
        )
    if abs(dt - PLAN_ENVIRONMENT["dt"]) > 1e-12:
        errors.append(f"simulation.dt: expected {PLAN_ENVIRONMENT['dt']}, got {dt}")
    if abs(dt - tau_ms / 1000.0) > 1e-12:
        errors.append("simulation.dt must equal sampling_interval_ms/1000 (plan §3 τ_o = 1 ms)")

    steps = int(sim.get("trajectory_steps", -1))
    if steps != PLAN_ENVIRONMENT["trajectory_steps"]:
        errors.append(
            f"simulation.trajectory_steps: expected {PLAN_ENVIRONMENT['trajectory_steps']}, got {steps}"
        )

    fmin = float(sim.get("control_min_N", float("nan")))
    fmax = float(sim.get("control_max_N", float("nan")))
    if abs(fmin - PLAN_ENVIRONMENT["force_min_N"]) > 1e-9:
        errors.append(f"simulation.control_min_N: expected {PLAN_ENVIRONMENT['force_min_N']}, got {fmin}")
    if abs(fmax - PLAN_ENVIRONMENT["force_max_N"]) > 1e-9:
        errors.append(f"simulation.control_max_N: expected {PLAN_ENVIRONMENT['force_max_N']}, got {fmax}")

    stride = int(sim.get("observation_stride_steps", -1))
    if stride != PLAN_ENVIRONMENT["observation_stride_steps"]:
        errors.append(
            f"simulation.observation_stride_steps: expected {PLAN_ENVIRONMENT['observation_stride_steps']} "
            f"(store every τ_o; Fig. 4 does not change sampling), got {stride}"
        )

    kappa = int(inp.get("kappa", -1))
    channels = int(inp.get("channels_per_rgb_frame", -1))
    if kappa != PLAN_ENVIRONMENT["kappa"]:
        errors.append(f"input.kappa: expected {PLAN_ENVIRONMENT['kappa']}, got {kappa}")
    if channels != PLAN_ENVIRONMENT["channels"]:
        errors.append(f"input.channels_per_rgb_frame: expected {PLAN_ENVIRONMENT['channels']}, got {channels}")

    resize = inp.get("resize")
    expected_resize = [PLAN_ENVIRONMENT["encoder_height"], PLAN_ENVIRONMENT["encoder_width"]]
    if resize != expected_resize:
        errors.append(f"input.resize: expected {expected_resize}, got {resize}")

    if errors:
        raise ValueError("Plan §3 environment config mismatch:\n  - " + "\n  - ".join(errors))
