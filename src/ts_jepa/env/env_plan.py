"""Plan §3 custom inverted cart-pole environment specification."""

from __future__ import annotations

from typing import Any

PLAN_ENVIRONMENT: dict[str, Any] = {
    "name": "inverted_cart_pole",
    "observation": "RGB",
    "backend": "custom",  # plan §3.2 — not Gym CartPole-v1
    "render_height": 64,
    "render_width": 128,
    "channels": 3,
    "kappa": 2,
    "context_channels": 6,  # κ × RGB channels
    "context_tensor_hw": [64, 128],
    "force_min_N": -20.0,
    "force_max_N": 20.0,
    "sampling_interval_ms": 1.0,
    "dt": 0.001,
    "multi_frame_tensor_construction": "channel_concat",
}


def assert_plan_environment_config(config: dict[str, Any]) -> None:
    """Raise ValueError when environment settings deviate from plan §3."""
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
        errors.append(f"simulation.dt must equal sampling_interval_ms/1000 (plan §3.2 τ_o = 1 ms)")

    fmin = float(sim.get("control_min_N", float("nan")))
    fmax = float(sim.get("control_max_N", float("nan")))
    if abs(fmin - PLAN_ENVIRONMENT["force_min_N"]) > 1e-9:
        errors.append(f"simulation.control_min_N: expected {PLAN_ENVIRONMENT['force_min_N']}, got {fmin}")
    if abs(fmax - PLAN_ENVIRONMENT["force_max_N"]) > 1e-9:
        errors.append(f"simulation.control_max_N: expected {PLAN_ENVIRONMENT['force_max_N']}, got {fmax}")

    rh = int(sim.get("render_height", -1))
    rw = int(sim.get("render_width", -1))
    if rh != PLAN_ENVIRONMENT["render_height"] or rw != PLAN_ENVIRONMENT["render_width"]:
        errors.append(
            f"simulation render size: expected {PLAN_ENVIRONMENT['render_height']}×"
            f"{PLAN_ENVIRONMENT['render_width']}, got {rh}×{rw}"
        )

    kappa = int(inp.get("kappa", -1))
    channels = int(inp.get("channels_per_rgb_frame", -1))
    if kappa != PLAN_ENVIRONMENT["kappa"]:
        errors.append(f"input.kappa: expected {PLAN_ENVIRONMENT['kappa']}, got {kappa}")
    if channels != PLAN_ENVIRONMENT["channels"]:
        errors.append(f"input.channels_per_rgb_frame: expected {PLAN_ENVIRONMENT['channels']}, got {channels}")

    resize = inp.get("resize")
    expected_resize = [PLAN_ENVIRONMENT["render_height"], PLAN_ENVIRONMENT["render_width"]]
    if resize != expected_resize:
        errors.append(f"input.resize: expected {expected_resize}, got {resize}")

    construction = inp.get("multi_frame_tensor_construction")
    if construction != PLAN_ENVIRONMENT["multi_frame_tensor_construction"]:
        errors.append(
            f"input.multi_frame_tensor_construction: expected "
            f"{PLAN_ENVIRONMENT['multi_frame_tensor_construction']!r}, got {construction!r}"
        )

    if errors:
        raise ValueError("Plan §3 environment config mismatch:\n  - " + "\n  - ".join(errors))
