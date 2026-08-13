"""Build plan §3 inverted cart-pole environment from config."""

from __future__ import annotations

from typing import Any

from ts_jepa.env.cartpole_ode import CartPoleParams
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv


def build_inverted_cartpole_env(config: dict[str, Any]) -> InvertedCartPoleEnv:
    """Construct `InvertedCartPoleEnv` from baseline config (`simulation` block)."""
    sim = config["simulation"]
    phys = sim.get("physics", {})
    params = CartPoleParams(
        cart_mass=float(phys.get("cart_mass", 1.0)),
        pole_mass=float(phys.get("pole_mass", 0.1)),
        pole_length=float(phys.get("pole_length", 0.5)),
        gravity=float(phys.get("gravity", 9.81)),
        track_limit=float(phys.get("track_limit", 2.4)),
    )
    return InvertedCartPoleEnv(
        render_height=int(sim["render_height"]),
        render_width=int(sim["render_width"]),
        dt=float(sim["dt"]),
        force_min=float(sim["control_min_N"]),
        force_max=float(sim["control_max_N"]),
        desired_state=list(sim["desired_state"]),
        params=params,
        process_noise_std=float(sim.get("process_noise_std", 0.0)),
        init_noise=float(sim.get("init_noise", 0.05)),
        observation_stride_steps=int(sim.get("observation_stride_steps", 1)),
    )
