from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ts_jepa.env.cartpole_ode import CartPoleODE, CartPoleParams
from ts_jepa.env.renderer import CartPoleRenderer


class ControlPolicy(Protocol):
    def act(self, state: np.ndarray) -> float: ...


class InvertedCartPoleEnv:
    """Inverted cart-pole environment with RGB observations and force actuation."""

    def __init__(
        self,
        render_height: int = 64,
        render_width: int = 128,
        dt: float = 0.001,
        force_min: float = -20.0,
        force_max: float = 20.0,
        desired_state: list[float] | None = None,
        params: CartPoleParams | None = None,
        process_noise_std: float = 0.0,
        init_noise: float = 0.05,
    ) -> None:
        self.params = params or CartPoleParams()
        self.ode = CartPoleODE(params=self.params, dt=dt)
        self.renderer = CartPoleRenderer(
            height=render_height,
            width=render_width,
            params=self.params,
        )
        self.force_min = float(force_min)
        self.force_max = float(force_max)
        self.desired_state = np.asarray(
            desired_state if desired_state is not None else [0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.process_noise_std = float(process_noise_std)
        self.init_noise = float(init_noise) if init_noise is not None else 0.05
        self.state = self.desired_state.copy()

    def reset(self, seed: int | None = None, init_noise: float | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noise_scale = float(self.init_noise if init_noise is None else init_noise)
        noise = rng.uniform(-noise_scale, noise_scale, size=4)
        # Keep angle relatively smaller than cart states so upright balancing is feasible.
        noise[2] *= 0.5
        self.state = self.desired_state + noise
        return self.state.copy()

    def clip_force(self, force: float) -> float:
        return float(np.clip(force, self.force_min, self.force_max))

    def step(self, force: float) -> tuple[np.ndarray, np.ndarray]:
        force = self.clip_force(force)
        self.state = self.ode.step(self.state, force, process_noise_std=self.process_noise_std)
        frame = self.renderer.render(self.state)
        return self.state.copy(), frame

    def render(self, state: np.ndarray | None = None) -> np.ndarray:
        return self.renderer.render(self.state if state is None else state)

    def control_score(self, state: np.ndarray | None = None, position_tol: float = 0.05, angle_tol: float = 0.05) -> int:
        s = self.state if state is None else state
        x_ok = abs(float(s[0] - self.desired_state[0])) <= position_tol
        theta_ok = abs(float(s[2])) <= angle_tol
        return int(x_ok and theta_ok)

    def rollout(self, teacher: ControlPolicy, steps: int, seed: int) -> dict[str, np.ndarray]:
        """
        Roll out `steps` transitions with a control policy.

        Plan §4.1 per step k:
          observe state_k → render frame_k → teacher.act(state_k) → u*_k → step → state_{k+1}
        """
        state = self.reset(seed=seed)
        frames = []
        commands = []
        states = []
        for _ in range(steps):
            force = self.clip_force(float(teacher.act(state)))
            frames.append(self.render(state))
            commands.append(force)
            states.append(state.copy())
            state, _ = self.step(force)
        return {
            "frames": np.stack(frames, axis=0).astype(np.uint8),
            "commands": np.asarray(commands, dtype=np.float32),
            "states": np.stack(states, axis=0).astype(np.float64),
        }
