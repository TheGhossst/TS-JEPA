"""Continuous-force CartPole environment."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class ContinuousCartPoleEnv(CartPoleEnv):
    """
    CartPole with a continuous action: applied cart force in Newtons.

    The action is a single float clipped to [-force_limit, force_limit] (default ±20 N).
    Observation space matches standard CartPole-v1:
        [cart position, cart velocity, pole angle, pole angular velocity]
    """

    metadata = CartPoleEnv.metadata

    def __init__(
        self,
        render_mode: str | None = None,
        force_limit: float = 20.0,
    ) -> None:
        super().__init__(render_mode=render_mode)
        self.force_limit = float(force_limit)
        self.action_space = spaces.Box(
            low=-self.force_limit,
            high=self.force_limit,
            shape=(1,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset with a wider initial-state distribution than the Gym default."""
        super().reset(seed=seed, options=options)
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action: float | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        err_msg = f"{action!r} invalid for action space {self.action_space!r}"
        if not self.action_space.contains(np.asarray(action, dtype=np.float32)):
            raise ValueError(err_msg)

        assert self.state is not None, "Call reset() before step()"

        x, x_dot, theta, theta_dot = self.state
        force = float(np.clip(np.asarray(action).reshape(-1)[0], -self.force_limit, self.force_limit))

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = -1.0 if self._sutton_barto_reward else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                gym.logger.warn(
                    "You are calling 'step()' even though this environment has already "
                    "returned terminated=True. Be sure to call 'reset()' if you want to "
                    "restart the episode."
                )
            self.steps_beyond_terminated += 1
            reward = -1.0 if self._sutton_barto_reward else 0.0

        if self.render_mode == "human":
            self.render()

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    @classmethod
    def register(cls, force_limit: float = 20.0) -> None:
        """Register this environment with Gymnasium under id ContinuousCartPole-v0."""
        gym.register(
            id="ContinuousCartPole-v0",
            entry_point="cartpole_pipeline.envs.continuous_cartpole:ContinuousCartPoleEnv",
            max_episode_steps=500,
            kwargs={"force_limit": force_limit},
        )
