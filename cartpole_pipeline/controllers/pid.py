"""PID controller for inverted CartPole balancing."""

from __future__ import annotations

import numpy as np


class PIDController:
    """
    Simple PID controller that balances the pole using angle and angular velocity.

    Uses CartPole observation indices 2 (pole angle) and 3 (pole angular velocity).
    Output is a continuous cart force in Newtons, clipped to [-force_limit, force_limit].
    """

    def __init__(
        self,
        kp: float = 50.0,
        ki: float = 0.0,
        kd: float = 12.0,
        force_limit: float = 20.0,
        target_angle: float = 0.0,
        angle_index: int = 2,
        angular_velocity_index: int = 3,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.force_limit = force_limit
        self.target_angle = target_angle
        self.angle_index = angle_index
        self.angular_velocity_index = angular_velocity_index
        self._integral = 0.0

    def reset(self) -> None:
        """Clear integral state at the start of a new episode."""
        self._integral = 0.0

    def __call__(self, state: np.ndarray, dt: float = 0.02) -> float:
        """
        Compute control force from the current environment state.

        Args:
            state: CartPole observation vector of length 4.
            dt: Timestep duration in seconds (CartPole-v1 uses 0.02 s).

        Returns:
            Force in Newtons, clipped to [-force_limit, force_limit].
        """
        theta = float(state[self.angle_index])
        theta_dot = float(state[self.angular_velocity_index])

        error = theta - self.target_angle
        self._integral += error * dt

        force = (
            self.kp * error
            + self.ki * self._integral
            + self.kd * theta_dot
        )
        return float(np.clip(force, -self.force_limit, self.force_limit))
