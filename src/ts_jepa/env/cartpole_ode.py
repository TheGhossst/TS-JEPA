from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartPoleParams:
    """Nonlinear inverted cart-pole parameters (IMPLEMENTATION CHOICE)."""

    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_length: float = 0.5  # half-length to center of mass
    gravity: float = 9.81
    track_limit: float = 2.4


class CartPoleODE:
    """Custom nonlinear inverted cart-pole ODE with deterministic Ns=0 baseline."""

    def __init__(self, params: CartPoleParams | None = None, dt: float = 0.001) -> None:
        self.params = params or CartPoleParams()
        self.dt = float(dt)

    def derivatives(self, state: np.ndarray, force: float) -> np.ndarray:
        """
        State: [x, x_dot, theta, theta_dot]
        theta = 0 is upright; positive theta is counterclockwise.
        Supports a single state `(4,)` or a batch `(N, 4)`.
        """
        single = state.ndim == 1
        s = state.reshape(-1, 4)
        force_arr = np.full((s.shape[0],), float(np.clip(force, -20.0, 20.0)), dtype=np.float64)

        x_dot = s[:, 1]
        theta = s[:, 2]
        theta_dot = s[:, 3]
        m_c = self.params.cart_mass
        m_p = self.params.pole_mass
        length = self.params.pole_length
        g = self.params.gravity
        total_mass = m_c + m_p

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)

        temp = (force_arr + m_p * length * theta_dot**2 * sin_th) / total_mass
        theta_acc_den = length * (4.0 / 3.0 - (m_p * cos_th**2) / total_mass)
        theta_acc = (g * sin_th - cos_th * temp) / theta_acc_den
        x_acc = temp - (m_p * length * theta_acc * cos_th) / total_mass
        out = np.stack([x_dot, x_acc, theta_dot, theta_acc], axis=-1)
        return out[0] if single else out

    def step(self, state: np.ndarray, force: float, process_noise_std: float = 0.0) -> np.ndarray:
        """Semi-implicit Euler step. process_noise_std=0 for paper-faithful Ns=0 baseline."""
        single = state.ndim == 1
        s = state.reshape(-1, 4).astype(np.float64)
        force = float(np.clip(force, -20.0, 20.0))
        deriv = self.derivatives(s, force)
        next_state = s.copy()
        next_state[:, 1] = s[:, 1] + self.dt * deriv[:, 1]
        next_state[:, 3] = s[:, 3] + self.dt * deriv[:, 3]
        next_state[:, 0] = s[:, 0] + self.dt * next_state[:, 1]
        next_state[:, 2] = s[:, 2] + self.dt * next_state[:, 3]
        if process_noise_std > 0.0:
            next_state = next_state + np.random.normal(0.0, process_noise_std, size=next_state.shape)
        return next_state[0] if single else next_state
