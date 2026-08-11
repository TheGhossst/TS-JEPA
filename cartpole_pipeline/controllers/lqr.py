"""LQR controller for continuous CartPole balancing."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are


def _linearized_cartpole_matrices(
    gravity: float = 9.8,
    masscart: float = 1.0,
    masspole: float = 0.1,
    length: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuous-time linearization of Gym CartPole dynamics about upright.

    Uses the same parameters and inertia model as gymnasium's CartPoleEnv
    (pole half-length ``length``, moment factor 4/3).
    State: [x, x_dot, theta, theta_dot], input: cart force (N).
    """
    g = gravity
    M = masscart
    m = masspole
    l = length
    total = M + m
    denom = 4.0 * M + m

    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -(3.0 * m * g) / denom, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, (3.0 * total * g) / (l * denom), 0.0],
        ],
        dtype=np.float64,
    )
    B = np.array(
        [
            [0.0],
            [4.0 / denom],
            [0.0],
            [-3.0 / (l * denom)],
        ],
        dtype=np.float64,
    )
    return A, B


class LQRController:
    """
    Continuous-time LQR stabilizing the upright CartPole equilibrium.

    Solves the algebraic Riccati equation for gain K and applies u = -K x,
    with force clipped to [-force_limit, force_limit].
    """

    def __init__(
        self,
        q_diag: tuple[float, float, float, float] = (10.0, 1.0, 10.0, 1.0),
        r: float = 0.1,
        force_limit: float = 20.0,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
    ) -> None:
        self.force_limit = float(force_limit)
        self.Q = np.diag(np.asarray(q_diag, dtype=np.float64))
        self.R = np.array([[float(r)]], dtype=np.float64)

        A, B = _linearized_cartpole_matrices(
            gravity=gravity,
            masscart=masscart,
            masspole=masspole,
            length=length,
        )
        P = solve_continuous_are(A, B, self.Q, self.R)
        self.K = np.linalg.solve(self.R, B.T @ P)  # shape (1, 4)

    def reset(self) -> None:
        """No-op; LQR is memoryless. Kept for controller API compatibility."""

    def __call__(self, state: np.ndarray) -> float:
        """
        Compute control force from the current environment state.

        Args:
            state: CartPole observation vector of length 4.

        Returns:
            Force in Newtons, clipped to [-force_limit, force_limit].
        """
        x = np.asarray(state, dtype=np.float64).reshape(4)
        force = float(-(self.K @ x).item())
        return float(np.clip(force, -self.force_limit, self.force_limit))
