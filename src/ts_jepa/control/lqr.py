"""Discrete LQR for the inverted cart-pole at the stored control period."""

from __future__ import annotations

from typing import Any

import numpy as np

from ts_jepa.env.cartpole_ode import CartPoleODE
from ts_jepa.env.factory import build_inverted_cartpole_env
from ts_jepa.evaluation.evaluate import control_loop_stride


def _rollout_force(ode: CartPoleODE, state: np.ndarray, force: float, stride: int) -> np.ndarray:
    s = np.asarray(state, dtype=np.float64).reshape(4)
    for _ in range(int(stride)):
        s = ode.step(s, float(force), process_noise_std=0.0)
    return s


def discrete_linearization(
    ode: CartPoleODE,
    stride: int,
    *,
    eps_state: float = 1e-5,
    eps_force: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference A, B of one held-force observation step about upright."""
    s0 = np.zeros(4, dtype=np.float64)
    a = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        d = np.zeros(4, dtype=np.float64)
        d[i] = eps_state
        sp = _rollout_force(ode, s0 + d, 0.0, stride)
        sm = _rollout_force(ode, s0 - d, 0.0, stride)
        a[:, i] = (sp - sm) / (2.0 * eps_state)
    bp = _rollout_force(ode, s0, eps_force, stride)
    bm = _rollout_force(ode, s0, -eps_force, stride)
    b = ((bp - bm) / (2.0 * eps_force)).reshape(4, 1)
    return a, b


def discrete_lqr_gain(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    *,
    iters: int = 400,
    tol: float = 1e-10,
) -> np.ndarray:
    """Iterate the discrete Riccati equation; returns K with u = -K x."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(4, 1)
    q = np.asarray(q, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64).reshape(1, 1)
    p = q.copy()
    k = np.zeros((1, 4), dtype=np.float64)
    for _ in range(int(iters)):
        bt_p_b = float((b.T @ p @ b + r)[0, 0])
        if abs(bt_p_b) < 1e-12:
            break
        k_new = (b.T @ p @ a) / bt_p_b
        a_cl = a - b @ k_new
        p_new = q + a.T @ p @ a_cl
        if float(np.max(np.abs(k_new - k))) < tol and float(np.max(np.abs(p_new - p))) < tol:
            k = k_new
            p = p_new
            break
        k = k_new
        p = p_new
    return k.reshape(1, 4)


def default_lqr_weights() -> tuple[np.ndarray, np.ndarray]:
    # Heavier on x and theta so the cart stays in the Eq. 28 band.
    q = np.diag([40.0, 4.0, 80.0, 4.0]).astype(np.float64)
    r = np.array([[0.05]], dtype=np.float64)
    return q, r


def lqr_gain_from_config(config: dict[str, Any], *, stride: int | None = None) -> np.ndarray:
    env = build_inverted_cartpole_env(config)
    hold = int(stride) if stride is not None else control_loop_stride(config)
    a, b = discrete_linearization(env.ode, hold)
    q, r = default_lqr_weights()
    return discrete_lqr_gain(a, b, q, r)


def lqr_force(gain: np.ndarray, state: np.ndarray, force_min: float, force_max: float) -> float:
    x = np.asarray(state, dtype=np.float64).reshape(4)
    u = float((-np.asarray(gain, dtype=np.float64).reshape(1, 4) @ x).reshape(-1)[0])
    return float(np.clip(u, force_min, force_max))


def lqr_forces(gain: np.ndarray, states: np.ndarray, force_min: float, force_max: float) -> np.ndarray:
    x = np.asarray(states, dtype=np.float64).reshape(-1, 4)
    k = np.asarray(gain, dtype=np.float64).reshape(1, 4)
    u = -(x @ k.T).reshape(-1)
    return np.clip(u, force_min, force_max).astype(np.float64)
