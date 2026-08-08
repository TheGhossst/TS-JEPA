from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv


@dataclass
class DPGrid:
    x: np.ndarray
    x_dot: np.ndarray
    theta: np.ndarray
    theta_dot: np.ndarray

    @classmethod
    def from_config(cls, grid: dict[str, Any]) -> "DPGrid":
        return cls(
            x=np.linspace(grid["x"][0], grid["x"][1], int(grid["x"][2])),
            x_dot=np.linspace(grid["x_dot"][0], grid["x_dot"][1], int(grid["x_dot"][2])),
            theta=np.linspace(grid["theta"][0], grid["theta"][1], int(grid["theta"][2])),
            theta_dot=np.linspace(grid["theta_dot"][0], grid["theta_dot"][1], int(grid["theta_dot"][2])),
        )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (len(self.x), len(self.x_dot), len(self.theta), len(self.theta_dot))


class DPControlTeacher:
    """
    Finite-horizon / discounted dynamic-programming teacher on a discretized grid.

    Method is paper-specified (nonlinear DP). Grids, R, discount, and iteration
    count are IMPLEMENTATION CHOICES.
    """

    def __init__(
        self,
        env: InvertedCartPoleEnv,
        grid: dict[str, Any],
        force_min: float = -20.0,
        force_max: float = 20.0,
        force_bins: int = 21,
        control_effort_weight: float = 0.001,
        discount: float = 0.99,
        value_iteration_iters: int = 40,
        desired_state: list[float] | None = None,
    ) -> None:
        self.env = env
        self.grid = DPGrid.from_config(grid)
        self.forces = np.linspace(force_min, force_max, int(force_bins), dtype=np.float64)
        self.control_effort_weight = float(control_effort_weight)
        self.discount = float(discount)
        self.value_iteration_iters = int(value_iteration_iters)
        self.desired_state = np.asarray(
            desired_state if desired_state is not None else [0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        self.value = np.zeros(self.grid.shape, dtype=np.float64)
        self.policy = np.zeros(self.grid.shape, dtype=np.float64)
        self._xx, self._xd, self._th, self._thd = np.meshgrid(
            self.grid.x, self.grid.x_dot, self.grid.theta, self.grid.theta_dot, indexing="ij"
        )
        self._solve()

    def _state_cost_batch(self, states: np.ndarray, force: float) -> np.ndarray:
        err = states - self.desired_state.reshape(1, 1, 1, 1, 4)
        state_term = 0.5 * np.sum(err * err, axis=-1)
        return state_term + 0.5 * self.control_effort_weight * (force * force)

    def _index_of(self, state: np.ndarray) -> tuple[int, int, int, int]:
        ix = int(np.argmin(np.abs(self.grid.x - state[0])))
        ixd = int(np.argmin(np.abs(self.grid.x_dot - state[1])))
        ith = int(np.argmin(np.abs(self.grid.theta - state[2])))
        ithd = int(np.argmin(np.abs(self.grid.theta_dot - state[3])))
        return ix, ixd, ith, ithd

    def _lookup_value(self, state: np.ndarray) -> float:
        return float(self.value[self._index_of(state)])

    def _step_batch(self, states: np.ndarray, force: float) -> np.ndarray:
        """Vectorized ODE step over a grid of states shaped [..., 4]."""
        flat = states.reshape(-1, 4)
        next_states = self.env.ode.step(flat, force, process_noise_std=0.0)
        return next_states.reshape(states.shape)

    def _nearest_value(self, states: np.ndarray, value: np.ndarray) -> np.ndarray:
        # Quantize continuous next-states onto the DP grid via searchsorted.
        flat = states.reshape(-1, 4)
        ix = np.clip(np.searchsorted(self.grid.x, flat[:, 0]), 0, len(self.grid.x) - 1)
        # Choose nearer neighbor.
        ix = np.where(
            (ix > 0) & (np.abs(self.grid.x[ix] - flat[:, 0]) > np.abs(self.grid.x[ix - 1] - flat[:, 0])),
            ix - 1,
            ix,
        )
        ixd = np.clip(np.searchsorted(self.grid.x_dot, flat[:, 1]), 0, len(self.grid.x_dot) - 1)
        ixd = np.where(
            (ixd > 0)
            & (np.abs(self.grid.x_dot[ixd] - flat[:, 1]) > np.abs(self.grid.x_dot[ixd - 1] - flat[:, 1])),
            ixd - 1,
            ixd,
        )
        ith = np.clip(np.searchsorted(self.grid.theta, flat[:, 2]), 0, len(self.grid.theta) - 1)
        ith = np.where(
            (ith > 0)
            & (np.abs(self.grid.theta[ith] - flat[:, 2]) > np.abs(self.grid.theta[ith - 1] - flat[:, 2])),
            ith - 1,
            ith,
        )
        ithd = np.clip(np.searchsorted(self.grid.theta_dot, flat[:, 3]), 0, len(self.grid.theta_dot) - 1)
        ithd = np.where(
            (ithd > 0)
            & (
                np.abs(self.grid.theta_dot[ithd] - flat[:, 3])
                > np.abs(self.grid.theta_dot[ithd - 1] - flat[:, 3])
            ),
            ithd - 1,
            ithd,
        )
        out = value[ix, ixd, ith, ithd]
        return out.reshape(states.shape[:-1])

    def _solve(self) -> None:
        states = np.stack([self._xx, self._xd, self._th, self._thd], axis=-1)
        value = np.zeros(self.grid.shape, dtype=np.float64)
        policy = np.zeros(self.grid.shape, dtype=np.float64)

        # Precompute transitions for each discrete force.
        transitions = []
        stage_costs = []
        for force in self.forces:
            nxt = self._step_batch(states, float(force))
            transitions.append(nxt)
            stage_costs.append(self._state_cost_batch(states, float(force)))

        for _ in range(self.value_iteration_iters):
            best_cost = np.full(self.grid.shape, np.inf, dtype=np.float64)
            best_force = np.zeros(self.grid.shape, dtype=np.float64)
            for force, nxt, stage in zip(self.forces, transitions, stage_costs):
                cont = self.discount * self._nearest_value(nxt, value)
                total = stage + cont
                improve = total < best_cost
                best_cost = np.where(improve, total, best_cost)
                best_force = np.where(improve, force, best_force)
            value = best_cost
            policy = best_force

        self.value = value
        self.policy = policy

    def act(self, state: np.ndarray) -> float:
        force = float(self.policy[self._index_of(state)])
        best = force
        best_cost = np.inf
        candidates = np.unique(
            np.clip(
                np.array(
                    [force - 2.0, force, force + 2.0, *self.forces[:: max(1, len(self.forces) // 7)]],
                    dtype=np.float64,
                ),
                self.forces[0],
                self.forces[-1],
            )
        )
        for candidate in candidates:
            nxt = self.env.ode.step(state, float(candidate), process_noise_std=0.0)
            total = 0.5 * (
                float(np.sum((state - self.desired_state) ** 2))
                + self.control_effort_weight * float(candidate * candidate)
            ) + self.discount * self._lookup_value(nxt)
            if total < best_cost:
                best_cost = total
                best = float(candidate)
        return best
