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
    Nonlinear DP teacher on a discretized 4D cart-pole grid.

    Method is paper-specified (nonlinear DP). Eq. 2 is subject to Eq. 1, whose
    index k has period τ_o; with env.ode.dt = τ_o this means one Bellman
    transition = one physics step (dp_substeps=1).

    Continuation uses a nodal first-order Taylor expansion of V (IC):
      V(s') ≈ V(s_g) + ∇V(s_g)·(s'−s_g)
    with central-difference ∇V. Nearest-neighbor lookup collapses every 1 ms
    successor into one cell. Multilinear interpolation still fails: V is convex,
    the velocity axes include a node at 0, and piecewise-linear V has a kink so
    any 1 ms motion looks costly and u=0 wins. The Taylor form recovers the
    mixed partial ∂²V/∂θ∂θ̇ that prefers restoring forces at 1 ms.

    Remaining IMPLEMENTATION CHOICES (paper-silent):
      - integrated successor state cost + control cost charged once per DP decision
      - R, finite horizon K (VI iters), grid resolution, force bins
      - nodal Taylor continuation of V at continuous 1 ms successors
      - undiscounted backups (Eq. 3 has no γ; γ=0.99 per 1 ms is unusable)
    """

    def __init__(
        self,
        env: InvertedCartPoleEnv,
        grid: dict[str, Any],
        force_min: float = -20.0,
        force_max: float = 20.0,
        force_bins: int = 11,
        control_effort_weight: float = 0.001,
        discount: float = 1.0,
        value_iteration_iters: int = 300,
        desired_state: list[float] | None = None,
        dp_substeps: int = 1,
    ) -> None:
        self.env = env
        self.grid = DPGrid.from_config(grid)
        self.forces = np.linspace(force_min, force_max, int(force_bins), dtype=np.float64)
        self.control_effort_weight = float(control_effort_weight)
        self.discount = float(discount)
        self.value_iteration_iters = int(value_iteration_iters)
        self.dp_substeps = int(dp_substeps)
        if self.dp_substeps != 1:
            raise ValueError(
                f"dp_substeps must be 1 so one Bellman transition equals one τ_o "
                f"physics step (Eq. 2 subject to Eq. 1); got {self.dp_substeps}"
            )
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

    @property
    def effective_dp_dt(self) -> float:
        return float(self.dp_substeps) * float(self.env.ode.dt)

    def _index_of(self, state: np.ndarray) -> tuple[int, int, int, int]:
        ix = int(np.argmin(np.abs(self.grid.x - state[0])))
        ixd = int(np.argmin(np.abs(self.grid.x_dot - state[1])))
        ith = int(np.argmin(np.abs(self.grid.theta - state[2])))
        ithd = int(np.argmin(np.abs(self.grid.theta_dot - state[3])))
        return ix, ixd, ith, ithd

    def _lookup_value(self, state: np.ndarray) -> float:
        return float(self._interpolated_value(np.asarray(state, dtype=np.float64).reshape(1, 4), self.value)[0])

    def _quantize_indices(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Nearest-neighbor grid indices for states shaped [..., 4] (table lookup only)."""
        flat = states.reshape(-1, 4)

        def _nearest(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
            ix = np.clip(np.searchsorted(axis, values), 0, len(axis) - 1)
            left = np.maximum(ix - 1, 0)
            choose_left = np.abs(axis[ix] - values) > np.abs(axis[left] - values)
            return np.where(choose_left, left, ix)

        ix = _nearest(self.grid.x, flat[:, 0])
        ixd = _nearest(self.grid.x_dot, flat[:, 1])
        ith = _nearest(self.grid.theta, flat[:, 2])
        ithd = _nearest(self.grid.theta_dot, flat[:, 3])
        shape = states.shape[:-1]
        return ix.reshape(shape), ixd.reshape(shape), ith.reshape(shape), ithd.reshape(shape)

    def _nearest_value(self, states: np.ndarray, value: np.ndarray) -> np.ndarray:
        ix, ixd, ith, ithd = self._quantize_indices(states)
        return value[ix, ixd, ith, ithd]

    def _value_gradients(
        self, value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gx, gxd, gth, gthd = np.gradient(
            value,
            self.grid.x,
            self.grid.x_dot,
            self.grid.theta,
            self.grid.theta_dot,
            edge_order=1,
        )
        return gx, gxd, gth, gthd

    def _interpolated_value(
        self,
        states: np.ndarray,
        value: np.ndarray,
        grads: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """
        Nodal Taylor continuation: V(s') ≈ V(s_g) + ∇V(s_g)·(s'−s_g).

        s_g is the nearest grid node. ∇V is central difference on the table.
        Exact at grid nodes. Resolves 1 ms successor differences that neither
        nearest-neighbor nor multilinear interpolation can see.
        """
        states = np.asarray(states, dtype=np.float64)
        ix, ixd, ith, ithd = self._quantize_indices(states)
        if grads is None:
            grads = self._value_gradients(value)
        gx, gxd, gth, gthd = grads
        node = np.stack(
            [
                self.grid.x[ix],
                self.grid.x_dot[ixd],
                self.grid.theta[ith],
                self.grid.theta_dot[ithd],
            ],
            axis=-1,
        )
        delta = states - node
        return (
            value[ix, ixd, ith, ithd]
            + gx[ix, ixd, ith, ithd] * delta[..., 0]
            + gxd[ix, ixd, ith, ithd] * delta[..., 1]
            + gth[ix, ixd, ith, ithd] * delta[..., 2]
            + gthd[ix, ixd, ith, ithd] * delta[..., 3]
        )

    def _rollout_constant_force(
        self, states: np.ndarray, force: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Hold `force` for one τ_o physics step (dp_substeps=1).

        Stage cost (IC numerics; one-step structure follows Eq. 3's per-k term):
          0.5 ||s_{k+1} - s*||^2   +   0.5 R u^2
        Control cost is charged once per DP decision.
        """
        single = states.ndim == 1
        s = states.reshape(-1, 4).astype(np.float64)
        u = float(force)
        control_term = 0.5 * self.control_effort_weight * (u * u)
        stage = np.zeros(s.shape[0], dtype=np.float64)
        cur = s
        for _ in range(self.dp_substeps):
            cur = self.env.ode.step(cur, u, process_noise_std=0.0)
            err = cur - self.desired_state.reshape(1, 4)
            stage = stage + 0.5 * np.sum(err * err, axis=-1)
        stage = stage + control_term
        final = cur
        if single:
            return final[0], float(stage[0])
        return final.reshape(states.shape), stage.reshape(states.shape[:-1])

    def _bellman_cost(self, state: np.ndarray, force: float, value: np.ndarray | None = None) -> float:
        """Same one-transition cost used in VI and act()."""
        value = self.value if value is None else value
        final, stage = self._rollout_constant_force(state, float(force))
        assert isinstance(stage, float)
        return float(stage + self.discount * self._interpolated_value(final.reshape(1, 4), value)[0])

    def _solve(self) -> None:
        states = np.stack([self._xx, self._xd, self._th, self._thd], axis=-1)
        value = np.zeros(self.grid.shape, dtype=np.float64)
        policy = np.zeros(self.grid.shape, dtype=np.float64)

        # Precompute N-step successors + integrated stage costs for each discrete force.
        transitions: list[np.ndarray] = []
        stage_costs: list[np.ndarray] = []
        for force in self.forces:
            nxt, stage = self._rollout_constant_force(states, float(force))
            transitions.append(nxt)
            stage_costs.append(stage)

        for _ in range(self.value_iteration_iters):
            grads = self._value_gradients(value)
            best_cost = np.full(self.grid.shape, np.inf, dtype=np.float64)
            best_force = np.zeros(self.grid.shape, dtype=np.float64)
            for force, nxt, stage in zip(self.forces, transitions, stage_costs):
                total = stage + self.discount * self._interpolated_value(nxt, value, grads)
                # Prefer improvement; on ties keep smaller |u| (then existing).
                improve = total < best_cost - 1e-12
                tie = np.abs(total - best_cost) <= 1e-12
                prefer_tie = tie & (np.abs(force) < np.abs(best_force))
                take = improve | prefer_tie
                best_cost = np.where(take, total, best_cost)
                best_force = np.where(take, force, best_force)
            value = best_cost
            policy = best_force

        self.value = value
        self.policy = policy

    def act(self, state: np.ndarray) -> float:
        """Local discrete re-optimization with the same one-step Bellman cost as VI."""
        state = np.asarray(state, dtype=np.float64).reshape(4)
        table_u = float(self.policy[self._index_of(state)])
        deltas = np.array([0.0, -4.0, 4.0, -8.0, 8.0], dtype=np.float64)
        local = np.clip(table_u + deltas, self.forces[0], self.forces[-1])
        # Also probe a coarse subset of the force table.
        coarse = self.forces[:: max(1, len(self.forces) // 5)]
        candidates = np.unique(np.concatenate([local, coarse]))

        best_u = table_u
        best_cost = self._bellman_cost(state, best_u)
        for candidate in candidates:
            cost = self._bellman_cost(state, float(candidate))
            if cost < best_cost - 1e-12 or (
                abs(cost - best_cost) <= 1e-12 and abs(candidate) < abs(best_u)
            ):
                best_cost = cost
                best_u = float(candidate)
        return float(best_u)
