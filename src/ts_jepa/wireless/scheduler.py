from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ts_jepa.wireless.channel import ChannelSample, WirelessChannelModel


SchedulerName = Literal["channel_aware", "round_robin", "opportunistic"]


@dataclass
class DeviceNetState:
    aoi: float = 1.0  # β_i,0 = 1 (Algorithm 2)
    virtual_queue: float = 0.0  # Q_i,0 = 0
    last_alpha: int = 0
    last_power: float = 0.0


@dataclass
class ScheduleDecision:
    alphas: dict[int, int]
    powers: dict[int, float]
    indices: dict[int, float] = field(default_factory=dict)
    costs: dict[int, float] = field(default_factory=dict)


class ChannelAwareScheduler:
    """
    Channel-aware drift-plus-penalty scheduler (paper Algorithm 2; eqs. 17, 18, 23–25).

    PAPER-SPECIFIED flow:
      observe H, β → p_req (23) → infeasible if > p_max → index → Top-J with S>0
      AoI (17): β←1 if scheduled else β←β+1
      Virtual queue (18): Q ← max(Q - β_th, 0) + β

    Index selection note:
      Eq. (24) minimizes sum_i α_i * C_i with
        C_i = 1 - (β+1)^2 - 2 Q β + V p_req   (same expression as typeset eq. 25)
      Algorithm 2 selects the largest positive indices. To reconcile minimization
      with that selection rule, we use urgency U_i = -C_i and select Top-J with U_i > 0.
      This is documented in IMPLEMENTATION_CHOICES.md.

    Unspecified numerical values (V, β_th, p_max, J, I) remain IMPLEMENTATION CHOICE.
    """

    def __init__(self, config: dict[str, Any], policy: SchedulerName = "channel_aware") -> None:
        w = config["wireless"]
        self.num_devices = int(w["num_devices"])
        self.J = int(w["max_devices_scheduled_J"])
        self.V = float(w["drift_plus_penalty_V"])
        self.beta_th = float(w["aoi_threshold_beta_th"])
        self.p_max = float(w["p_max_watt"])
        self.snr_target_db = float(w["snr_targets_db"][0])
        self.policy = policy
        self.channel = WirelessChannelModel(config)
        self.states = {i: DeviceNetState() for i in range(self.num_devices)}
        self._rr_cursor = 0

    def set_snr_target(self, snr_db: float) -> None:
        self.snr_target_db = float(snr_db)

    def drift_plus_penalty_cost(self, aoi: float, queue: float, p_req: float) -> float:
        """Per-device coefficient C_i in paper eqs. (24)–(25) as typeset."""
        return 1.0 - (aoi + 1.0) ** 2 - 2.0 * queue * aoi + self.V * p_req

    def urgency_index(self, aoi: float, queue: float, p_req: float) -> float:
        """U_i = -C_i so Algorithm 2 'largest positive' matches minimizing sum α C."""
        return -self.drift_plus_penalty_cost(aoi, queue, p_req)

    def schedule(self, rng: np.random.Generator | None = None) -> ScheduleDecision:
        rng = rng or np.random.default_rng()
        samples: dict[int, ChannelSample] = {
            i: self.channel.sample(i, self.snr_target_db, rng) for i in range(self.num_devices)
        }
        alphas = {i: 0 for i in range(self.num_devices)}
        powers = {i: 0.0 for i in range(self.num_devices)}
        indices = {i: float("-inf") for i in range(self.num_devices)}
        costs = {i: float("inf") for i in range(self.num_devices)}

        if self.policy == "round_robin":
            order = [(self._rr_cursor + i) % self.num_devices for i in range(self.num_devices)]
            self._rr_cursor = (self._rr_cursor + self.J) % self.num_devices
            chosen = []
            for i in order:
                if samples[i].feasible:
                    chosen.append(i)
                if len(chosen) >= self.J:
                    break
        elif self.policy == "opportunistic":
            ranked = sorted(
                [i for i in range(self.num_devices) if samples[i].feasible],
                key=lambda i: samples[i].h_complex_power,
                reverse=True,
            )
            chosen = ranked[: self.J]
        else:
            scored = []
            for i, st in self.states.items():
                sample = samples[i]
                if not sample.feasible:
                    indices[i] = float("-inf")
                    costs[i] = float("inf")
                    continue
                cost = self.drift_plus_penalty_cost(st.aoi, st.virtual_queue, sample.p_req)
                urgency = -cost
                costs[i] = cost
                indices[i] = urgency
                if urgency > 0.0:
                    scored.append((urgency, i))
            scored.sort(reverse=True)
            chosen = [i for _, i in scored[: self.J]]

        for i in chosen:
            alphas[i] = 1
            powers[i] = samples[i].p_req

        # Paper eqs. (17) and (18): update AoI then virtual queues using current β_i,k.
        for i, st in self.states.items():
            beta_k = st.aoi
            if alphas[i] == 1:
                st.aoi = 1.0
            else:
                st.aoi = beta_k + 1.0
            st.virtual_queue = max(st.virtual_queue - self.beta_th, 0.0) + beta_k
            st.last_alpha = alphas[i]
            st.last_power = powers[i]

        return ScheduleDecision(alphas=alphas, powers=powers, indices=indices, costs=costs)
