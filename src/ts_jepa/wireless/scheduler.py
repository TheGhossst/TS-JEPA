from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ts_jepa.wireless.channel import ChannelSample, WirelessChannelModel


SchedulerName = Literal["channel_aware", "round_robin", "opportunistic"]


@dataclass
class DeviceNetState:
    aoi: float = 1.0  # β_{i,0} = 1 (paper Eq. 17 / Algorithm 2)
    virtual_queue: float = 0.0  # Q_{i,0} = 0 (paper Eq. 18)
    last_alpha: int = 0
    last_power: float = 0.0


@dataclass
class ScheduleDecision:
    alphas: dict[int, int]
    powers: dict[int, float]
    successes: dict[int, int] = field(default_factory=dict)
    outages: dict[int, int] = field(default_factory=dict)
    realized_snr_db: dict[int, float] = field(default_factory=dict)
    indices: dict[int, float] = field(default_factory=dict)
    costs: dict[int, float] = field(default_factory=dict)


class ChannelAwareScheduler:
    """
    Channel-aware scheduler (plan §13; paper Algorithm 2; eqs. 17, 18, 23–25).

    PAPER-SPECIFIED:
      observe H, β → p_req (23)
      if p_req > p_max: S = −∞ (infeasible)
      else S_{i,k} from eq. (25); schedule up to J devices with largest positive S
      scheduled devices transmit at p_{i,k} = p_req
      AoI (Eq. 17): β←1 if α=1, else β←β+1  (slot index, not τ_o)
      Virtual queue (Eq. 18): Q ← max(Q − β_th, 0) + β_k
      Lyapunov constant B in eq. (22) is omitted (paper: does not affect performance)

    Eq. (24) minimizes Σ α_i C_i with
      C_i = 1 − (β+1)^2 − 2 Q β + V p_req
    Algorithm 2 selects largest positive indices. Eq. (25) is S_i = −C_i, so
    Top-J with S_i > 0 implements both. V, β_th, p_max, J, I are IMPLEMENTATION CHOICE.

    Packet delivery (γ ≥ γ_th) is separate from AoI: it decides whether the
    embedding arrives at the controller, not β.

    Round-robin / opportunistic are paper baselines (plan §16), not Algorithm 2.
    The scheduler is not an input to the predictor.
    """

    def __init__(self, config: dict[str, Any], policy: SchedulerName = "channel_aware") -> None:
        w = config["wireless"]
        self.num_devices = int(w["num_devices"])  # I — IMPLEMENTATION CHOICE
        self.J = int(w["max_devices_scheduled_J"])  # IMPLEMENTATION CHOICE
        self.V = float(w["drift_plus_penalty_V"])  # IMPLEMENTATION CHOICE
        self.beta_th = float(w["aoi_threshold_beta_th"])  # IMPLEMENTATION CHOICE
        self.p_max = float(w["p_max_watt"])  # IMPLEMENTATION CHOICE
        self.snr_target_db = float(w["snr_targets_db"][0])
        self.policy = policy
        self.channel = WirelessChannelModel(config)
        self.states = {i: DeviceNetState() for i in range(self.num_devices)}
        self._rr_cursor = 0

    def set_snr_target(self, snr_db: float) -> None:
        self.snr_target_db = float(snr_db)

    def drift_plus_penalty_cost(self, aoi: float, queue: float, p_req: float) -> float:
        """Per-device coefficient C_i in paper eq. (24). Lyapunov B omitted."""
        return 1.0 - (aoi + 1.0) ** 2 - 2.0 * queue * aoi + self.V * p_req

    def drift_plus_penalty_index(self, aoi: float, queue: float, p_req: float) -> float:
        """Paper eq. (25) index S_{i,k} = −C_i (Algorithm 2: largest positive S)."""
        return -self.drift_plus_penalty_cost(aoi, queue, p_req)

    def urgency_index(self, aoi: float, queue: float, p_req: float) -> float:
        """Alias for eq. (25) S_{i,k}."""
        return self.drift_plus_penalty_index(aoi, queue, p_req)

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
                index_s = -cost
                costs[i] = cost
                indices[i] = index_s
                if index_s > 0.0:
                    scored.append((index_s, i))
            scored.sort(reverse=True)
            chosen = [i for _, i in scored[: self.J]]

        for i in chosen:
            alphas[i] = 1
            powers[i] = samples[i].p_req  # Algorithm 2: p_{i,k} = p^{req}_{i,k}

        successes = {i: 0 for i in range(self.num_devices)}
        outages = {i: 0 for i in range(self.num_devices)}
        realized_snr_db = {i: float("-inf") for i in range(self.num_devices)}
        for i in range(self.num_devices):
            if alphas[i] != 1:
                continue
            sample = samples[i]
            snr_db = self.channel.snr_db(powers[i], sample.h_complex_power, sample.path_loss_db)
            realized_snr_db[i] = snr_db
            if self.channel.is_outage(snr_db, self.snr_target_db):
                outages[i] = 1
                successes[i] = 0
            else:
                successes[i] = 1

        # Plan §13 Eq. (17): β_{k+1}=1 if α_k=1, else 1+β_k. Eq. (18) uses β_k.
        for i, st in self.states.items():
            beta_k = st.aoi
            if alphas[i] == 1:
                st.aoi = 1.0
            else:
                st.aoi = beta_k + 1.0
            st.virtual_queue = max(st.virtual_queue - self.beta_th, 0.0) + beta_k
            st.last_alpha = alphas[i]
            st.last_power = powers[i]

        return ScheduleDecision(
            alphas=alphas,
            powers=powers,
            successes=successes,
            outages=outages,
            realized_snr_db=realized_snr_db,
            indices=indices,
            costs=costs,
        )
