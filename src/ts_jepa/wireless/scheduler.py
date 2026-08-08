from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ts_jepa.wireless.channel import ChannelSample, WirelessChannelModel


SchedulerName = Literal["channel_aware", "round_robin", "opportunistic"]


@dataclass
class DeviceNetState:
    aoi: float = 1.0
    virtual_queue: float = 0.0
    last_alpha: int = 0
    last_power: float = 0.0


@dataclass
class ScheduleDecision:
    alphas: dict[int, int]
    powers: dict[int, float]
    indices: dict[int, float] = field(default_factory=dict)


class ChannelAwareScheduler:
    """
    Channel-aware drift-plus-penalty scheduler (networking layer).

    Flow matches paper-faithful architecture summary:
    observe H, AoI, virtual queue → p_req → feasibility → index S → select up to J.
    Missing closed-form constants (J, V, queue coeffs, p_max) are IMPLEMENTATION CHOICE.
    """

    def __init__(self, config: dict[str, Any], policy: SchedulerName = "channel_aware") -> None:
        w = config["wireless"]
        self.num_devices = int(w["num_devices"])
        self.J = int(w["max_devices_scheduled_J"])
        self.V = float(w["drift_plus_penalty_V"])
        self.arrival = float(w["virtual_queue_arrival"])
        self.p_max = float(w["p_max_watt"])
        self.snr_target_db = float(w["snr_targets_db"][0])
        self.policy = policy
        self.channel = WirelessChannelModel(config)
        self.states = {i: DeviceNetState() for i in range(self.num_devices)}
        self._rr_cursor = 0

    def set_snr_target(self, snr_db: float) -> None:
        self.snr_target_db = float(snr_db)

    def _index(self, aoi: float, queue: float, p_req: float) -> float:
        # Drift-plus-penalty style urgency index (IC functional form).
        return self.V * aoi + queue - p_req / max(self.p_max, 1e-12)

    def schedule(self, rng: np.random.Generator | None = None) -> ScheduleDecision:
        rng = rng or np.random.default_rng()
        samples: dict[int, ChannelSample] = {
            i: self.channel.sample(i, self.snr_target_db, rng) for i in range(self.num_devices)
        }
        alphas = {i: 0 for i in range(self.num_devices)}
        powers = {i: 0.0 for i in range(self.num_devices)}
        indices = {i: 0.0 for i in range(self.num_devices)}

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
                key=lambda i: samples[i].h_gain,
                reverse=True,
            )
            chosen = ranked[: self.J]
        else:
            scored = []
            for i, st in self.states.items():
                sample = samples[i]
                if not sample.feasible:
                    indices[i] = -np.inf
                    continue
                s = self._index(st.aoi, st.virtual_queue, sample.p_req)
                indices[i] = s
                if s > 0:
                    scored.append((s, i))
            scored.sort(reverse=True)
            chosen = [i for _, i in scored[: self.J]]

        for i in chosen:
            alphas[i] = 1
            powers[i] = samples[i].p_req

        # Update AoI and virtual queues.
        for i, st in self.states.items():
            if alphas[i] == 1:
                st.aoi = 1.0
            else:
                st.aoi += 1.0
            # Queue tracks unmet reliability / backlog (IC update).
            st.virtual_queue = max(0.0, st.virtual_queue + self.arrival - alphas[i])
            st.last_alpha = alphas[i]
            st.last_power = powers[i]

        return ScheduleDecision(alphas=alphas, powers=powers, indices=indices)
