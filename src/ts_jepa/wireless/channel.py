from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ChannelSample:
    device_id: int
    h_gain: float
    snr_db: float
    p_req: float
    feasible: bool


class WirelessChannelModel:
    """
    Simplified indoor wireless channel using Table IV geometry/parameters.

    Exact fading law beyond Table IV scalars is IMPLEMENTATION CHOICE.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        w = config["wireless"]
        self.carrier_hz = float(w["carrier_frequency_hz"])
        self.bandwidth_hz = float(w["total_bandwidth_hz"])
        self.bs_height = float(w["bs_height_m"])
        self.device_height = float(w["device_height_m"])
        self.distance_2d = float(w["distance_2d_m"])
        self.noise_power_db = float(w["noise_power_db"])
        self.noise_watt = 10 ** (self.noise_power_db / 10.0)
        self.p_max = float(w["p_max_watt"])
        self.snr_targets_db = [float(x) for x in w["snr_targets_db"]]
        self.clutter_density = float(w["clutter_density"])
        # 3D distance BS → device
        dh = self.bs_height - self.device_height
        self.distance_3d = float(np.sqrt(self.distance_2d**2 + dh**2))
        c = 299_792_458.0
        wavelength = c / self.carrier_hz
        # Free-space path loss with clutter attenuation factor (IC).
        fspl = (wavelength / (4.0 * np.pi * self.distance_3d)) ** 2
        clutter_att = 10 ** (-3.0 * self.clutter_density)  # IC mapping
        self.mean_gain = float(fspl * clutter_att)

    def sample(self, device_id: int, snr_target_db: float, rng: np.random.Generator) -> ChannelSample:
        # Rayleigh-like small-scale fading around mean gain (IC).
        fading = float(rng.rayleigh(scale=1.0 / np.sqrt(2.0)))
        h_gain = max(self.mean_gain * fading * fading, 1e-18)
        gamma = 10 ** (snr_target_db / 10.0)
        # p_req such that SNR = p * |h|^2 / N0 >= gamma
        p_req = gamma * self.noise_watt / h_gain
        feasible = p_req <= self.p_max
        snr_db = 10.0 * np.log10(max(p_req * h_gain / self.noise_watt, 1e-18))
        return ChannelSample(
            device_id=device_id,
            h_gain=h_gain,
            snr_db=float(snr_db),
            p_req=float(p_req),
            feasible=bool(feasible),
        )
