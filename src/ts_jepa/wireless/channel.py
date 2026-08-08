from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ChannelSample:
    device_id: int
    h_complex_power: float  # |H|^2 Rayleigh power gain
    path_loss_db: float
    p_req: float
    feasible: bool
    is_los: bool


class WirelessChannelModel:
    """
    InF-SH wireless channel from the published paper (eqs. 4–8, 23).

    PAPER-SPECIFIED:
      PL_LoS_dB = 31.84 + 21.5 log10(D3D) + 19 log10(Wc_GHz)
      P_LoS = exp( -((D2D - D_clutter) / ln(1-δ)) * ((hBS - h_iR)/(hc - h_iR)) )
      PL_dB = 33.63 + 21.9 log10(D3D) + 20 log10(Wc_GHz)  (shadow σ=4.0 dB)
      PL_NLoS_dB = max(PL_dB, PL_LoS_dB)
      γ = 10^(-PL/10) * P * |H|^2 / N_c
      p_req = γ_th * N_c / |H|^2 * 10^(PL/10)

    Table IV geometry/radio scalars are paper-specified.
    p_max remains IMPLEMENTATION CHOICE when not numerically stated in Final.md.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        w = config["wireless"]
        self.carrier_ghz = float(w["carrier_frequency_hz"]) / 1e9
        self.bs_height = float(w["bs_height_m"])
        self.device_height = float(w["device_height_m"])
        self.distance_2d = float(w["distance_2d_m"])
        self.clutter_size = float(w["clutter_size_m"])
        self.clutter_density = float(w["clutter_density"])
        self.clutter_height = float(w["clutter_height_m"])
        self.noise_power_db = float(w["noise_power_db"])
        self.noise_watt = 10 ** (self.noise_power_db / 10.0)
        self.p_max = float(w["p_max_watt"])  # IMPLEMENTATION CHOICE if not in paper tables
        self.shadow_sigma_db = float(w.get("shadow_fading_std_db", 4.0))  # paper-specified for PL_dB
        dh = self.bs_height - self.device_height
        self.distance_3d = float(np.sqrt(self.distance_2d**2 + dh**2))

    def path_loss_los_db(self) -> float:
        # Paper eq. (4); Wc in GHz.
        return 31.84 + 21.5 * np.log10(self.distance_3d) + 19.0 * np.log10(self.carrier_ghz)

    def path_loss_general_db(self, rng: np.random.Generator) -> float:
        # Paper eq. (7) + shadow fading std 4.0 dB.
        pl = 33.63 + 21.9 * np.log10(self.distance_3d) + 20.0 * np.log10(self.carrier_ghz)
        return float(pl + rng.normal(0.0, self.shadow_sigma_db))

    def los_probability(self) -> float:
        # Paper eq. (5) / InF-SH LoS probability (3GPP-style factorization).
        # P_LoS = exp( -(D2D - D_clutter) / k ),
        # k = -D_clutter / ln(1-δ) * (hBS - h_iR) / (hc - h_iR)
        if self.distance_2d <= self.clutter_size:
            return 1.0
        den_ln = np.log(1.0 - self.clutter_density)
        height_ratio = (self.bs_height - self.device_height) / (self.clutter_height - self.device_height)
        if den_ln >= 0 or height_ratio <= 0 or not np.isfinite(height_ratio):
            return 0.0
        k = -self.clutter_size / den_ln * height_ratio
        if k <= 0 or not np.isfinite(k):
            return 0.0
        return float(np.clip(np.exp(-(self.distance_2d - self.clutter_size) / k), 0.0, 1.0))

    def sample_path_loss_db(self, rng: np.random.Generator) -> tuple[float, bool]:
        p_los = self.los_probability()
        is_los = bool(rng.random() < p_los)
        pl_los = self.path_loss_los_db()
        if is_los:
            return float(pl_los), True
        pl_nlos = max(self.path_loss_general_db(rng), pl_los)  # paper eq. (6)
        return float(pl_nlos), False

    def required_power(self, snr_target_db: float, h_power: float, path_loss_db: float) -> float:
        # Paper eq. (23): p_req = γ_th * N_c / |H|^2 * 10^(PL/10)
        gamma = 10 ** (snr_target_db / 10.0)
        return float(gamma * self.noise_watt / max(h_power, 1e-18) * (10 ** (path_loss_db / 10.0)))

    def sample(self, device_id: int, snr_target_db: float, rng: np.random.Generator) -> ChannelSample:
        # Rayleigh flat-fading power gain |H|^2 ~ Exp(1) via |CN(0,1)|^2.
        h_real = rng.normal(0.0, 1.0 / np.sqrt(2.0))
        h_imag = rng.normal(0.0, 1.0 / np.sqrt(2.0))
        h_power = float(h_real * h_real + h_imag * h_imag)
        path_loss_db, is_los = self.sample_path_loss_db(rng)
        p_req = self.required_power(snr_target_db, h_power, path_loss_db)
        feasible = p_req <= self.p_max
        return ChannelSample(
            device_id=device_id,
            h_complex_power=h_power,
            path_loss_db=path_loss_db,
            p_req=p_req,
            feasible=feasible,
            is_los=is_los,
        )
