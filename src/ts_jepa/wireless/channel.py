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
    InF-SH wireless channel (plan §13; paper eqs. 4–10, 23).

    PAPER-SPECIFIED:
      PL_LoS_dB = 31.84 + 21.5 log10(D^{3D}) + 19 log10(W_c)     (Eq. 4)
      P_LoS = exp( - D^{2D} / k ),  k = -D_clutter/ln(1-δ)
              * (h_BS - h_{i,R}) / (h_c - h_{i,R})               (Eq. 5 / InF-SH)
      PL_NLoS_dB = max(PL_dB, PL_LoS_dB)                         (Eq. 6)
      PL_dB = 33.63 + 21.9 log10(D^{3D}) + 20 log10(W_c)
              + shadow N(0, 4.0^2) dB                            (Eq. 7)
      γ = 10^(-PL/10) * P * |H|^2 / N_c                          (Eq. 8)
      R = W_i log2(1+γ)                                          (Eq. 9)
      p_req = γ_th * N_c / |H|^2 * 10^(PL/10)                    (Eq. 23)

    Rayleigh |H|^2 is block-fading: constant over τ_o, independent across slots.
    Carrier symbol is W_c (GHz), not f_c.

    IMPLEMENTATION CHOICE:
      D^{3D} = hypot(D^{2D}, h_BS - h_{i,R}) — Table IV gives D^{2D} and heights.
      W_i = total bandwidth 20 MHz (per-device split is NOT SPECIFIED).
      N_c linear = 10^(N_c,dB/10) with N_c = −95 dB as Table IV (not converted from dBm).
      Eq. 10 rate threshold R̄ is NOT SPECIFIED; slot outage uses γ < γ_th, which is
      Eq. 10 when R̄ = W_i log2(1+γ_th).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        w = config["wireless"]
        self.carrier_ghz = float(w["carrier_frequency_hz"]) / 1e9  # W_c [GHz]
        self.bandwidth_hz = float(w["total_bandwidth_hz"])  # IC: W_i ← total bandwidth
        self.bs_height = float(w["bs_height_m"])
        self.device_height = float(w["device_height_m"])
        self.distance_2d = float(w["distance_2d_m"])
        self.clutter_size = float(w["clutter_size_m"])
        self.clutter_density = float(w["clutter_density"])
        self.clutter_height = float(w["clutter_height_m"])
        self.noise_power_db = float(w["noise_power_db"])
        self.noise_watt = 10 ** (self.noise_power_db / 10.0)
        self.p_max = float(w["p_max_watt"])  # IMPLEMENTATION CHOICE (not in Table IV)
        self.shadow_sigma_db = float(w.get("shadow_fading_std_db", 4.0))
        dh = self.bs_height - self.device_height
        self.distance_3d = float(np.sqrt(self.distance_2d**2 + dh**2))

    def snr_linear(self, tx_power_watt: float, h_power: float, path_loss_db: float) -> float:
        """Paper eq. (8): γ = 10^(-PL/10) * P * |H|^2 / N_c (realized LoS or NLoS PL)."""
        return float(
            (10 ** (-path_loss_db / 10.0)) * float(tx_power_watt) * float(h_power) / max(self.noise_watt, 1e-30)
        )

    def snr_db(self, tx_power_watt: float, h_power: float, path_loss_db: float) -> float:
        gamma = self.snr_linear(tx_power_watt, h_power, path_loss_db)
        return float(10.0 * np.log10(max(gamma, 1e-30)))

    def channel_capacity_bps(self, snr_linear: float) -> float:
        """Paper eq. (9): R_{i,k} = W_i log2(1 + γ). W_i is IC (total bandwidth)."""
        return float(self.bandwidth_hz * np.log2(1.0 + max(float(snr_linear), 0.0)))

    def is_outage(self, snr_db: float, gamma_th_db: float) -> bool:
        """
        Slot outage: γ < γ_th.

        Paper eq. (10) is P[R < R̄]. R̄ is NOT SPECIFIED; this indicator is Eq. (10)
        for the IC R̄ = W_i log2(1+γ_th).
        """
        return float(snr_db) < float(gamma_th_db)

    def outage_probability_eq10(self, tx_power_watt: float, path_loss_db: float, gamma_th_db: float) -> float:
        """
        Closed-form eq. (10) under |H|^2 ~ Exp(1) with R̄ = W_i log2(1+γ_th) (IC).

        ε = 1 - exp[ - 10^{PL/10} * N_c / P * (2^{R̄/W_i} - 1) ]
          = 1 - exp[ - 10^{PL/10} * N_c / P * γ_th ]
        """
        gamma_th = 10 ** (float(gamma_th_db) / 10.0)
        scale = (10 ** (float(path_loss_db) / 10.0)) * self.noise_watt / max(float(tx_power_watt), 1e-30) * gamma_th
        return float(1.0 - np.exp(-scale))

    def estimate_outage_probability(
        self,
        gamma_th_db: float,
        *,
        tx_power_watt: float | None = None,
        num_samples: int = 2000,
        seed: int = 0,
    ) -> dict[str, float]:
        """Monte-Carlo P(γ < γ_th) under InF-SH + Rayleigh block fading."""
        rng = np.random.default_rng(seed)
        p_tx = float(self.p_max if tx_power_watt is None else tx_power_watt)
        outages = 0
        snrs: list[float] = []
        capacities: list[float] = []
        for _ in range(int(num_samples)):
            h_real = rng.normal(0.0, 1.0 / np.sqrt(2.0))
            h_imag = rng.normal(0.0, 1.0 / np.sqrt(2.0))
            h_power = float(h_real * h_real + h_imag * h_imag)
            path_loss_db, _ = self.sample_path_loss_db(rng)
            gamma = self.snr_linear(p_tx, h_power, path_loss_db)
            snr_db = float(10.0 * np.log10(max(gamma, 1e-30)))
            snrs.append(snr_db)
            capacities.append(self.channel_capacity_bps(gamma))
            if self.is_outage(snr_db, gamma_th_db):
                outages += 1
        n = max(1, int(num_samples))
        return {
            "gamma_th_db": float(gamma_th_db),
            "outage_probability": float(outages) / n,
            "mean_snr_db": float(np.mean(snrs)),
            "mean_capacity_bps": float(np.mean(capacities)),
            "bandwidth_hz": float(self.bandwidth_hz),
            "num_samples": float(n),
            "tx_power_watt": p_tx,
        }

    def path_loss_los_db(self) -> float:
        """Paper eq. (4); W_c in GHz."""
        return 31.84 + 21.5 * np.log10(self.distance_3d) + 19.0 * np.log10(self.carrier_ghz)

    def path_loss_general_db(self, rng: np.random.Generator) -> float:
        """Paper eq. (7) + shadow fading std 4.0 dB."""
        pl = 33.63 + 21.9 * np.log10(self.distance_3d) + 20.0 * np.log10(self.carrier_ghz)
        return float(pl + rng.normal(0.0, self.shadow_sigma_db))

    def los_probability(self) -> float:
        """
        Paper eq. (5) as InF-SH LoS probability (3GPP TR 38.901, cited by the paper):

            k = -D_clutter / ln(1-δ) * (h_BS - h_{i,R}) / (h_c - h_{i,R})
            P_LoS = exp( - D^{2D} / k )

        The paper typesets a product of fractions; that reading drives P_LoS → 0 at
        Table IV geometry. The cited InF-SH model (height ratio inside k) is used.
        """
        den_ln = np.log(1.0 - self.clutter_density)
        height_ratio = (self.bs_height - self.device_height) / (self.clutter_height - self.device_height)
        if den_ln >= 0 or height_ratio <= 0 or not np.isfinite(height_ratio):
            return 0.0
        k = -self.clutter_size / den_ln * height_ratio
        if k <= 0 or not np.isfinite(k):
            return 0.0
        return float(np.clip(np.exp(-self.distance_2d / k), 0.0, 1.0))

    def sample_path_loss_db(self, rng: np.random.Generator) -> tuple[float, bool]:
        p_los = self.los_probability()
        is_los = bool(rng.random() < p_los)
        pl_los = self.path_loss_los_db()
        if is_los:
            return float(pl_los), True
        pl_nlos = max(self.path_loss_general_db(rng), pl_los)  # paper eq. (6)
        return float(pl_nlos), False

    def required_power(self, snr_target_db: float, h_power: float, path_loss_db: float) -> float:
        """Paper eq. (23): p_req = γ_th * N_c / |H|^2 * 10^(PL/10)."""
        gamma = 10 ** (snr_target_db / 10.0)
        return float(gamma * self.noise_watt / max(h_power, 1e-18) * (10 ** (path_loss_db / 10.0)))

    def sample(self, device_id: int, snr_target_db: float, rng: np.random.Generator) -> ChannelSample:
        # Rayleigh block fading: |H|^2 ~ Exp(1) via |CN(0,1)|^2; new draw each slot.
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
