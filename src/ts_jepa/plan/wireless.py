"""Plan §13 wireless channel model specification (InF-SH / Table IV)."""

from __future__ import annotations

from typing import Any

from ts_jepa.plan.enforce import plan_enforced

# Paper Table IV + plan §13 SNR thresholds (paper-specified).
PLAN_WIRELESS: dict[str, Any] = {
    "scenario": "InF-SH",
    "hall_size_m": [300.0, 150.0],
    "room_height_m": 6.0,
    "bs_height_m": 10.0,
    "device_height_m": 1.5,
    "carrier_frequency_hz": 3.75e9,  # paper symbol W_c
    "total_bandwidth_hz": 20e6,
    "clutter_height_m": 3.0,
    "clutter_size_m": 2.0,
    "clutter_density": 0.60,
    "distance_2d_m": 50.0,
    "noise_power_db": -95.0,
    "shadow_fading_std_db": 4.0,
    "snr_targets_db": [5, 10, 20],
    "capacity_formula": "W_i * log2(1 + gamma)",
    "outage_definition": "P(gamma < gamma_th)",
    "rayleigh_block_fading": True,
    "aoi_resets_on": "scheduled",
    "lyapunov_B_omitted": True,
    # Numerically NOT SPECIFIED (plan §13); required to run Algorithm 2:
    "implementation_choice_keys": (
        "max_devices_scheduled_J",
        "p_max_watt",
        "drift_plus_penalty_V",
        "aoi_threshold_beta_th",
        "num_devices",
    ),
}


def assert_plan_wireless_config(config: dict[str, Any]) -> None:
    """
    Raise when wireless config drifts from plan §13 / Table IV paper-specified values.

    Does not hard-assert IMPLEMENTATION CHOICE scalars (J, I, V, β_th, p_max).
    """
    if not plan_enforced(config):
        return
    errors: list[str] = []
    w = config.get("wireless", {})
    if not w:
        raise ValueError("Plan §13 requires a wireless: block in config")

    checks = [
        ("hall_size_m", list(w.get("hall_size_m", [])), PLAN_WIRELESS["hall_size_m"]),
        ("room_height_m", float(w.get("room_height_m", float("nan"))), PLAN_WIRELESS["room_height_m"]),
        ("bs_height_m", float(w.get("bs_height_m", float("nan"))), PLAN_WIRELESS["bs_height_m"]),
        ("device_height_m", float(w.get("device_height_m", float("nan"))), PLAN_WIRELESS["device_height_m"]),
        (
            "carrier_frequency_hz",
            float(w.get("carrier_frequency_hz", float("nan"))),
            PLAN_WIRELESS["carrier_frequency_hz"],
        ),
        (
            "total_bandwidth_hz",
            float(w.get("total_bandwidth_hz", float("nan"))),
            PLAN_WIRELESS["total_bandwidth_hz"],
        ),
        ("clutter_height_m", float(w.get("clutter_height_m", float("nan"))), PLAN_WIRELESS["clutter_height_m"]),
        ("clutter_size_m", float(w.get("clutter_size_m", float("nan"))), PLAN_WIRELESS["clutter_size_m"]),
        ("clutter_density", float(w.get("clutter_density", float("nan"))), PLAN_WIRELESS["clutter_density"]),
        ("distance_2d_m", float(w.get("distance_2d_m", float("nan"))), PLAN_WIRELESS["distance_2d_m"]),
        ("noise_power_db", float(w.get("noise_power_db", float("nan"))), PLAN_WIRELESS["noise_power_db"]),
        (
            "shadow_fading_std_db",
            float(w.get("shadow_fading_std_db", float("nan"))),
            PLAN_WIRELESS["shadow_fading_std_db"],
        ),
    ]
    for name, got, expected in checks:
        if isinstance(expected, list):
            if list(got) != list(expected):
                errors.append(f"wireless.{name}: expected {expected}, got {got}")
        elif not (isinstance(got, float) and abs(got - float(expected)) <= 1e-6 * max(1.0, abs(float(expected)))):
            errors.append(f"wireless.{name}: expected {expected}, got {got}")

    snr = [int(x) for x in w.get("snr_targets_db", [])]
    if snr != PLAN_WIRELESS["snr_targets_db"]:
        errors.append(
            f"wireless.snr_targets_db: expected {PLAN_WIRELESS['snr_targets_db']}, got {snr}"
        )

    for key in PLAN_WIRELESS["implementation_choice_keys"]:
        if key not in w:
            errors.append(f"wireless.{key} must be set (IMPLEMENTATION CHOICE; plan §13)")

    if errors:
        raise ValueError("Plan §13 wireless config mismatch:\n  - " + "\n  - ".join(errors))
