from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Plan §15 Eq. (27): |max(u) − min(u)| = 40 N for cart-pole u ∈ [−20, +20].
DEFAULT_FORCE_RANGE_N = 40.0
NMAE_FORMULA = "mean(abs(u_pred - u_true)) / 40"

# Plan §18: embedding float bit-width is NOT SPECIFIED. The 8-bit accounting
# below is recovered so RGB 64×128×3×8 vs embedding 256×8 matches the paper's
# reported 98.95% reduction (1 − 2048/196608 = 98.958%). Not a paper dtype.
RECOVERED_EMBEDDING_BITS_PER_VALUE = 8
RECOVERED_RGB_BITS_PER_CHANNEL = 8
PAPER_COMMUNICATION_REDUCTION = 0.9895  # published headline figure


def physical_force_range_n(config: Mapping[str, Any] | None = None) -> float:
    """Return max(u)-min(u) in Newtons from simulation config (paper: 40 N)."""
    if config is None:
        return DEFAULT_FORCE_RANGE_N
    sim = config.get("simulation", {})
    return float(sim["control_max_N"]) - float(sim["control_min_N"])


def control_score(
    state: np.ndarray,
    desired_x: float = 0.0,
    position_tol: float = 0.05,
    angle_tol: float = 0.05,
) -> int:
    """Plan §15 Eq. (28): R=1 iff |x−x_d|≤0.05 and |ϑ|≤0.05."""
    x_ok = abs(float(state[0]) - desired_x) <= position_tol
    theta_ok = abs(float(state[2])) <= angle_tol
    return int(x_ok and theta_ok)


def nmae(
    pred: np.ndarray,
    target: np.ndarray,
    force_range_n: float = DEFAULT_FORCE_RANGE_N,
) -> float:
    """
    Plan §15 Eq. (27) physical-force NMAE.

    N^u = (1/K_p) Σ |ũ − u| / |max(u) − min(u)|
        = mean(|u_pred - u_true|) / 40  for cart-pole.

    Inputs must already be denormalized to Newtons.
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if pred.size == 0 or target.size == 0:
        return float("nan")
    denom = float(force_range_n)
    if denom <= 0.0 or not np.isfinite(denom):
        raise ValueError(f"force_range_n must be a positive finite Newton span, got {force_range_n}")
    return float(np.mean(np.abs(pred - target)) / denom)


def nmae_report_fields(
    *,
    value: float,
    force_range_n: float = DEFAULT_FORCE_RANGE_N,
    denormalized: bool = True,
) -> dict[str, Any]:
    return {
        "nmae": float(value),
        "nmae_formula": NMAE_FORMULA,
        "nmae_units": "N",
        "nmae_denominator": float(force_range_n),
        "denormalized": bool(denormalized),
        "plan_section": "15",
        "equation": "27",
    }


def consecutive_frame_mape(
    frame_t: np.ndarray,
    frame_tm1: np.ndarray,
    *,
    zero_denom: str = "skip",
) -> float:
    """
    Plan §15 Eq. (26): MAPE between consecutive frames, in percent.

        N^P_{i,k} = (1/p) Σ_υ |x_k(υ) − x_{k-1}(υ)| / |x_{k-1}(υ)| × 100%

    IMPLEMENTATION CHOICE: pixels with x_{k-1}(υ)=0 are skipped (`zero_denom='skip'`).
    The paper does not specify a zero-denominator convention.
    """
    curr = np.asarray(frame_t, dtype=np.float64).reshape(-1)
    prev = np.asarray(frame_tm1, dtype=np.float64).reshape(-1)
    if curr.size != prev.size or curr.size == 0:
        return float("nan")
    denom = np.abs(prev)
    if zero_denom == "skip":
        mask = denom > 0.0
        if not np.any(mask):
            return float("nan")
        return float(np.mean(np.abs(curr[mask] - prev[mask]) / denom[mask]) * 100.0)
    raise ValueError(f"unsupported zero_denom={zero_denom!r}")


def latent_cosine_by_horizon(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    """
    Per-horizon mean cosine similarity between predicted and target latents.

    Not a plan §15 paper metric (JEPA training diagnostic). pred/target: [N, Kp, D].
    """
    pred_a = np.asarray(pred, dtype=np.float64)
    tgt_a = np.asarray(target, dtype=np.float64)
    if pred_a.ndim == 2:
        pred_a = pred_a[np.newaxis, ...]
        tgt_a = tgt_a[np.newaxis, ...]
    if pred_a.shape != tgt_a.shape or pred_a.ndim != 3:
        raise ValueError(f"pred/target must be [N, Kp, D], got {pred_a.shape} vs {tgt_a.shape}")
    kp = int(pred_a.shape[1])
    out: dict[str, float] = {}
    for j in range(kp):
        a = pred_a[:, j, :]
        b = tgt_a[:, j, :]
        num = np.sum(a * b, axis=-1)
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        cos = num / np.maximum(den, eps)
        out[str(j + 1)] = float(np.mean(cos))
    return out


def communication_bits_rgb(
    height: int,
    width: int,
    channels: int = 3,
    bits_per_channel: int = RECOVERED_RGB_BITS_PER_CHANNEL,
) -> int:
    return int(height * width * channels * bits_per_channel)


def communication_bits_embedding(
    embedding_dim: int = 256,
    bits_per_value: int = RECOVERED_EMBEDDING_BITS_PER_VALUE,
) -> int:
    return int(embedding_dim * bits_per_value)


def communication_reduction_report(
    *,
    height: int,
    width: int,
    embedding_dim: int = 256,
    channels: int = 3,
    bits_per_channel: int = RECOVERED_RGB_BITS_PER_CHANNEL,
    bits_per_value: int = RECOVERED_EMBEDDING_BITS_PER_VALUE,
) -> dict[str, Any]:
    """
    Plan §15 communication efficiency: bits to send a frame vs an embedding.

    Default 8-bit width is recovered from the paper's 98.95% figure (plan §18),
    not a paper-stated embedding dtype and not an invented 256×32 identity.
    """
    rgb_bits = communication_bits_rgb(height, width, channels=channels, bits_per_channel=bits_per_channel)
    emb_bits = communication_bits_embedding(embedding_dim, bits_per_value=bits_per_value)
    reduction_ratio = float(rgb_bits) / float(max(1, emb_bits))
    reduction_fraction = 1.0 - (float(emb_bits) / float(max(1, rgb_bits)))
    return {
        "rgb_bits": rgb_bits,
        "embedding_bits": emb_bits,
        "reduction_ratio": reduction_ratio,
        "reduction_fraction": reduction_fraction,
        "reduction_percent": reduction_fraction * 100.0,
        "bits_saved": int(rgb_bits - emb_bits),
        "frame_shape": [int(height), int(width), int(channels)],
        "embedding_dim": int(embedding_dim),
        "bits_per_channel": int(bits_per_channel),
        "bits_per_value": int(bits_per_value),
        "embedding_bitwidth_source": "recovered_from_paper_98.95_percent_reduction",
        "plan_section": "15",
        "bitwidth_paper_exact": False,
    }


def summarize_scores(scores: list[float], *, reported_result: str = "best") -> dict[str, Any]:
    arr = np.asarray(scores, dtype=np.float64)
    payload = {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "best": float(arr.max()) if len(arr) else float("nan"),
        "worst": float(arr.min()) if len(arr) else float("nan"),
        "repetitions": len(arr),
        "reported_result": str(reported_result),
    }
    if str(reported_result) == "best":
        payload["reported"] = payload["best"]
    else:
        payload["reported"] = payload["mean"]
    return payload
