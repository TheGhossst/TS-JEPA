"""Evaluation plotting helpers (IMPLEMENTATION CHOICE — not paper figures verbatim)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


def plot_nmae_by_horizon(
    nmae_report: Mapping[str, Any],
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """
    Plot per-horizon NMAE for h = 1..Kp from evaluate_prediction_horizon_nmae output.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_h = dict(nmae_report.get("nmae_by_horizon") or {})
    if not by_h:
        raise ValueError("nmae_report has empty nmae_by_horizon")
    horizons = sorted(int(h) for h in by_h.keys())
    values = [float(by_h[str(h)]) for h in horizons]
    overall = nmae_report.get("nmae")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(horizons, values, marker="o", linewidth=2)
    ax.set_xlabel("Prediction horizon h")
    ax.set_ylabel("Command NMAE (mean |Δu| / 40 N)")
    ax.set_xticks(horizons)
    ax.grid(True, alpha=0.3)
    if title is None:
        kp = nmae_report.get("kp", max(horizons) if horizons else "?")
        split = nmae_report.get("split", "jepa_test")
        title = f"NMAE vs horizon (Kp={kp}, split={split})"
    ax.set_title(title)
    if overall is not None and overall == overall:  # not NaN
        ax.axhline(float(overall), color="C1", linestyle="--", label=f"overall={float(overall):.4f}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_embedding_tsne(
    tsne_report: Mapping[str, Any],
    out_path: Path | str,
    *,
    title: str = "Context-encoder embeddings (t-SNE)",
) -> Path:
    """
    Plot 2-D t-SNE of context embeddings colored by cart position.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray(tsne_report.get("coords") or [])
    colors = np.asarray(tsne_report.get("cart_positions") or [])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("tsne_report.coords must be [N, 2]")
    if colors.size != coords.shape[0]:
        raise ValueError("tsne_report.cart_positions length must match coords rows")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="viridis", s=8, alpha=0.75)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("cart position x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_wireless_control_scores(
    wireless_report: Mapping[str, Any],
    out_path: Path | str,
    *,
    title: str = "Wireless closed-loop mean control score",
) -> Path:
    """
    Plot mean control score vs SNR for each scheduler policy.

    Accepts either:
      {policy: {snr_str: {mean_control_score, ...}}}
    or the nested plan §17 report:
      {channel_model: ..., policies: {policy: {snr_str: {...}}}}
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not wireless_report:
        raise ValueError("wireless_report is empty")

    policies = wireless_report.get("policies") if isinstance(wireless_report, dict) else None
    plot_data = policies if isinstance(policies, dict) else wireless_report

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for policy, snr_map in plot_data.items():
        if policy in {"channel_model", "skipped", "reason", "checks"}:
            continue
        if not isinstance(snr_map, dict):
            continue
        items: list[tuple[float, float]] = []
        for snr_key, payload in snr_map.items():
            if not isinstance(payload, dict) or "mean_control_score" not in payload:
                continue
            items.append((float(snr_key), float(payload["mean_control_score"])))
        if not items:
            continue
        items.sort(key=lambda x: x[0])
        snrs = [s for s, _ in items]
        scores = [v for _, v in items]
        ax.plot(snrs, scores, marker="o", linewidth=2, label=str(policy))
    ax.set_xlabel("SNR target (dB)")
    ax.set_ylabel("Mean control score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_fig4_mape(
    fig4_report: Mapping[str, Any],
    out_path: Path | str,
    *,
    title: str = "Fig. 4 consecutive-frame MAPE vs sampling interval",
) -> Path:
    """Plot Eq. (26) MAPE vs sampling interval, with and without augmentation."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_rate = dict(fig4_report.get("by_sampling_interval_ms") or {})
    if not by_rate:
        raise ValueError("fig4_report has empty by_sampling_interval_ms")

    def _payload(rate: float) -> Mapping[str, Any]:
        for key in (str(rate), str(int(rate)) if float(rate).is_integer() else str(rate)):
            if key in by_rate:
                return by_rate[key]
        raise KeyError(rate)

    rates = sorted(float(k) for k in by_rate.keys())
    raw = [float(_payload(rate)["without_augmentation"]["mean_mape_percent"]) for rate in rates]
    aug = [float(_payload(rate)["with_augmentation"]["mean_mape_percent"]) for rate in rates]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rates, raw, marker="o", linewidth=2, label="without augmentation")
    ax.plot(rates, aug, marker="s", linewidth=2, label="with augmentation")
    ax.set_xlabel("Sampling interval (ms)")
    ax.set_ylabel("Consecutive-frame MAPE (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
