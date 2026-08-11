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
    ax.set_ylabel("NMAE")
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

    Expects baseline_report['wireless'] structure:
      {policy: {snr_str: {mean_control_score, schedule_receive_rate}}}
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not wireless_report:
        raise ValueError("wireless_report is empty")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for policy, snr_map in wireless_report.items():
        items: list[tuple[float, float]] = []
        for snr_key, payload in snr_map.items():
            items.append((float(snr_key), float(payload["mean_control_score"])))
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
