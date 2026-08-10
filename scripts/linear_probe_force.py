#!/usr/bin/env python
"""
Closed-form linear probe: frozen JEPA embedding -> command_norm.

Reuses ActorEmbeddingDataset + context_encoder exactly as train_semantic_actor does.
No gradient descent, dropout, or nonlinearity in the probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ts_jepa.config import jepa_run_dirname, load_config, project_root
from ts_jepa.data.datasets import ActorEmbeddingDataset, load_command_normalizer
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.models.ts_jepa import TSJEPA


def collect_xy(dataset: ActorEmbeddingDataset) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for i in range(len(dataset)):
        item = dataset[i]
        xs.append(item["embedding"].numpy().astype(np.float64))
        ys.append(float(item["command_norm"].reshape(-1)[0].item()))
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.float64)


def linear_probe(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Closed-form least squares with intercept: y ~ [1, x] @ w."""
    n, d = x.shape
    a = np.concatenate([np.ones((n, 1), dtype=np.float64), x], axis=1)
    w, residuals, rank, singular = np.linalg.lstsq(a, y, rcond=None)
    y_hat = a @ w
    ss_res = float(np.sum((y - y_hat) ** 2))
    y_mean = float(y.mean())
    ss_tot = float(np.sum((y - y_mean) ** 2))
    mean_predictor_mse = float(np.var(y))  # population variance, ddof=0
    probe_mse = ss_res / n
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n_samples": float(n),
        "embedding_dim": float(d),
        "lstsq_rank": float(rank),
        "r2": float(r2),
        "probe_mse": float(probe_mse),
        "mean_predictor_mse": float(mean_predictor_mse),
        "probe_mse_over_mean_mse": float(probe_mse / mean_predictor_mse) if mean_predictor_mse > 0 else float("nan"),
        "y_mean": y_mean,
        "y_std": float(y.std()),
        "y_var": float(np.var(y)),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "y_hat_std": float(y_hat.std()),
        "y_hat_min": float(y_hat.min()),
        "y_hat_max": float(y_hat.max()),
        "embedding_global_std": float(x.std()),
        "embedding_mean_per_dim_std": float(x.std(axis=0).mean()),
        "embedding_median_per_dim_std": float(np.median(x.std(axis=0))),
        "embedding_min": float(x.min()),
        "embedding_max": float(x.max()),
        "intercept": float(w[0]),
        "coef_l2": float(np.linalg.norm(w[1:])),
        "coef_abs_max": float(np.max(np.abs(w[1:]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_fixed.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root(config)
    data_root = root / config["paths"]["data_root"]
    runs_root = root / config["paths"]["runs_root"]
    device = select_device(args.device)
    print(describe_device(device))

    jepa_ckpt = resolve_run_checkpoint(
        runs_root,
        jepa_run_dirname(config),
        explicit=Path(args.jepa_checkpoint) if args.jepa_checkpoint else None,
        seed=args.seed,
    )
    print(f"JEPA checkpoint: {jepa_ckpt}")

    ckpt = torch.load(jepa_ckpt, map_location="cpu", weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    normalizer = load_command_normalizer(config, data_root=data_root)
    train_dir = data_root / "trajectories" / "actor" / "train"
    print(f"Building ActorEmbeddingDataset from {train_dir} ...")
    train_ds = ActorEmbeddingDataset(
        train_dir, config, normalizer, jepa.context_encoder, device, training=True
    )
    print(f"Collected {len(train_ds)} actor-train samples")

    x, y = collect_xy(train_ds)
    report = {
        "jepa_checkpoint": str(jepa_ckpt),
        "data_root": str(data_root),
        "train_dir": str(train_dir),
        "normalizer_mean": float(normalizer.mean),
        "normalizer_std": float(normalizer.std),
        "probe": linear_probe(x, y),
    }

    out = (
        Path(args.out)
        if args.out
        else runs_root / "eval" / "linear_probe_force_dp_fixed_seed0.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    p = report["probe"]
    print("=== Linear probe (embedding -> command_norm) ===")
    print(f"n={int(p['n_samples'])}  dim={int(p['embedding_dim'])}  lstsq_rank={int(p['lstsq_rank'])}")
    print(f"R^2                 = {p['r2']:.8f}")
    print(f"probe MSE           = {p['probe_mse']:.8f}")
    print(f"mean-predictor MSE  = {p['mean_predictor_mse']:.8f}  (= np.var(y))")
    print(f"probe/mean MSE      = {p['probe_mse_over_mean_mse']:.8f}")
    print(f"y: mean={p['y_mean']:.6g} std={p['y_std']:.6g} range=[{p['y_min']:.6g}, {p['y_max']:.6g}]")
    print(f"y_hat: std={p['y_hat_std']:.6g} range=[{p['y_hat_min']:.6g}, {p['y_hat_max']:.6g}]")
    print(
        f"embedding: global_std={p['embedding_global_std']:.6g} "
        f"mean_per_dim_std={p['embedding_mean_per_dim_std']:.6g} "
        f"median_per_dim_std={p['embedding_median_per_dim_std']:.6g}"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
