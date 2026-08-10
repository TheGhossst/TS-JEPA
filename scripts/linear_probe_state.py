#!/usr/bin/env python
"""
Closed-form linear probe: frozen JEPA embedding -> physical state [x, x_dot, theta, theta_dot].

Mirrors ActorEmbeddingDataset embedding extraction exactly:
  NPZ frames -> PreprocessPipeline(training=False, stochastic=False)
  -> assemble_context(kappa) -> frozen context_encoder -> 256-D embedding.

State targets are states[time_index] at the same timestep as each embedding sample.
Fit independent Ridge probes (L2-regularized linear regression) on actor/train only;
report in-sample and OOS metrics on actor/test.

Diagnostic only — no JEPA or actor training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from sklearn.linear_model import Ridge
except ImportError:  # pragma: no cover - optional dependency
    Ridge = None  # type: ignore[misc, assignment]

from ts_jepa.config import jepa_run_dirname, load_config, project_root
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.preprocessing.pipeline import PreprocessPipeline

STATE_NAMES = ("x", "x_dot", "theta", "theta_dot")


def collect_embeddings_and_states(
    trajectory_dir: Path,
    config: dict,
    encoder: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect (X, Y) aligned with ActorEmbeddingDataset sample ordering.

    One sample per (trajectory file, time_index) for time_index in range(len(commands)).
    """
    pipeline = PreprocessPipeline(config, training=False)
    kappa = int(config["input"]["kappa"])
    files = sorted(Path(trajectory_dir).glob("*.npz"))
    if not files:
        raise ValueError(f"No trajectory files found under {trajectory_dir}")

    embeddings: list[np.ndarray] = []
    states: list[np.ndarray] = []

    encoder.eval()
    with torch.no_grad():
        for file_path in files:
            with np.load(file_path) as data:
                frames = np.asarray(data["frames"])
                commands = np.asarray(data["commands"], dtype=np.float32)
                state_arr = np.asarray(data["states"], dtype=np.float64)

            if state_arr.ndim != 2 or state_arr.shape[1] != 4:
                raise ValueError(
                    f"Expected states [T, 4] in {file_path}, got shape {state_arr.shape}"
                )
            n = int(commands.shape[0])
            if frames.shape[0] != n or state_arr.shape[0] != n:
                raise ValueError(
                    f"Length mismatch in {file_path}: "
                    f"frames={frames.shape[0]} commands={n} states={state_arr.shape[0]}"
                )

            cached = [pipeline.process_frame(frames[t], stochastic=False) for t in range(n)]
            batch_contexts: list[torch.Tensor] = []
            batch_states: list[np.ndarray] = []
            for time_index in range(n):
                processed = {t: cached[t] for t in range(n)}
                context = pipeline.assemble_context(processed, time_index, kappa=kappa)
                batch_contexts.append(context)
                batch_states.append(state_arr[time_index].copy())

            bs = 64
            file_embeddings: list[np.ndarray] = []
            for start in range(0, len(batch_contexts), bs):
                chunk = torch.stack(batch_contexts[start : start + bs], dim=0).to(device)
                file_embeddings.append(encoder(chunk).cpu().numpy())

            emb = np.concatenate(file_embeddings, axis=0)
            for i in range(emb.shape[0]):
                embeddings.append(emb[i].astype(np.float64))
                states.append(batch_states[i])

    return np.stack(embeddings, axis=0), np.stack(states, axis=0)


def count_unique_embeddings_6dp(x: np.ndarray) -> int:
    rounded = np.round(x, 6)
    return len({tuple(row) for row in rounded})


def embedding_stats(x: np.ndarray) -> dict[str, float | int]:
    return {
        "n_samples": int(x.shape[0]),
        "embedding_dim": int(x.shape[1]),
        "global_std": float(x.std()),
        "mean_per_dim_std": float(x.std(axis=0).mean()),
        "unique_embeddings_6dp": int(count_unique_embeddings_6dp(x)),
    }


def fit_ridge_with_intercept(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit y ~ [1, x] @ w with L2 (Ridge) regularization on embedding coefficients only.

    Returns (intercept, coef) where coef has shape [d].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if Ridge is not None:
        model = Ridge(alpha=float(alpha), fit_intercept=True)
        model.fit(x, y)
        return np.asarray(model.intercept_, dtype=np.float64), np.asarray(model.coef_, dtype=np.float64)

    # Manual Tikhonov on augmented design [1, x]; regularize embedding dims only.
    n, d = x.shape
    design = np.concatenate([np.ones((n, 1), dtype=np.float64), x], axis=1)
    reg = np.zeros((d + 1, d + 1), dtype=np.float64)
    reg[1:, 1:] = float(alpha) * np.eye(d, dtype=np.float64)
    w = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return w[0], w[1:]


def predict_ridge(x: np.ndarray, intercept: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return intercept + x @ coef


def regression_metrics(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(y_hat, dtype=np.float64).reshape(-1)
    n = y.shape[0]
    mse = float(np.mean((y - y_hat) ** 2))
    y_var = float(np.var(y))
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "n_samples": float(n),
        "mse": mse,
        "target_variance": y_var,
        "r2": float(r2),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "pred_std": float(y_hat.std()),
    }


def probe_variable(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    intercept, coef = fit_ridge_with_intercept(x_train, y_train, alpha=alpha)
    y_hat_train = predict_ridge(x_train, intercept, coef)
    y_hat_test = predict_ridge(x_test, intercept, coef)
    train_metrics = regression_metrics(y_train, y_hat_train)
    test_metrics = regression_metrics(y_test, y_hat_test)
    return {
        "ridge_alpha": float(alpha),
        "intercept": float(intercept),
        "coef_l2": float(np.linalg.norm(coef)),
        "coef_abs_max": float(np.max(np.abs(coef))),
        "train": train_metrics,
        "test": test_metrics,
    }


def print_results_table(per_variable: dict[str, dict[str, object]]) -> None:
    header = f"{'variable':<12}{'train_R2':>12}{'test_R2':>12}{'train_MSE':>14}{'test_MSE':>14}"
    print(header)
    for name in STATE_NAMES:
        row = per_variable[name]
        train = row["train"]
        test = row["test"]
        print(
            f"{name:<12}"
            f"{train['r2']:>12.6f}"
            f"{test['r2']:>12.6f}"
            f"{train['mse']:>14.8f}"
            f"{test['mse']:>14.8f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_35ms.yaml")
    parser.add_argument(
        "--jepa-checkpoint",
        type=str,
        default="runs/ts_jepa_dp_35ms/seed_0/best.pt",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge L2 regularization strength")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root(config)
    data_root = root / config["paths"]["data_root"]
    runs_root = root / config["paths"]["runs_root"]
    device = select_device(args.device)
    print(describe_device(device))

    explicit_ckpt = Path(args.jepa_checkpoint)
    if not explicit_ckpt.is_absolute():
        explicit_ckpt = root / explicit_ckpt
    jepa_ckpt = resolve_run_checkpoint(
        runs_root,
        jepa_run_dirname(config),
        explicit=explicit_ckpt if explicit_ckpt.exists() else None,
        seed=args.seed,
    )
    print(f"JEPA checkpoint: {jepa_ckpt}")

    ckpt = torch.load(jepa_ckpt, map_location="cpu", weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(ckpt["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    train_dir = data_root / "trajectories" / "actor" / "train"
    test_dir = data_root / "trajectories" / "actor" / "test"
    print(f"Collecting actor/train from {train_dir} ...")
    x_train, y_train = collect_embeddings_and_states(
        train_dir, config, jepa.context_encoder, device
    )
    print(f"Collecting actor/test from {test_dir} ...")
    x_test, y_test = collect_embeddings_and_states(
        test_dir, config, jepa.context_encoder, device
    )
    print(f"Train samples: {x_train.shape[0]}  Test samples: {x_test.shape[0]}  dim={x_train.shape[1]}")

    train_emb_stats = embedding_stats(x_train)
    test_emb_stats = embedding_stats(x_test)

    per_variable: dict[str, dict[str, object]] = {}
    for j, name in enumerate(STATE_NAMES):
        per_variable[name] = probe_variable(
            x_train,
            y_train[:, j],
            x_test,
            y_test[:, j],
            alpha=float(args.alpha),
        )

    report = {
        "jepa_checkpoint": str(jepa_ckpt),
        "config": args.config,
        "ridge_alpha": float(args.alpha),
        "solver": "sklearn.linear_model.Ridge" if Ridge is not None else "manual_tikhonov",
        "data_root": str(data_root),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "alignment": "states[time_index] paired with ActorEmbeddingDataset-equivalent embedding path",
        "preprocessing": "PreprocessPipeline(training=False), process_frame(stochastic=False), kappa context",
        "embedding_train": train_emb_stats,
        "embedding_test": test_emb_stats,
        "per_variable": per_variable,
    }

    out = (
        Path(args.out)
        if args.out
        else runs_root / "eval" / "linear_probe_state_dp_35ms_seed0.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("\n=== Ridge linear probe (embedding -> physical state) ===")
    print(f"alpha={args.alpha}  solver={'sklearn.Ridge' if Ridge is not None else 'manual_tikhonov'}")
    print_results_table(per_variable)
    print("\n=== Embedding statistics ===")
    print(
        f"train: n={train_emb_stats['n_samples']} "
        f"global_std={train_emb_stats['global_std']:.6g} "
        f"mean_per_dim_std={train_emb_stats['mean_per_dim_std']:.6g} "
        f"unique_6dp={train_emb_stats['unique_embeddings_6dp']}"
    )
    print(
        f"test:  n={test_emb_stats['n_samples']} "
        f"global_std={test_emb_stats['global_std']:.6g} "
        f"mean_per_dim_std={test_emb_stats['mean_per_dim_std']:.6g} "
        f"unique_6dp={test_emb_stats['unique_embeddings_6dp']}"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
