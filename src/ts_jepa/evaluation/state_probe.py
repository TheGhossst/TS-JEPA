"""Frozen-JEPA kinematic probes: z → (x, ẋ, θ, θ̇).

Does not train or modify TS-JEPA. Embeddings come from the frozen context
encoder on existing actor trajectories (same pairing as ActorEmbeddingDataset).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_jepa.config import jepa_run_dirname, project_root
from ts_jepa.data.datasets import ActorEmbeddingDataset, load_command_normalizer
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.evaluation.command_linear_probe import (
    collect_actor_arrays,
    fit_linear_ols,
    pearson_corr,
    predict_linear_ols,
    _standardize_apply,
    _standardize_fit,
)
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.plan.enforce import jepa_uses_kappa_stack

STATE_DIM_NAMES: tuple[str, ...] = (
    "cart_position",
    "cart_velocity",
    "pole_angle",
    "pole_angular_velocity",
)

PEARSON_GOOD = 0.50
PEARSON_WEAK = 0.20
REL_MSE_GOOD = 0.10


def kappa_for_state_history(config: dict[str, Any]) -> int:
    """Match the encoder's temporal window when JEPA uses a κ-stack; else one state."""
    if jepa_uses_kappa_stack(config):
        return max(1, int(config["input"]["kappa"]))
    return 1


def load_state_history_aligned(trajectory_dir: Path, kappa: int) -> np.ndarray:
    """
    Per-timestep κ-stack of raw states, aligned with ActorEmbeddingDataset order.

    Padding repeats the first state of that trajectory (same rule as
    PreprocessPipeline.assemble_context). Trajectories are not concatenated
    before stacking, so history never crosses episode boundaries.
    """
    k = max(1, int(kappa))
    chunks: list[np.ndarray] = []
    for file_path in sorted(Path(trajectory_dir).glob("*.npz")):
        with np.load(file_path) as data:
            states = np.asarray(data["states"], dtype=np.float32)
        if states.ndim != 2 or states.shape[1] < 4:
            raise ValueError(f"{file_path} states must be [T, 4], got {states.shape}")
        t_len = int(states.shape[0])
        rows = np.empty((t_len, k * states.shape[1]), dtype=np.float32)
        for t in range(t_len):
            start = t - k + 1
            parts: list[np.ndarray] = []
            for i in range(k):
                idx = start + i
                parts.append(states[0] if idx < 0 else states[idx])
            rows[t] = np.concatenate(parts, axis=0)
        chunks.append(rows)
    if not chunks:
        raise FileNotFoundError(f"no trajectory npz files in {trajectory_dir}")
    return np.concatenate(chunks, axis=0)


def regression_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    train_mean: float,
) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    baseline = np.full_like(target, float(train_mean))
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    base_mse = float(np.mean((baseline - target) ** 2))
    base_mae = float(np.mean(np.abs(baseline - target)))
    return {
        "n": int(target.size),
        "mse": mse,
        "mae": mae,
        "pearson": pearson_corr(pred, target),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "mean_baseline_mse": base_mse,
        "mean_baseline_mae": base_mae,
        "relative_mse_improvement_vs_mean": float((base_mse - mse) / max(base_mse, 1e-12)),
        "relative_mae_improvement_vs_mean": float((base_mae - mae) / max(base_mae, 1e-12)),
        "beats_mean_baseline_mse": bool(mse < base_mse - 1e-12),
        "beats_mean_baseline_mae": bool(mae < base_mae - 1e-12),
        "train_mean_baseline": float(train_mean),
    }


def _quality(metrics: dict[str, Any]) -> str:
    pearson = abs(float(metrics.get("pearson", 0.0)))
    rel = float(metrics.get("relative_mse_improvement_vs_mean", 0.0))
    if pearson >= PEARSON_GOOD and rel >= REL_MSE_GOOD:
        return "good"
    if pearson >= PEARSON_WEAK and metrics.get("beats_mean_baseline_mse"):
        return "weak"
    return "poor"


def interpret_state_probes(per_dim: dict[str, dict[str, Any]]) -> dict[str, Any]:
    qualities = {name: _quality(per_dim[name]["test"]) for name in STATE_DIM_NAMES if name in per_dim}
    pose = [qualities.get("cart_position", "poor"), qualities.get("pole_angle", "poor")]
    vel = [qualities.get("cart_velocity", "poor"), qualities.get("pole_angular_velocity", "poor")]
    pose_ok = any(q in {"good", "weak"} for q in pose)
    vel_ok = any(q in {"good", "weak"} for q in vel)
    all_poor = all(q == "poor" for q in qualities.values()) if qualities else True
    all_good = all(q == "good" for q in qualities.values()) if qualities else False

    if all_good:
        code = "z_contains_state"
        conclusion = (
            "Frozen z linearly recovers cart-pole kinematics. "
            "A closed-loop score of 0 is unlikely to be 'z is missing state'."
        )
    elif all_poor:
        code = "z_missing_state"
        conclusion = (
            "Frozen z does not beat a mean baseline on any kinematic coordinate. "
            "The embedding is missing the state information a controller needs."
        )
    elif pose_ok and not vel_ok:
        code = "z_has_pose_not_velocity"
        conclusion = (
            "z recovers pose (x and/or θ) but not velocities. "
            "A static-frame encoder would look like this; closed-loop control can still fail."
        )
    elif vel_ok or pose_ok:
        code = "z_partial_state"
        conclusion = (
            "z contains some kinematic signal but not a complete state estimate. "
            "That can be enough for a weak command probe and still fail closed-loop control."
        )
    else:
        code = "inconclusive"
        conclusion = "Kinematic probes are mixed or weak; treat z as not a proven state encoding."
    return {
        "code": code,
        "honest_conclusion": conclusion,
        "per_dim_quality": qualities,
        "thresholds": {
            "pearson_good": PEARSON_GOOD,
            "pearson_weak": PEARSON_WEAK,
            "relative_mse_improvement_good": REL_MSE_GOOD,
        },
    }


def _ols_dim(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    ridge: float = 1e-4,
) -> dict[str, Any]:
    mean, std = _standardize_fit(x_train)
    xtr = _standardize_apply(x_train, mean, std)
    xte = _standardize_apply(x_test, mean, std)
    train_mean = float(np.mean(y_train))
    coef = fit_linear_ols(xtr, y_train, ridge=ridge)
    return {
        "train": regression_metrics(predict_linear_ols(xtr, coef), y_train, train_mean=train_mean),
        "test": regression_metrics(predict_linear_ols(xte, coef), y_test, train_mean=train_mean),
        "ridge": float(ridge),
        "input_standardized": True,
    }


def run_state_probe(
    config: dict[str, Any],
    *,
    jepa_ckpt: Path,
    device: torch.device,
    data_root: Path | None = None,
    ridge: float = 1e-4,
) -> dict[str, Any]:
    """Freeze Ψθ and probe z → each CartPole state coordinate on actor splits."""
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    payload = torch.load(jepa_ckpt, map_location="cpu", weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(payload["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    normalizer = load_command_normalizer(config, data_root=root)
    train_dir = root / "trajectories" / "actor" / "train"
    test_dir = root / "trajectories" / "actor" / "test"
    print(f"encoding frozen JEPA embeddings from {train_dir} and {test_dir} ...")
    train_full = ActorEmbeddingDataset(
        train_dir, config, normalizer, jepa.context_encoder, device, training=True
    )
    test_ds = ActorEmbeddingDataset(
        test_dir, config, normalizer, jepa.context_encoder, device, training=False
    )
    print(f"encoded train={len(train_full)} test={len(test_ds)}")
    train_arrays = collect_actor_arrays(train_full, train_dir)
    test_arrays = collect_actor_arrays(test_ds, test_dir)

    z_train, z_test = train_arrays["z"], test_arrays["z"]
    s_train, s_test = train_arrays["state"], test_arrays["state"]
    if s_train.shape[1] < 4 or s_test.shape[1] < 4:
        raise ValueError(f"expected state dim ≥ 4, got train={s_train.shape} test={s_test.shape}")

    per_dim: dict[str, Any] = {}
    for i, name in enumerate(STATE_DIM_NAMES):
        per_dim[name] = _ols_dim(
            z_train,
            s_train[:, i],
            z_test,
            s_test[:, i],
            ridge=ridge,
        )

    multivariate = {}
    mean, std = _standardize_fit(z_train)
    xtr = _standardize_apply(z_train, mean, std)
    xte = _standardize_apply(z_test, mean, std)
    pred_test = np.stack(
        [
            predict_linear_ols(xte, fit_linear_ols(xtr, s_train[:, i], ridge=ridge))
            for i in range(4)
        ],
        axis=1,
    )
    residual = s_test[:, :4].astype(np.float64) - pred_test
    tgt = s_test[:, :4].astype(np.float64)
    sst = np.sum((tgt - tgt.mean(axis=0)) ** 2, axis=0)
    sse = np.sum(residual ** 2, axis=0)
    r2 = 1.0 - sse / np.maximum(sst, 1e-12)
    multivariate = {
        "test_mse_per_dim": [float(x) for x in np.mean(residual ** 2, axis=0)],
        "test_r2_per_dim": [float(x) for x in r2],
        "test_mean_r2": float(np.mean(r2)),
    }

    interpretation = interpret_state_probes(per_dim)
    return {
        "jepa_checkpoint": str(Path(jepa_ckpt).resolve()),
        "data_root": str(root),
        "jepa_frozen": True,
        "probe": "ridge_ols_linear",
        "splits": {
            "actor_train_samples": int(z_train.shape[0]),
            "actor_test_samples": int(z_test.shape[0]),
            "embedding_dim": int(z_train.shape[1]),
            "state_dim": int(s_train.shape[1]),
        },
        "state_dim_names": list(STATE_DIM_NAMES),
        "per_dim": per_dim,
        "multivariate": multivariate,
        "interpretation": interpretation,
    }


def resolve_jepa_ckpt(config: dict[str, Any], *, explicit: Path | None, seed: int | None) -> Path:
    runs_root = project_root(config) / config["paths"]["runs_root"]
    return resolve_run_checkpoint(
        runs_root,
        jepa_run_dirname(config),
        explicit=explicit,
        seed=seed,
    )
