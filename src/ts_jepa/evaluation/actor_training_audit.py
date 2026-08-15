"""
Read-only Semantic Actor training-path audit.

Identifies implementation mismatches and degenerate (near-constant) actor solutions
without modifying architecture, hyperparameters, or checkpoints.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ts_jepa.config import project_root
from ts_jepa.data.datasets import ActorEmbeddingDataset, load_command_normalizer
from ts_jepa.data.trajectory_generator import CONTROL_TEACHER_TYPE
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.evaluation.metrics import nmae, physical_force_range_n
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.preprocessing.pipeline import PreprocessPipeline
from ts_jepa.training.actor_helpers import evaluate_mse, split_train_val_actor


def _state_dict_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state.keys()):
        digest.update(key.encode())
        arr = state[key].detach().cpu().numpy()
        digest.update(arr.tobytes())
    return digest.hexdigest()[:16]


def _distribution_stats(values: np.ndarray, *, unique_round: int = 6) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "num_unique_rounded": 0,
            "positive_fraction": float("nan"),
            "negative_fraction": float("nan"),
        }
    uniq = np.unique(np.round(v, unique_round))
    return {
        "count": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "num_unique_rounded": int(uniq.size),
        "positive_fraction": float(np.mean(v > 0.0)),
        "negative_fraction": float(np.mean(v < 0.0)),
    }


def _embedding_distribution(z: np.ndarray, *, near_tol: float = 1e-4, seed: int = 0) -> dict[str, Any]:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    norms = np.linalg.norm(z, axis=1)
    per_dim_mean = z.mean(axis=0)
    per_dim_std = z.std(axis=0)
    rounded = np.round(z, 6)
    unique_rows = {tuple(row) for row in rounded}
    n = z.shape[0]
    rng = np.random.default_rng(seed)
    num_pairs = min(5000, max(0, n * (n - 1) // 2))
    near_identical_pairs = 0
    if num_pairs > 0 and n >= 2:
        i_idx = rng.integers(0, n, size=num_pairs)
        j_idx = rng.integers(0, n, size=num_pairs)
        mask = i_idx == j_idx
        while mask.any():
            j_idx[mask] = rng.integers(0, n, size=int(mask.sum()))
            mask = i_idx == j_idx
        dists = np.linalg.norm(z[i_idx] - z[j_idx], axis=1)
        near_identical_pairs = int(np.sum(dists <= near_tol))
    return {
        "num_samples": int(n),
        "embedding_dim": int(z.shape[1]),
        "per_dimension_mean": {
            "mean": float(per_dim_mean.mean()),
            "std": float(per_dim_mean.std()),
            "min": float(per_dim_mean.min()),
            "max": float(per_dim_mean.max()),
        },
        "per_dimension_std": {
            "mean": float(per_dim_std.mean()),
            "std": float(per_dim_std.std()),
            "min": float(per_dim_std.min()),
            "max": float(per_dim_std.max()),
        },
        "norm_distribution": _distribution_stats(norms),
        "num_unique_embeddings_rounded_6dp": int(len(unique_rows)),
        "fraction_samples_with_duplicate_embedding": float(1.0 - len(unique_rows) / max(n, 1)),
        "near_identical_pairs_l2_le_tol": near_identical_pairs,
        "near_identical_pair_fraction": float(near_identical_pairs / max(num_pairs, 1)),
        "near_identical_tol": float(near_tol),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _correlation_report(z: np.ndarray, targets_norm: np.ndarray, preds_norm: np.ndarray) -> dict[str, Any]:
    y_tgt = targets_norm.reshape(-1)
    y_pred = preds_norm.reshape(-1)
    dim_corrs = []
    for d in range(z.shape[1]):
        dim_corrs.append(_pearson(z[:, d], y_tgt))
    dim_corrs_arr = np.asarray(dim_corrs, dtype=np.float64)
    abs_dim = np.abs(dim_corrs_arr)
    top_dims = np.argsort(-abs_dim)[:10]
    return {
        "pred_vs_target_pearson": _pearson(y_pred, y_tgt),
        "embedding_dim_max_abs_pearson_with_target": float(abs_dim.max()) if abs_dim.size else 0.0,
        "embedding_dim_mean_abs_pearson_with_target": float(abs_dim.mean()) if abs_dim.size else 0.0,
        "top_embedding_dims_by_abs_correlation_with_target": [
            {"dim": int(i), "pearson": float(dim_corrs_arr[i])} for i in top_dims
        ],
    }


def _collect_split_arrays(
    dataset: ActorEmbeddingDataset,
    actor: SemanticActor,
    normalizer: CommandNormalizer,
    device: torch.device,
    *,
    max_samples: int | None = None,
) -> dict[str, np.ndarray]:
    n = len(dataset)
    take = n if max_samples is None else min(n, max_samples)
    z_list: list[np.ndarray] = []
    tgt_norm: list[float] = []
    tgt_phys: list[float] = []
    pred_norm: list[float] = []
    pred_phys_list: list[float] = []
    actor.eval()
    with torch.no_grad():
        for i in range(take):
            item = dataset[i]
            emb = item["embedding"].unsqueeze(0).to(device)
            pred = float(actor(emb).reshape(-1)[0].item())
            cmd_n = float(item["command_norm"].reshape(-1)[0].item())
            cmd_p = float(item["command"].reshape(-1)[0].item())
            z_list.append(item["embedding"].numpy())
            tgt_norm.append(cmd_n)
            tgt_phys.append(cmd_p)
            pred_norm.append(float(normalizer.normalize(np.array([pred], dtype=np.float32))[0]))
            pred_phys_list.append(pred)
    pred_norm_arr = np.asarray(pred_norm, dtype=np.float64)
    pred_phys = np.asarray(pred_phys_list, dtype=np.float64)
    return {
        "embeddings": np.stack(z_list, axis=0),
        "targets_norm": np.asarray(tgt_norm, dtype=np.float64),
        "targets_phys": np.asarray(tgt_phys, dtype=np.float64),
        "preds_norm": pred_norm_arr,
        "preds_phys": np.asarray(pred_phys, dtype=np.float64).reshape(-1),
    }


def _command_stats_from_trajectories(trajectory_dir: Path, normalizer: CommandNormalizer) -> dict[str, Any]:
    physical: list[float] = []
    for file_path in sorted(trajectory_dir.glob("*.npz")):
        with np.load(file_path) as data:
            physical.extend(np.asarray(data["commands"], dtype=np.float32).reshape(-1).tolist())
    phys = np.asarray(physical, dtype=np.float64)
    norm = normalizer.normalize(phys.astype(np.float32))
    return {
        "physical_N": _distribution_stats(phys),
        "normalized": _distribution_stats(norm),
    }


def _build_sample_index(trajectory_dir: Path) -> list[dict[str, Any]]:
    meta: list[dict[str, Any]] = []
    for file_idx, file_path in enumerate(sorted(trajectory_dir.glob("*.npz"))):
        with np.load(file_path) as data:
            commands = np.asarray(data["commands"], dtype=np.float32)
            n_steps = int(commands.shape[0])
        for time_index in range(n_steps):
            meta.append(
                {
                    "sample_index": len(meta),
                    "trajectory_id": file_path.stem,
                    "file_idx": file_idx,
                    "time_index": time_index,
                    "command_physical_N": float(commands[time_index]),
                }
            )
    return meta


def _verify_command_indexing(dataset: ActorEmbeddingDataset, trajectory_dir: Path, *, max_check: int = 50) -> dict[str, Any]:
    meta = _build_sample_index(trajectory_dir)
    mismatches: list[dict[str, Any]] = []
    n = min(len(dataset), len(meta), max_check)
    for i in range(n):
        item = dataset[i]
        got_norm = float(item["command_norm"].reshape(-1)[0].item())
        expected_phys = meta[i]["command_physical_N"]
        expected_norm = float(dataset.normalizer.normalize(np.array([expected_phys], dtype=np.float32))[0])
        if abs(got_norm - expected_norm) > 1e-5:
            mismatches.append(
                {
                    "sample_index": i,
                    "trajectory_id": meta[i]["trajectory_id"],
                    "time_index": meta[i]["time_index"],
                    "expected_norm": expected_norm,
                    "got_norm": got_norm,
                }
            )
    train_ids = {m["trajectory_id"] for m in meta}
    return {
        "samples_checked": n,
        "command_index_mismatches": mismatches,
        "command_indexing_ok": len(mismatches) == 0,
        "num_trajectory_files": len(train_ids),
        "first_10_samples": meta[:10],
    }


def _split_disjointness(train_dir: Path, test_dir: Path) -> dict[str, Any]:
    train_ids = {p.stem for p in sorted(train_dir.glob("*.npz"))}
    test_ids = {p.stem for p in sorted(test_dir.glob("*.npz"))}
    overlap = sorted(train_ids & test_ids)
    return {
        "train_trajectory_ids": sorted(train_ids),
        "test_trajectory_ids": sorted(test_ids),
        "overlap_ids": overlap,
        "splits_disjoint": len(overlap) == 0,
    }


def _runtime_embedding_parity(
    config: dict[str, Any],
    jepa: TSJEPA,
    trajectory_dir: Path,
    device: torch.device,
    *,
    num_checks: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare ActorEmbeddingDataset embeddings vs runtime encode_context on identical frames."""
    rng = np.random.default_rng(seed)
    files = sorted(trajectory_dir.glob("*.npz"))
    if not files:
        return {"checked": 0, "max_abs_diff": float("nan"), "parity_ok": False, "reason": "no trajectories"}
    pipeline = PreprocessPipeline(config, training=False)
    jepa.eval()
    diffs: list[float] = []
    details: list[dict[str, Any]] = []
    for file_path in files[: max(1, min(3, len(files)))]:
        with np.load(file_path) as data:
            frames = np.asarray(data["frames"])
        n_steps = frames.shape[0]
        if n_steps < 1:
            continue
        indices = sorted(rng.choice(n_steps, size=min(num_checks, n_steps), replace=False).tolist())
        cached = [pipeline.process_frame(frames[t], stochastic=False) for t in range(n_steps)]
        for time_index in indices:
            ctx_dataset = pipeline.assemble_jepa_input(cached, time_index).unsqueeze(0).to(device)
            with torch.no_grad():
                z_dataset = jepa.encode_context(ctx_dataset).cpu().numpy().reshape(-1)
            ctx_runtime = pipeline.make_jepa_input(frames, time_index).unsqueeze(0).to(device)
            with torch.no_grad():
                z_runtime = jepa.encode_context(ctx_runtime).cpu().numpy().reshape(-1)
            diff = float(np.max(np.abs(z_dataset - z_runtime)))
            diffs.append(diff)
            details.append(
                {
                    "trajectory_id": file_path.stem,
                    "time_index": int(time_index),
                    "max_abs_embedding_diff": diff,
                }
            )
    max_diff = float(np.max(diffs)) if diffs else float("nan")
    return {
        "checked": len(diffs),
        "max_abs_diff": max_diff,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else float("nan"),
        "parity_ok": bool(np.isfinite(max_diff) and max_diff < 1e-5),
        "details": details[:10],
        "note": "ActorEmbeddingDataset and runtime both use eval preprocessing (training=False).",
    }


def _constant_mean_analysis(
    targets_norm: np.ndarray,
    preds_norm: np.ndarray,
    targets_phys: np.ndarray,
    preds_phys: np.ndarray,
) -> dict[str, Any]:
    tgt = targets_norm.reshape(-1)
    pred = preds_norm.reshape(-1)
    if tgt.size == 0:
        return {"detected": False, "reason": "empty"}
    mean_baseline = float(tgt.mean())
    mse_model = float(np.mean((pred - tgt) ** 2))
    mse_mean_baseline = float(np.mean((mean_baseline - tgt) ** 2))
    pred_std_norm = float(pred.std())
    tgt_std_norm = float(tgt.std())
    pred_std_phys = float(preds_phys.std())
    tgt_std_phys = float(targets_phys.std())
    uniq_pred_phys = int(np.unique(np.round(preds_phys, 4)).size)
    both_signs_pred = bool(np.any(preds_phys > 0.1) and np.any(preds_phys < -0.1))
    both_signs_tgt = bool(np.any(targets_phys > 0.1) and np.any(targets_phys < -0.1))
    detected = (
        pred_std_phys < 0.5 * max(tgt_std_phys, 1e-6)
        or uniq_pred_phys <= 3
        or (not both_signs_pred and both_signs_tgt)
        or (mse_model >= 0.95 * mse_mean_baseline and pred_std_norm < 0.25 * max(tgt_std_norm, 1e-6))
    )
    return {
        "detected_near_constant_mean_solution": bool(detected),
        "mse_normalized": mse_model,
        "mse_predicting_train_mean_baseline": mse_mean_baseline,
        "mse_ratio_model_over_mean_baseline": float(mse_model / max(mse_mean_baseline, 1e-12)),
        "pred_std_normalized": pred_std_norm,
        "target_std_normalized": tgt_std_norm,
        "pred_std_physical_N": pred_std_phys,
        "target_std_physical_N": tgt_std_phys,
        "pred_num_unique_physical_rounded_4dp": uniq_pred_phys,
        "target_has_both_signs": both_signs_tgt,
        "pred_has_both_signs": both_signs_pred,
    }


def _checkpoint_integrity(
    config: dict[str, Any],
    actor_payload: dict[str, Any],
    actor: SemanticActor,
    device: torch.device,
) -> dict[str, Any]:
    ckpt_state = actor_payload.get("actor")
    if ckpt_state is None:
        return {"ok": False, "reason": "actor key missing from checkpoint"}
    fresh = SemanticActor.from_config(config).to(device)
    loaded = SemanticActor.from_config(config).to(device)
    loaded.load_state_dict(ckpt_state)
    ckpt_fp = _state_dict_fingerprint(ckpt_state)
    loaded_fp = _state_dict_fingerprint(loaded.state_dict())
    fresh_fp = _state_dict_fingerprint(fresh.state_dict())
    actor_fp = _state_dict_fingerprint(actor.state_dict())
    keys_match = set(ckpt_state.keys()) == set(actor.state_dict().keys())
    values_match = all(
        torch.allclose(ckpt_state[k], actor.state_dict()[k], atol=0.0, rtol=0.0)
        for k in ckpt_state.keys()
    )
    param_vals = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in actor.parameters()])
    return {
        "ok": bool(keys_match and values_match and ckpt_fp == actor_fp == loaded_fp),
        "checkpoint_fingerprint": ckpt_fp,
        "loaded_actor_fingerprint": actor_fp,
        "differs_from_random_init": ckpt_fp != fresh_fp,
        "all_keys_present": keys_match,
        "weights_bitwise_match_loaded_actor": values_match,
        "stored_val_loss": actor_payload.get("val_loss"),
        "stored_test_loss": actor_payload.get("test_loss"),
        "stored_jepa_checkpoint": actor_payload.get("jepa_checkpoint"),
        "stored_normalizer": actor_payload.get("normalizer"),
        "parameter_abs_mean": float(np.mean(np.abs(param_vals))),
        "parameter_std": float(param_vals.std()),
        "fraction_exact_zeros": float(np.mean(param_vals == 0.0)),
    }


def _jepa_checkpoint_consistency(
    actor_payload: dict[str, Any],
    jepa_checkpoint: Path,
    jepa: TSJEPA,
    jepa_payload: dict[str, Any],
) -> dict[str, Any]:
    stored = actor_payload.get("jepa_checkpoint")
    resolved_inference = str(Path(jepa_checkpoint).resolve())
    stored_resolved = str(Path(stored).resolve()) if stored else None
    paths_match = stored_resolved == resolved_inference if stored_resolved else False
    ckpt_encoder_fp = _state_dict_fingerprint(
        {k: v for k, v in jepa_payload["model"].items() if k.startswith("context_encoder")}
    )
    loaded_encoder_fp = _state_dict_fingerprint(
        {k: v for k, v in jepa.state_dict().items() if k.startswith("context_encoder")}
    )
    return {
        "inference_jepa_checkpoint": resolved_inference,
        "actor_checkpoint_records_jepa": stored,
        "paths_match": paths_match,
        "context_encoder_fingerprint_from_checkpoint_file": ckpt_encoder_fp,
        "context_encoder_fingerprint_loaded_model": loaded_encoder_fp,
        "encoder_weights_match_checkpoint_file": ckpt_encoder_fp == loaded_encoder_fp,
        "uses_same_frozen_encoder_as_actor_training": bool(paths_match and ckpt_encoder_fp == loaded_encoder_fp),
    }


def _metrics_block(
    actor: SemanticActor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    arrays_train: dict[str, np.ndarray],
    arrays_test: dict[str, np.ndarray],
    normalizer: CommandNormalizer,
    device: torch.device,
    config: dict[str, Any],
    *,
    checkpoint_metrics: dict[str, Any],
) -> dict[str, Any]:
    criterion = nn.MSELoss()
    force_range = physical_force_range_n(config)
    return {
        "checkpoint_recorded": checkpoint_metrics,
        "mse_normalized": {
            "train": evaluate_mse(actor, train_loader, device, criterion),
            "validation": evaluate_mse(actor, val_loader, device, criterion),
            "test": evaluate_mse(actor, test_loader, device, criterion),
        },
        "nmae_physical": {
            "train": nmae(arrays_train["preds_phys"], arrays_train["targets_phys"], force_range_n=force_range),
            "test": nmae(arrays_test["preds_phys"], arrays_test["targets_phys"], force_range_n=force_range),
        },
        "normalizer": normalizer.to_dict(),
        "normalizer_fit_source": "jepa_train_commands",
    }


def audit_actor_training(
    config: dict[str, Any],
    *,
    jepa_checkpoint: Path | str,
    actor_checkpoint: Path | str,
    data_root: Path | None = None,
    device: torch.device | None = None,
    max_embedding_samples: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """
    End-to-end read-only audit of Semantic Actor training vs inference paths.

    Does not modify checkpoints, datasets, or model weights.
    """
    from ts_jepa.device import select_device

    device = select_device(device)
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    jepa_ckpt = Path(jepa_checkpoint)
    actor_ckpt = Path(actor_checkpoint)

    jepa_payload = torch.load(jepa_ckpt, map_location=device, weights_only=False)
    actor_payload = torch.load(actor_ckpt, map_location=device, weights_only=False)

    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(jepa_payload["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    actor = SemanticActor.from_config(config).to(device)
    actor.load_state_dict(actor_payload["actor"])
    actor.eval()

    normalizer = load_command_normalizer(config, data_root=root)
    stats_path = root / "stats" / "command_norm.json"
    train_dir = root / "trajectories" / "actor" / "train"
    test_dir = root / "trajectories" / "actor" / "test"

    train_full = ActorEmbeddingDataset(train_dir, config, normalizer, jepa.context_encoder, device, training=True)
    test_ds = ActorEmbeddingDataset(test_dir, config, normalizer, jepa.context_encoder, device, training=False)

    val_fraction = float(config["semantic_actor"]["early_stopping"].get("val_fraction", 0.2))
    train_ds, val_ds = split_train_val_actor(train_full, val_fraction)
    batch_size = int(config["semantic_actor"]["optimizer"]["batch_size"])

    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, max(1, len(train_ds))),
        shuffle=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, max(1, len(val_ds))),
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=min(batch_size, max(1, len(test_ds))),
        shuffle=False,
        num_workers=0,
    )

    arrays_train = _collect_split_arrays(train_full, actor, normalizer, device, max_samples=max_embedding_samples)
    arrays_test = _collect_split_arrays(test_ds, actor, normalizer, device, max_samples=max_embedding_samples)

    jepa_fit_commands = _command_stats_from_trajectories(root / "trajectories" / "jepa" / "train", normalizer)

    report: dict[str, Any] = {
        "audit_version": 1,
        "seed": int(seed),
        "data_root": str(root),
        "jepa_checkpoint": str(jepa_ckpt.resolve()),
        "actor_checkpoint": str(actor_ckpt.resolve()),
        "1_jepa_checkpoint_consistency": _jepa_checkpoint_consistency(
            actor_payload, jepa_ckpt, jepa, jepa_payload
        ),
        "2_embedding_distributions": {
            "train": _embedding_distribution(arrays_train["embeddings"], seed=seed),
            "test": _embedding_distribution(arrays_test["embeddings"], seed=seed + 1),
        },
        "3_target_command_distributions": {
            "actor_train": _command_stats_from_trajectories(train_dir, normalizer),
            "actor_test": _command_stats_from_trajectories(test_dir, normalizer),
            "jepa_train_fit_source": jepa_fit_commands,
            "normalizer": normalizer.to_dict(),
            "normalizer_stats_path": str(stats_path),
        },
        "4_actor_prediction_distributions": {
            "train_normalized": _distribution_stats(arrays_train["preds_norm"]),
            "train_physical_N": _distribution_stats(arrays_train["preds_phys"]),
            "test_normalized": _distribution_stats(arrays_test["preds_norm"]),
            "test_physical_N": _distribution_stats(arrays_test["preds_phys"]),
        },
        "5_correlation": {
            "train": _correlation_report(
                arrays_train["embeddings"], arrays_train["targets_norm"], arrays_train["preds_norm"]
            ),
            "test": _correlation_report(
                arrays_test["embeddings"], arrays_test["targets_norm"], arrays_test["preds_norm"]
            ),
        },
        "6_training_metrics": _metrics_block(
            actor,
            train_loader,
            val_loader,
            test_loader,
            arrays_train,
            arrays_test,
            normalizer,
            device,
            config,
            checkpoint_metrics={
                "val_loss": actor_payload.get("val_loss"),
                "test_loss": actor_payload.get("test_loss"),
                "metrics_json": _load_metrics_json(actor_ckpt.parent / "metrics.json"),
            },
        ),
        "7_constant_mean_solution": {
            "train": _constant_mean_analysis(
                arrays_train["targets_norm"],
                arrays_train["preds_norm"],
                arrays_train["targets_phys"],
                arrays_train["preds_phys"],
            ),
            "test": _constant_mean_analysis(
                arrays_test["targets_norm"],
                arrays_test["preds_norm"],
                arrays_test["targets_phys"],
                arrays_test["preds_phys"],
            ),
        },
        "8_preprocessing_runtime_parity": {
            "embedding_encode_context": _runtime_embedding_parity(
                config, jepa, train_dir, device, num_checks=20, seed=seed
            ),
            "normalizer": {
                "checkpoint": actor_payload.get("normalizer"),
                "disk_command_norm_json": normalizer.to_dict(),
                "checkpoint_matches_disk": actor_payload.get("normalizer") == normalizer.to_dict(),
                "actor_training_uses_disk_normalizer": True,
                "runtime_loads_checkpoint_normalizer_first": True,
                "note": (
                    "FrozenRuntimeController.from_checkpoints uses actor_payload['normalizer'] "
                    "with fallback to jepa_payload['normalizer']; ActorEmbeddingDataset uses "
                    "load_command_normalizer (disk JSON fit on jepa train)."
                ),
            },
            "preprocessing_pipeline": {
                "actor_embedding_dataset": "PreprocessPipeline(training=False), stochastic=False",
                "runtime_closed_loop": "PreprocessPipeline(training=False) via FrozenRuntimeController",
                "jepa_training_augmentation": "not used for actor embeddings",
            },
        },
        "9_split_and_command_indexing": {
            "split_disjointness": _split_disjointness(train_dir, test_dir),
            "train_command_indexing": _verify_command_indexing(train_full, train_dir),
            "test_command_indexing": _verify_command_indexing(test_ds, test_dir),
            "control_teacher_type_expected": CONTROL_TEACHER_TYPE,
        },
        "10_checkpoint_integrity": _checkpoint_integrity(config, actor_payload, actor, device),
    }
    report["findings"] = _summarize_findings(report)
    return report


def _load_metrics_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import json

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summarize_findings(report: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    jepa = report["1_jepa_checkpoint_consistency"]
    if not jepa.get("uses_same_frozen_encoder_as_actor_training"):
        findings.append("JEPA checkpoint used at inference may differ from actor training record.")
    emb = report["2_embedding_distributions"]["train"]
    if emb["fraction_samples_with_duplicate_embedding"] > 0.5:
        findings.append("Train embeddings are highly duplicated (>50% samples share rounded embedding).")
    if emb["per_dimension_std"]["mean"] < 1e-4:
        findings.append("Train embedding per-dimension std is near zero (input collapse).")
    corr = report["5_correlation"]["train"]
    if corr["embedding_dim_max_abs_pearson_with_target"] < 0.05:
        findings.append("Weak embedding–target correlation (max |r| < 0.05).")
    if corr["pred_vs_target_pearson"] < 0.05:
        findings.append("Actor predictions weakly correlated with targets on train.")
    const = report["7_constant_mean_solution"]["test"]
    if const.get("detected_near_constant_mean_solution"):
        findings.append("Test predictions match near-constant-mean collapse heuristics.")
    if not report["8_preprocessing_runtime_parity"]["embedding_encode_context"].get("parity_ok", False):
        findings.append("ActorEmbeddingDataset vs runtime encode_context mismatch detected.")
    norm = report["8_preprocessing_runtime_parity"].get("normalizer", {})
    if norm and not norm.get("checkpoint_matches_disk", True):
        findings.append("Actor checkpoint normalizer differs from disk command_norm.json.")
    split = report["9_split_and_command_indexing"]["split_disjointness"]
    if not split.get("splits_disjoint", True):
        findings.append("Actor train/test trajectory IDs overlap.")
    idx = report["9_split_and_command_indexing"]["train_command_indexing"]
    if not idx.get("command_indexing_ok", True):
        findings.append("Command indexing does not match commands[time_index].")
    ckpt = report["10_checkpoint_integrity"]
    if not ckpt.get("ok", False):
        findings.append("Actor checkpoint weights do not match loaded model state.")
    if not ckpt.get("differs_from_random_init", True):
        findings.append("Actor checkpoint weights match fresh random initialization (not trained).")
    if not findings:
        findings.append("No obvious implementation bug; inspect metrics and constant-mean block for degenerate fit.")
    return findings


def audit_actor_training_from_runs(
    config: dict[str, Any],
    *,
    jepa_checkpoint: Path | str | None = None,
    actor_checkpoint: Path | str | None = None,
    seed: int | None = None,
    data_root: Path | None = None,
    device: torch.device | None = None,
    max_embedding_samples: int | None = None,
) -> dict[str, Any]:
    """
    Resolve default checkpoints from runs/.

    JEPA path always comes from the actor checkpoint metadata so evaluation
    cannot silently load seed_0/best.pt.
    """
    from ts_jepa.config import actor_run_dirname, project_root
    from ts_jepa.device import select_device
    from ts_jepa.evaluation.checkpoints import resolve_jepa_checkpoint_from_actor, resolve_run_checkpoint

    device = select_device(device)
    root = project_root(config)
    runs_root = root / config["paths"]["runs_root"]
    actor_ckpt = resolve_run_checkpoint(
        runs_root,
        actor_run_dirname(config),
        explicit=Path(actor_checkpoint) if actor_checkpoint else None,
        seed=seed,
    )
    jepa_ckpt = resolve_jepa_checkpoint_from_actor(actor_ckpt, project_dir=root)
    return audit_actor_training(
        config,
        jepa_checkpoint=jepa_ckpt,
        actor_checkpoint=actor_ckpt,
        data_root=data_root,
        device=device,
        max_embedding_samples=max_embedding_samples,
        seed=int(seed or 0),
    )
