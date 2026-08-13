#!/usr/bin/env python
"""
Read-only Semantic Actor training-path diagnostics.

Reuses ActorEmbeddingDataset and train_actor batch construction without
modifying training code, checkpoints, or datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ts_jepa.config import actor_run_dirname, load_config, project_root
from ts_jepa.data.datasets import ActorEmbeddingDataset, load_command_normalizer
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.checkpoints import resolve_jepa_checkpoint_from_actor, resolve_run_checkpoint
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.training.actor_helpers import split_train_val_actor


def _tensor_stats(x: np.ndarray, *, num_pairs: int = 5000, seed: int = 0) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    rounded = np.round(x, 6)
    unique_rows = {tuple(row) for row in rounded}
    per_dim_std = x.std(axis=0)
    pair_stats = _pairwise_l2(x, num_pairs=num_pairs, seed=seed)
    return {
        "shape": list(x.shape),
        "num_samples": int(x.shape[0]),
        "global_mean": float(x.mean()),
        "global_std": float(x.std()),
        "mean_per_dim_std": float(per_dim_std.mean()),
        "median_per_dim_std": float(np.median(per_dim_std)),
        "min": float(x.min()),
        "max": float(x.max()),
        "num_unique_rounded_6dp": int(len(unique_rows)),
        "pairwise_l2": pair_stats,
    }


def _pairwise_l2(x: np.ndarray, *, num_pairs: int, seed: int) -> dict[str, Any]:
    n = x.shape[0]
    if n < 2:
        return {"num_pairs": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    rng = np.random.default_rng(seed)
    num_pairs = min(int(num_pairs), n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=num_pairs)
    j_idx = rng.integers(0, n, size=num_pairs)
    mask = i_idx == j_idx
    while mask.any():
        j_idx[mask] = rng.integers(0, n, size=int(mask.sum()))
        mask = i_idx == j_idx
    dists = np.linalg.norm(x[i_idx] - x[j_idx], axis=1)
    return {
        "num_pairs": int(num_pairs),
        "mean": float(dists.mean()),
        "median": float(np.median(dists)),
        "min": float(dists.min()),
        "max": float(dists.max()),
    }


def _collect_embeddings(
    dataset: ActorEmbeddingDataset,
    *,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(dataset)
    take = n if max_samples is None else min(n, max_samples)
    z_list: list[np.ndarray] = []
    norm_list: list[float] = []
    phys_list: list[float] = []
    normalizer = dataset.normalizer
    for i in range(take):
        item = dataset[i]
        z_list.append(item["embedding"].numpy())
        cmd_norm = float(item["command_norm"].reshape(-1)[0].item())
        norm_list.append(cmd_norm)
        phys_list.append(float(item["command"].reshape(-1)[0].item()))
    return np.stack(z_list, axis=0), np.asarray(norm_list), np.asarray(phys_list)


def _command_groups(z: np.ndarray, commands_phys: np.ndarray, *, zero_tol: float = 1e-6) -> dict[str, Any]:
    cmd = commands_phys.reshape(-1)
    masks = {
        "negative": cmd < -zero_tol,
        "zero": np.abs(cmd) <= zero_tol,
        "positive": cmd > zero_tol,
    }
    out: dict[str, Any] = {}
    means: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        if not mask.any():
            out[name] = {"count": 0, "embedding_mean": None, "embedding_std": None}
            continue
        g = z[mask]
        out[name] = {
            "count": int(mask.sum()),
            "embedding_mean": float(g.mean()),
            "embedding_std": float(g.std()),
        }
        means[name] = g.mean(axis=0)

    def _dist(a: str, b: str) -> float | None:
        if a not in means or b not in means:
            return None
        return float(np.linalg.norm(means[a] - means[b]))

    out["mean_vector_distances"] = {
        "negative_to_zero": _dist("negative", "zero"),
        "positive_to_zero": _dist("positive", "zero"),
        "positive_to_negative": _dist("positive", "negative"),
    }
    return out


def _correlation_diagnostic(z: np.ndarray, cmd_norm: np.ndarray) -> dict[str, Any]:
    y = cmd_norm.reshape(-1)
    corrs = []
    for d in range(z.shape[1]):
        x = z[:, d]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
    corrs_arr = np.asarray(corrs, dtype=np.float64)
    abs_corrs = np.abs(corrs_arr)
    top_idx = np.argsort(-abs_corrs)[:10]
    return {
        "max_abs_pearson": float(abs_corrs.max()),
        "mean_abs_pearson": float(abs_corrs.mean()),
        "top_10_dims_by_abs_correlation": [
            {"dim": int(i), "pearson": float(corrs_arr[i]), "abs_pearson": float(abs_corrs[i])}
            for i in top_idx
        ],
    }


def _build_sample_index(trajectory_dir: Path) -> list[dict[str, Any]]:
    """Mirror ActorEmbeddingDataset sample order for indexing checks."""
    meta: list[dict[str, Any]] = []
    files = sorted(trajectory_dir.glob("*.npz"))
    for file_idx, file_path in enumerate(files):
        with np.load(file_path) as data:
            n_steps = int(np.asarray(data["commands"]).shape[0])
        for time_index in range(n_steps):
            meta.append(
                {
                    "sample_index": len(meta),
                    "trajectory_id": file_path.stem,
                    "file_idx": file_idx,
                    "time_index": time_index,
                }
            )
    return meta


def _indexing_report(
    dataset: ActorEmbeddingDataset,
    trajectory_dir: Path,
    *,
    split_name: str,
    expected_trajectories: int,
) -> dict[str, Any]:
    meta = _build_sample_index(trajectory_dir)
    z_all, _, _ = _collect_embeddings(dataset)
    rounded = np.round(z_all, 6)
    unique_rows = {tuple(row) for row in rounded}

    # Context repetition: same embedding reused across samples
    row_to_indices: dict[tuple[float, ...], list[int]] = {}
    for i, row in enumerate(rounded):
        key = tuple(row)
        row_to_indices.setdefault(key, []).append(i)

    duplicate_groups = [idxs for idxs in row_to_indices.values() if len(idxs) > 1]
    max_reuse = max((len(g) for g in duplicate_groups), default=1)

    traj_ids = {m["trajectory_id"] for m in meta}
    time_indices = {m["time_index"] for m in meta}

    # Per-trajectory embedding uniqueness
    traj_unique_counts = []
    offset = 0
    files = sorted(trajectory_dir.glob("*.npz"))
    for file_path in files:
        with np.load(file_path) as data:
            n_steps = int(np.asarray(data["commands"]).shape[0])
        traj_emb = rounded[offset : offset + n_steps]
        offset += n_steps
        traj_unique_counts.append(int(len({tuple(r) for r in traj_emb})))

    return {
        "split": split_name,
        "num_samples": len(dataset),
        "num_unique_trajectory_ids": len(traj_ids),
        "expected_trajectory_files": expected_trajectories,
        "spans_all_expected_trajectories": len(traj_ids) == expected_trajectories,
        "num_unique_time_indices": len(time_indices),
        "first_20_samples": meta[:20],
        "num_unique_embeddings_rounded_6dp": int(len(unique_rows)),
        "fraction_samples_with_duplicate_embedding": float(
            1.0 - len(unique_rows) / max(len(dataset), 1)
        ),
        "max_samples_sharing_identical_embedding": int(max_reuse),
        "per_trajectory_unique_embedding_counts": {
            "min": int(min(traj_unique_counts) if traj_unique_counts else 0),
            "max": int(max(traj_unique_counts) if traj_unique_counts else 0),
            "mean": float(np.mean(traj_unique_counts) if traj_unique_counts else 0.0),
        },
        "all_contexts_within_trajectory_identical": bool(
            traj_unique_counts and max(traj_unique_counts) == 1
        ),
    }


def _forward_pass_report(
    actor: SemanticActor,
    z: np.ndarray,
    normalizer,
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    actor.eval()
    z_t = torch.from_numpy(z).to(device)
    with torch.no_grad():
        out_phys = actor(z_t).detach().cpu().numpy().reshape(-1)
    out_norm = normalizer.normalize(out_phys.astype(np.float32))

    rounded = np.unique(np.round(out_norm, 6))
    pair_in = _pairwise_l2(z, num_pairs=2000, seed=seed)

    # Pick pairs with large input distance
    rng = np.random.default_rng(seed)
    n = z.shape[0]
    pairs = []
    for _ in range(50):
        i, j = rng.integers(0, n, size=2)
        while i == j:
            j = int(rng.integers(0, n))
        dist = float(np.linalg.norm(z[i] - z[j]))
        if dist > 0.1:
            pairs.append((int(i), int(j), dist))
        if len(pairs) >= 10:
            break
    # fallback: farthest pairs
    if len(pairs) < 5:
        dists = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
        np.fill_diagonal(dists, -1.0)
        for _ in range(5):
            i, j = np.unravel_index(int(np.argmax(dists)), dists.shape)
            pairs.append((int(i), int(j), float(dists[i, j])))
            dists[i, j] = -1.0

    sensitivity_pairs = []
    for i, j, dist in pairs[:10]:
        sensitivity_pairs.append(
            {
                "input_l2_distance": dist,
                "output_norm_abs_diff": float(abs(out_norm[i] - out_norm[j])),
                "output_phys_abs_diff_N": float(abs(out_phys[i] - out_phys[j])),
            }
        )

    return {
        "input_stats": {
            "mean": float(z.mean()),
            "std": float(z.std()),
        },
        "output_normalized": {
            "mean": float(out_norm.mean()),
            "std": float(out_norm.std()),
            "min": float(out_norm.min()),
            "max": float(out_norm.max()),
            "num_unique_rounded_6dp": int(rounded.size),
        },
        "output_physical_N": {
            "mean": float(out_phys.mean()),
            "std": float(out_phys.std()),
            "min": float(out_phys.min()),
            "max": float(out_phys.max()),
        },
        "input_pairwise_l2": pair_in,
        "large_input_distance_output_pairs": sensitivity_pairs,
        "input_changes_produce_output_changes": bool(
            any(p["output_norm_abs_diff"] > 1e-6 for p in sensitivity_pairs)
        ),
    }


def _gradient_report(
    actor: SemanticActor,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    actor.train()
    for p in actor.parameters():
        if p.grad is not None:
            p.grad = None

    emb = batch["embedding"].to(device)
    target = batch["command"].to(device)
    pred = actor(emb)
    criterion = nn.MSELoss()
    loss = criterion(pred, target)
    loss.backward()

    per_param = []
    total_sq = 0.0
    any_nonzero = False
    for name, param in actor.named_parameters():
        grad = param.grad
        if grad is None:
            per_param.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "gradient": None,
                }
            )
            continue
        g = grad.detach().cpu().numpy()
        g_flat = g.reshape(-1)
        g_norm = float(np.linalg.norm(g_flat))
        total_sq += g_norm**2
        frac_zero = float(np.mean(g_flat == 0.0))
        nonzero = bool(np.any(g_flat != 0.0))
        any_nonzero = any_nonzero or nonzero
        per_param.append(
            {
                "name": name,
                "shape": list(param.shape),
                "param_mean": float(param.detach().cpu().numpy().mean()),
                "param_std": float(param.detach().cpu().numpy().std()),
                "gradient_mean": float(g_flat.mean()),
                "gradient_std": float(g_flat.std()),
                "gradient_min": float(g_flat.min()),
                "gradient_max": float(g_flat.max()),
                "gradient_l2_norm": g_norm,
                "fraction_zero_gradients": frac_zero,
                "has_nonzero_gradients": nonzero,
            }
        )

    return {
        "loss": float(loss.detach().item()),
        "batch_size": int(emb.shape[0]),
        "total_gradient_l2_norm": float(np.sqrt(total_sq)),
        "any_trainable_parameter_has_nonzero_gradients": any_nonzero,
        "parameters": per_param,
    }


def _controlled_sensitivity(
    actor: SemanticActor,
    z: np.ndarray,
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    actor.eval()
    z_ref = torch.from_numpy(z[0:1]).to(device)
    z_mean = torch.from_numpy(z.mean(axis=0, keepdims=True)).to(device)
    z_zero = torch.zeros_like(z_ref)
    rng = np.random.default_rng(seed)
    perturb = torch.from_numpy((rng.standard_normal(z.shape[1]) * 0.01).astype(np.float32))
    z_perturbed = z_ref + perturb.to(device)

    # Two real embeddings with large distance
    dists = np.linalg.norm(z - z[0:1], axis=1)
    far_idx = int(np.argmax(dists))
    z_far = torch.from_numpy(z[far_idx : far_idx + 1]).to(device)

    def _eval(t: torch.Tensor) -> float:
        with torch.no_grad():
            return float(actor(t).reshape(-1)[0].item())

    outputs = {
        "actor_z": _eval(z_ref),
        "actor_z_zero": _eval(z_zero),
        "actor_z_mean": _eval(z_mean),
        "actor_z_perturbed": _eval(z_perturbed),
        "actor_z_far": _eval(z_far),
    }
    diffs = {
        "z_vs_z_zero": abs(outputs["actor_z"] - outputs["actor_z_zero"]),
        "z_vs_z_mean": abs(outputs["actor_z"] - outputs["actor_z_mean"]),
        "z_vs_z_perturbed": abs(outputs["actor_z"] - outputs["actor_z_perturbed"]),
        "z_vs_z_far": abs(outputs["actor_z"] - outputs["actor_z_far"]),
        "input_l2_z_vs_z_far": float(np.linalg.norm(z[0] - z[far_idx])),
    }
    return {"outputs_normalized": outputs, "pairwise_output_abs_diffs": diffs}


def _architecture_health(actor: SemanticActor, z: np.ndarray, device: torch.device) -> dict[str, Any]:
    actor.eval()
    total_params = sum(p.numel() for p in actor.parameters())
    trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    param_values = np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in actor.parameters()])
    zero_frac = float(np.mean(param_values == 0.0))

    activations: dict[str, Any] = {}
    hooks = []
    z_t = torch.from_numpy(z[: min(256, z.shape[0])]).to(device)

    def _make_hook(name: str):
        def hook(_module, _inp, out):
            arr = out.detach().cpu().numpy()
            activations[name] = {
                "shape": list(arr.shape),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }

        return hook

    for i, layer in enumerate(actor.net):
        hooks.append(layer.register_forward_hook(_make_hook(f"layer_{i}_{layer.__class__.__name__}")))
    with torch.no_grad():
        out = actor(z_t)
    for h in hooks:
        h.remove()

    activations["final_output"] = {
        "mean": float(out.detach().cpu().numpy().mean()),
        "std": float(out.detach().cpu().numpy().std()),
        "min": float(out.detach().cpu().numpy().min()),
        "max": float(out.detach().cpu().numpy().max()),
    }

    return {
        "architecture": str(actor),
        "num_trainable_parameters": int(trainable),
        "total_parameters": int(total_params),
        "parameter_value_min": float(param_values.min()),
        "parameter_value_max": float(param_values.max()),
        "parameter_value_std": float(param_values.std()),
        "fraction_parameters_exactly_zero": zero_frac,
        "layer_activation_ranges": activations,
    }


def _classify(report: dict[str, Any]) -> str:
    inp = report["actor_input_diversity"]["combined_train_test"]
    grad = report["gradient_flow"]
    fwd = report["forward_pass"]
    idx = report["indexing"]["train"]
    corr = report["input_vs_target"]["correlation"]
    sens = report["controlled_sensitivity"]["pairwise_output_abs_diffs"]

    unique_frac = inp["num_unique_rounded_6dp"] / max(inp["num_samples"], 1)
    if unique_frac < 0.05 or inp["pairwise_l2"]["mean"] < 1e-4:
        return "A. ACTOR INPUT COLLAPSE"

    if not grad["any_trainable_parameter_has_nonzero_gradients"] or grad["total_gradient_l2_norm"] < 1e-12:
        return "B. ACTOR GRADIENT/OPTIMIZATION BUG"

    if not fwd["input_changes_produce_output_changes"] or fwd["output_normalized"]["num_unique_rounded_6dp"] <= 1:
        return "C. ACTOR FORWARD SENSITIVITY FAILURE"

    if not idx["spans_all_expected_trajectories"] or idx["all_contexts_within_trajectory_identical"]:
        return "D. DATA/INDEXING PROBLEM"

    if corr["max_abs_pearson"] < 0.1 and corr["mean_abs_pearson"] < 0.02:
        if sens["z_vs_z_far"] < 1e-4:
            return "C. ACTOR FORWARD SENSITIVITY FAILURE"
        return "E. ACTOR INPUT HAS WEAK CONTROL SIGNAL"

    return "F. NO OBVIOUS IMPLEMENTATION BUG"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_fixed.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
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

    actor_ckpt = resolve_run_checkpoint(
        runs_root,
        actor_run_dirname(config),
        explicit=Path(args.actor_checkpoint) if args.actor_checkpoint else None,
        seed=args.seed,
    )
    jepa_ckpt = resolve_jepa_checkpoint_from_actor(actor_ckpt, project_dir=root)

    jepa_payload = torch.load(jepa_ckpt, map_location=device, weights_only=False)
    actor_payload = torch.load(actor_ckpt, map_location=device, weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(jepa_payload["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    normalizer = load_command_normalizer(config, data_root=data_root)
    train_dir = data_root / "trajectories" / "actor" / "train"
    test_dir = data_root / "trajectories" / "actor" / "test"

    train_full = ActorEmbeddingDataset(train_dir, config, normalizer, jepa.context_encoder, device, training=True)
    test_ds = ActorEmbeddingDataset(test_dir, config, normalizer, jepa.context_encoder, device, training=False)

    z_train, cmd_norm_train, cmd_phys_train = _collect_embeddings(train_full)
    z_test, cmd_norm_test, cmd_phys_test = _collect_embeddings(test_ds)
    z_all = np.concatenate([z_train, z_test], axis=0)

    actor_cfg = config["semantic_actor"]["architecture"]
    actor = SemanticActor(
        embedding_dim=int(config["ts_jepa"]["encoder"]["embedding_dim"]),
        hidden_dims=tuple(actor_cfg["hidden_dims"]),
        dropout=float(actor_cfg["dropout"]),
    ).to(device)
    actor.load_state_dict(actor_payload["actor"])

    # Normalization targets from train split
    all_train_norm = cmd_norm_train
    all_train_phys = cmd_phys_train

    val_fraction = float(config["semantic_actor"]["early_stopping"].get("val_fraction", 0.2))
    train_ds, val_ds = split_train_val_actor(train_full, val_fraction)
    batch_size = int(config["semantic_actor"]["optimizer"]["batch_size"])
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, max(1, len(train_ds))),
        shuffle=True,
        generator=g,
        num_workers=0,
    )
    batch = next(iter(train_loader))

    report: dict[str, Any] = {
        "jepa_checkpoint": str(jepa_ckpt),
        "actor_checkpoint": str(actor_ckpt),
        "data_root": str(data_root),
        "actor_input_diversity": {
            "train": _tensor_stats(z_train, seed=args.seed),
            "test": _tensor_stats(z_test, seed=args.seed + 1),
            "combined_train_test": _tensor_stats(z_all, seed=args.seed + 2),
            "actor_receives_different_embeddings": bool(
                _tensor_stats(z_all, seed=args.seed)["num_unique_rounded_6dp"] > 1
                and _tensor_stats(z_all, seed=args.seed)["pairwise_l2"]["mean"] > 1e-6
            ),
        },
        "input_vs_target": {
            "command_groups": _command_groups(z_train, cmd_phys_train),
            "correlation": _correlation_diagnostic(z_train, cmd_norm_train),
        },
        "forward_pass": _forward_pass_report(actor, z_test, normalizer, device, seed=args.seed),
        "gradient_flow": _gradient_report(actor, batch, device),
        "controlled_sensitivity": _controlled_sensitivity(actor, z_test, device, seed=args.seed),
        "indexing": {
            "train": _indexing_report(
                train_full,
                train_dir,
                split_name="actor_train",
                expected_trajectories=int(config["semantic_actor"]["dataset"]["train_trajectories"]),
            ),
            "test": _indexing_report(
                test_ds,
                test_dir,
                split_name="actor_test",
                expected_trajectories=int(config["semantic_actor"]["dataset"]["test_trajectories"]),
            ),
        },
        "normalization": {
            "command_mean": float(normalizer.mean),
            "command_std": float(normalizer.std),
            "normalized_target_train_mean": float(all_train_norm.mean()),
            "normalized_target_train_std": float(all_train_norm.std()),
            "normalized_target_train_min": float(all_train_norm.min()),
            "normalized_target_train_max": float(all_train_norm.max()),
            "physical_target_train_std": float(all_train_phys.std()),
            "target_has_substantial_variance": bool(all_train_norm.std() > 0.1),
        },
        "architecture_health": _architecture_health(actor, z_test, device),
    }
    report["classification"] = _classify(report)

    out = (
        Path(args.out)
        if args.out
        else runs_root / "eval" / "actor_training_diagnostics_dp_fixed_seed0.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=== Actor training diagnostic summary ===")
    comb = report["actor_input_diversity"]["combined_train_test"]
    print(f"actor input shape: {comb['shape']} | unique (6dp): {comb['num_unique_rounded_6dp']}")
    print(f"input pairwise L2 mean: {comb['pairwise_l2']['mean']:.6g}")
    print(
        f"correlation max|pearson|={report['input_vs_target']['correlation']['max_abs_pearson']:.6g} "
        f"mean|pearson|={report['input_vs_target']['correlation']['mean_abs_pearson']:.6g}"
    )
    fwd = report["forward_pass"]["output_physical_N"]
    print(f"actor output physical std={fwd['std']:.6g} range=[{fwd['min']:.6g}, {fwd['max']:.6g}]")
    grad = report["gradient_flow"]
    print(
        f"gradient total L2={grad['total_gradient_l2_norm']:.6g} "
        f"any_nonzero={grad['any_trainable_parameter_has_nonzero_gradients']}"
    )
    print(f"classification: {report['classification']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
