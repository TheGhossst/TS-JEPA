#!/usr/bin/env python
"""
Read-only JEPA embedding diagnostics on real test trajectories.

Measures context-embedding diversity, cross-sample distances, command-group
separation, and horizon-wise predicted representation variation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from ts_jepa.config import jepa_run_dirname, load_config, project_root
from ts_jepa.data.datasets import TrajectoryDataset, load_command_normalizer
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.models.ts_jepa import TSJEPA

# #region agent log
def _agent_dbg(hypothesis_id: str, location: str, message: str, data: dict[str, Any], run_id: str = "pre-fix") -> None:
    import time

    payload = {
        "sessionId": "1367dc",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(r"c:\code\TS-JEPA\debug-1367dc.log", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
# #endregion


def _embedding_stats(z: np.ndarray, *, zero_std_tol: float = 1e-6) -> dict[str, Any]:
    """Global and per-dimension stats for embeddings [N, D]."""
    z = np.asarray(z, dtype=np.float64)
    per_dim_std = z.std(axis=0)
    rounded = np.round(z, 6)
    unique_rows = {tuple(row) for row in rounded}
    near_zero_dims = int(np.sum(per_dim_std < zero_std_tol))
    return {
        "shape": list(z.shape),
        "num_samples": int(z.shape[0]),
        "global_mean": float(z.mean()),
        "global_std": float(z.std()),
        "min": float(z.min()),
        "max": float(z.max()),
        "mean_per_dim_std": float(per_dim_std.mean()),
        "median_per_dim_std": float(np.median(per_dim_std)),
        "fraction_dims_near_zero_std": float(near_zero_dims / max(z.shape[1], 1)),
        "num_dims_near_zero_std": near_zero_dims,
        "num_unique_embeddings_rounded_6dp": int(len(unique_rows)),
        "effectively_identical_across_contexts": bool(len(unique_rows) <= 1),
    }


def _pairwise_distances(z: np.ndarray, num_pairs: int, seed: int) -> dict[str, Any]:
    """Random-pair L2 distances between embeddings [N, D]."""
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    if n < 2:
        return {
            "num_pairs": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    rng = np.random.default_rng(seed)
    num_pairs = min(int(num_pairs), n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=num_pairs)
    j_idx = rng.integers(0, n, size=num_pairs)
    mask = i_idx == j_idx
    while mask.any():
        j_idx[mask] = rng.integers(0, n, size=int(mask.sum()))
        mask = i_idx == j_idx
    dists = np.linalg.norm(z[i_idx] - z[j_idx], axis=1)
    return {
        "num_pairs": int(num_pairs),
        "mean": float(dists.mean()),
        "median": float(np.median(dists)),
        "min": float(dists.min()),
        "max": float(dists.max()),
    }


def _command_group_stats(
    z: np.ndarray,
    commands_phys: np.ndarray,
    *,
    zero_tol: float = 1e-6,
) -> dict[str, Any]:
    """Group embeddings by teacher command sign at context time."""
    z = np.asarray(z, dtype=np.float64)
    cmd = np.asarray(commands_phys, dtype=np.float64).reshape(-1)
    neg_mask = cmd < -zero_tol
    zero_mask = np.abs(cmd) <= zero_tol
    pos_mask = cmd > zero_tol

    def _group(name: str, mask: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return {"name": name, "num_samples": 0, "mean": None, "std": None}
        g = z[mask]
        return {
            "name": name,
            "num_samples": int(mask.sum()),
            "mean": float(g.mean()),
            "std": float(g.std()),
            "embedding_mean_vector_norm": float(np.linalg.norm(g.mean(axis=0))),
            "embedding_std_global": float(g.std()),
        }

    groups = {
        "negative": _group("negative", neg_mask),
        "zero": _group("zero", zero_mask),
        "positive": _group("positive", pos_mask),
    }

    def _mean_vec(key: str) -> np.ndarray | None:
        m = {"negative": neg_mask, "zero": zero_mask, "positive": pos_mask}[key]
        if not m.any():
            return None
        return z[m].mean(axis=0)

    mean_neg = _mean_vec("negative")
    mean_zero = _mean_vec("zero")
    mean_pos = _mean_vec("positive")

    def _dist(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
        if a is None or b is None:
            return None
        return float(np.linalg.norm(a - b))

    groups["mean_vector_distances"] = {
        "negative_to_zero": _dist(mean_neg, mean_zero),
        "positive_to_zero": _dist(mean_pos, mean_zero),
        "positive_to_negative": _dist(mean_pos, mean_neg),
    }
    return groups


def _temporal_prediction_stats(
    z_pred: np.ndarray,
    *,
    reference_horizon_1based: int = 1,
) -> dict[str, Any]:
    """
    z_pred: [N, Kp, D] with horizon h mapped to index h-1 (same as evaluate.py).
    """
    z_pred = np.asarray(z_pred, dtype=np.float64)
    n, kp, _ = z_pred.shape
    ref_idx = reference_horizon_1based - 1
    z_ref = z_pred[:, ref_idx : ref_idx + 1, :]
    per_h: list[dict[str, Any]] = []
    for h in range(1, kp + 1):
        idx = h - 1
        z_h = z_pred[:, idx, :]
        per_h.append(
            {
                "horizon_h": h,
                "std_global": float(z_h.std()),
                "mean_per_dim_std": float(z_h.std(axis=0).mean()),
                "mean_l2_distance_to_reference_horizon": float(
                    np.linalg.norm(z_h - z_pred[:, ref_idx, :], axis=1).mean()
                ),
            }
        )
    return {
        "convention": (
            "z_pred shape [N, Kp, D]; horizon h (1-based) uses z_pred[:, h-1, :]; "
            f"reference horizon for deltas is h={reference_horizon_1based}"
        ),
        "reference_horizon_1based": reference_horizon_1based,
        "per_horizon": per_h,
    }


def _input_sensitivity_samples(
    z: np.ndarray,
    commands_phys: np.ndarray,
    *,
    sample_ids: np.ndarray,
    max_per_group: int = 2,
    zero_tol: float = 1e-6,
) -> list[dict[str, Any]]:
    z = np.asarray(z, dtype=np.float64)
    cmd = np.asarray(commands_phys, dtype=np.float64).reshape(-1)
    ref_idx = 0
    ref_z = z[ref_idx]
    out: list[dict[str, Any]] = []

    def _add(idx: int) -> None:
        zi = z[idx]
        out.append(
            {
                "sample_index": int(sample_ids[idx]),
                "teacher_command_phys": float(cmd[idx]),
                "l2_distance_to_reference": float(np.linalg.norm(zi - ref_z)),
                "embedding_norm": float(np.linalg.norm(zi)),
                "embedding_mean": float(zi.mean()),
                "embedding_std": float(zi.std()),
            }
        )

    _add(ref_idx)
    for name, mask_fn in (
        ("negative", lambda c: c < -zero_tol),
        ("zero", lambda c: np.abs(c) <= zero_tol),
        ("positive", lambda c: c > zero_tol),
    ):
        candidates = np.where(mask_fn(cmd))[0]
        candidates = candidates[candidates != ref_idx]
        for idx in candidates[:max_per_group]:
            _add(int(idx))
    return out


def _effective_rank(z: np.ndarray) -> dict[str, Any]:
    """Covariance participation ratio; rank ~1 is directional collapse even if std>0."""
    z = np.asarray(z, dtype=np.float64)
    zc = z - z.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(zc, compute_uv=False)
    energy = singular ** 2
    total = float(energy.sum())
    if total <= 1e-12:
        return {
            "effective_rank": 0.0,
            "top1_var_frac": 1.0,
            "top5_var_frac": 1.0,
            "singular_top": 0.0,
            "singular_median": 0.0,
        }
    p = energy / total
    return {
        "effective_rank": float(1.0 / np.sum(p ** 2)),
        "top1_var_frac": float(p[0]),
        "top5_var_frac": float(p[: min(5, p.size)].sum()),
        "singular_top": float(singular[0]),
        "singular_median": float(np.median(singular)),
    }


def _pairwise_cosine(z: np.ndarray, num_pairs: int, seed: int) -> dict[str, Any]:
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    if n < 2:
        return {"num_pairs": 0, "mean": float("nan"), "median": float("nan")}
    rng = np.random.default_rng(seed)
    num_pairs = min(int(num_pairs), n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=num_pairs)
    j_idx = rng.integers(0, n, size=num_pairs)
    mask = i_idx == j_idx
    while mask.any():
        j_idx[mask] = rng.integers(0, n, size=int(mask.sum()))
        mask = i_idx == j_idx
    zi = z[i_idx]
    zj = z[j_idx]
    ni = np.linalg.norm(zi, axis=1)
    nj = np.linalg.norm(zj, axis=1)
    denom = np.clip(ni * nj, 1e-12, None)
    cos = np.sum(zi * zj, axis=1) / denom
    return {
        "num_pairs": int(num_pairs),
        "mean": float(cos.mean()),
        "median": float(np.median(cos)),
        "min": float(cos.min()),
        "max": float(cos.max()),
    }


def _predictor_command_sensitivity(
    jepa: TSJEPA,
    z: np.ndarray,
    normalizer: Any,
    device: torch.device,
    *,
    n: int = 32,
) -> dict[str, Any]:
    """Does P(z, u) change when u is 0 vs ±20 N? Tiny deltas => command is ignored."""
    n = min(int(n), int(z.shape[0]))
    z_t = torch.from_numpy(np.asarray(z[:n], dtype=np.float32)).to(device)
    kp = int(jepa.kp)

    def _pred(u_phys: float) -> np.ndarray:
        u_norm = normalizer.normalize(np.full((n, kp), u_phys, dtype=np.float32))
        with torch.no_grad():
            return jepa.predict(z_t, torch.from_numpy(u_norm).to(device)).cpu().numpy()

    z0 = _pred(0.0)
    zp = _pred(20.0)
    zn = _pred(-20.0)
    return {
        "n_contexts": n,
        "kp": kp,
        "mean_l2_zero_vs_plus20": float(np.linalg.norm(z0 - zp, axis=-1).mean()),
        "mean_l2_zero_vs_minus20": float(np.linalg.norm(z0 - zn, axis=-1).mean()),
        "mean_l2_plus20_vs_minus20": float(np.linalg.norm(zp - zn, axis=-1).mean()),
        "mean_cosine_plus20_vs_minus20": float(
            np.mean(
                np.sum(zp * zn, axis=-1)
                / np.clip(np.linalg.norm(zp, axis=-1) * np.linalg.norm(zn, axis=-1), 1e-12, None)
            )
        ),
        "std_pred_zero_cmd": float(z0.std()),
        "std_pred_plus20": float(zp.std()),
        "std_pred_minus20": float(zn.std()),
    }


def _classify(
    diversity: dict[str, Any],
    command_groups: dict[str, Any],
    cross_sample: dict[str, Any],
) -> str:
    if (
        diversity["effectively_identical_across_contexts"]
        or diversity["global_std"] < 1e-4
        or diversity["fraction_dims_near_zero_std"] > 0.95
    ):
        return "A) COLLAPSED REPRESENTATION"

    dists = command_groups.get("mean_vector_distances", {})
    d_vals = [v for v in dists.values() if v is not None]
    mean_cross = cross_sample.get("mean", float("nan"))
    group_sep = max(d_vals) if d_vals else 0.0

    # Control-relevant separation: group means should be meaningfully separated
    # relative to typical cross-sample distance.
    if np.isfinite(mean_cross) and mean_cross > 0 and group_sep < 0.05 * mean_cross:
        return "B) DIVERSE BUT POSSIBLY CONTROL-INSENSITIVE REPRESENTATION"
    if d_vals and group_sep > 0.01 * max(mean_cross, 1e-8):
        return "C) DIVERSE AND CONTROL-DIFFERENTIATED REPRESENTATION"
    return "B) DIVERSE BUT POSSIBLY CONTROL-INSENSITIVE REPRESENTATION"


@torch.no_grad()
def run_diagnostic(
    config: dict[str, Any],
    *,
    jepa_ckpt: Path,
    data_root: Path,
    device: torch.device,
    num_pairs: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    payload = torch.load(jepa_ckpt, map_location=device, weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(payload["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    normalizer = load_command_normalizer(config, data_root=data_root)
    jepa_test = TrajectoryDataset(
        data_root / "trajectories" / "jepa" / "test",
        config,
        normalizer,
        training=False,
    )
    actor_test = TrajectoryDataset(
        data_root / "trajectories" / "actor" / "test",
        config,
        normalizer,
        training=False,
    )
    dataset = ConcatDataset([jepa_test, actor_test])
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    z_list: list[np.ndarray] = []
    z_pred_list: list[np.ndarray] = []
    cmd_list: list[np.ndarray] = []
    sample_ids: list[int] = []
    offset = 0

    for batch in loader:
        context = batch["context"].to(device)
        teacher_norm = batch["teacher_commands_norm"].to(device)
        teacher_phys = batch["teacher_commands"][:, 0].cpu().numpy()
        z = jepa.encode_context(context).cpu().numpy()
        z_pred = jepa.predict(torch.from_numpy(z).to(device), teacher_norm).cpu().numpy()
        b = z.shape[0]
        z_list.append(z)
        z_pred_list.append(z_pred)
        cmd_list.append(teacher_phys)
        sample_ids.extend(range(offset, offset + b))
        offset += b

    z_all = np.concatenate(z_list, axis=0)
    z_pred_all = np.concatenate(z_pred_list, axis=0)
    cmd_all = np.concatenate(cmd_list, axis=0)
    sample_ids_arr = np.asarray(sample_ids, dtype=np.int64)

    diversity = _embedding_stats(z_all)
    rank_stats = _effective_rank(z_all)
    pairwise_cos = _pairwise_cosine(z_all, num_pairs=num_pairs, seed=seed)
    cross_sample = _pairwise_distances(z_all, num_pairs=num_pairs, seed=seed)
    command_groups = _command_group_stats(z_all, cmd_all)
    temporal = _temporal_prediction_stats(z_pred_all, reference_horizon_1based=1)
    pred_cmd = _predictor_command_sensitivity(jepa, z_all, normalizer, device, n=32)
    sensitivity = _input_sensitivity_samples(
        z_all,
        cmd_all,
        sample_ids=sample_ids_arr,
        max_per_group=2,
    )

    classification = _classify(diversity, command_groups, cross_sample)
    cmd = np.asarray(cmd_all, dtype=np.float64).reshape(-1)
    zero_frac = float(np.mean(np.abs(cmd) <= 1e-6))

    # #region agent log
    _agent_dbg(
        "H1",
        "diagnose_jepa_embeddings.py:run_diagnostic",
        "context embedding collapse stats",
        {
            "global_std": diversity["global_std"],
            "mean_per_dim_std": diversity["mean_per_dim_std"],
            "unique_6dp": diversity["num_unique_embeddings_rounded_6dp"],
            "effectively_identical": diversity["effectively_identical_across_contexts"],
            "frac_near_zero_std_dims": diversity["fraction_dims_near_zero_std"],
            "effective_rank": rank_stats["effective_rank"],
            "top1_var_frac": rank_stats["top1_var_frac"],
            "top5_var_frac": rank_stats["top5_var_frac"],
            "pairwise_cosine_mean": pairwise_cos["mean"],
            "cross_l2_mean": cross_sample["mean"],
            "classification": classification,
            "n": diversity["num_samples"],
        },
    )
    _agent_dbg(
        "H3",
        "diagnose_jepa_embeddings.py:command_groups",
        "command-group embedding separation and zero fraction",
        {
            "zero_fraction": zero_frac,
            "neg_n": command_groups["negative"]["num_samples"],
            "zero_n": command_groups["zero"]["num_samples"],
            "pos_n": command_groups["positive"]["num_samples"],
            "neg_zero_dist": command_groups["mean_vector_distances"]["negative_to_zero"],
            "pos_zero_dist": command_groups["mean_vector_distances"]["positive_to_zero"],
            "pos_neg_dist": command_groups["mean_vector_distances"]["positive_to_negative"],
        },
    )
    _agent_dbg(
        "H6",
        "diagnose_jepa_embeddings.py:predictor_sensitivity",
        "predictor output change under extreme commands",
        pred_cmd,
    )
    _agent_dbg(
        "H6",
        "diagnose_jepa_embeddings.py:temporal_pred",
        "horizon-wise predicted embedding variation",
        {
            "h1_std": temporal["per_horizon"][0]["std_global"],
            "h15_std": temporal["per_horizon"][-1]["std_global"],
            "h15_l2_to_h1": temporal["per_horizon"][-1]["mean_l2_distance_to_reference_horizon"],
        },
    )
    # #endregion

    return {
        "label": label,
        "jepa_checkpoint": str(jepa_ckpt),
        "data_root": str(data_root),
        "splits": {
            "jepa_test_samples": len(jepa_test),
            "actor_test_samples": len(actor_test),
            "total_samples": int(z_all.shape[0]),
        },
        "context_embedding_diversity": diversity,
        "embedding_effective_rank": rank_stats,
        "pairwise_cosine": pairwise_cos,
        "predictor_command_sensitivity": pred_cmd,
        "command_zero_fraction": zero_frac,
        "cross_sample_embedding_distance": cross_sample,
        "embedding_vs_command_groups": command_groups,
        "temporal_predictive_representation": temporal,
        "input_sensitivity_check": {
            "reference_sample_index": int(sample_ids_arr[0]),
            "reference_teacher_command_phys": float(cmd_all[0]),
            "samples": sensitivity,
        },
        "classification": classification,
    }


def _print_summary(report: dict[str, Any]) -> None:
    main = report["primary"]
    d = main["context_embedding_diversity"]
    c = main["cross_sample_embedding_distance"]
    g = main["embedding_vs_command_groups"]
    print("=== JEPA embedding diagnostic summary ===")
    print(f"checkpoint: {main['jepa_checkpoint']}")
    print(f"samples: {main['splits']['total_samples']} (jepa_test={main['splits']['jepa_test_samples']}, actor_test={main['splits']['actor_test_samples']})")
    print(f"embedding shape: {d['shape']}")
    print(f"global std: {d['global_std']:.6g} | mean per-dim std: {d['mean_per_dim_std']:.6g}")
    print(f"unique embeddings (6dp): {d['num_unique_embeddings_rounded_6dp']} | effectively identical: {d['effectively_identical_across_contexts']}")
    r = main.get("embedding_effective_rank", {})
    pc = main.get("pairwise_cosine", {})
    print(
        f"effective rank: {r.get('effective_rank')} | top1 var frac: {r.get('top1_var_frac')} | "
        f"pairwise cosine mean: {pc.get('mean')}"
    )
    print(f"command zero fraction: {main.get('command_zero_fraction')}")
    ps = main.get("predictor_command_sensitivity", {})
    print(
        "predictor Δ||z(+20)-z(-20)||: "
        f"{ps.get('mean_l2_plus20_vs_minus20')} | cosine(+20,-20)={ps.get('mean_cosine_plus20_vs_minus20')}"
    )
    print(f"cross-sample ||z_i-z_j|| mean/median: {c['mean']:.6g} / {c['median']:.6g}")
    dists = g["mean_vector_distances"]
    print(
        "group mean distances: "
        f"neg-zero={dists['negative_to_zero']}, "
        f"pos-zero={dists['positive_to_zero']}, "
        f"pos-neg={dists['positive_to_negative']}"
    )
    print("horizon std(z_pred[:,h-1,:]):")
    for row in main["temporal_predictive_representation"]["per_horizon"]:
        print(
            f"  h={row['horizon_h']:2d}: std={row['std_global']:.6g} "
            f"mean||z_h-z_ref||={row['mean_l2_distance_to_reference_horizon']:.6g}"
        )
    print(f"classification: {main['classification']}")
    if report.get("comparison"):
        old = report["comparison"]
        print(f"old checkpoint classification: {old['classification']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_fixed.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-pairs", type=int, default=5000)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--compare-old-checkpoint",
        type=str,
        default="runs/ts_jepa/seed_0/best.pt",
        help="Optional old checkpoint path; skipped if missing.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root(config)
    data_root = Path(args.data_root) if args.data_root else root / config["paths"]["data_root"]
    runs_root = root / config["paths"]["runs_root"]
    device = select_device(args.device)
    print(describe_device(device))

    jepa_ckpt = resolve_run_checkpoint(
        runs_root,
        jepa_run_dirname(config),
        explicit=Path(args.jepa_checkpoint) if args.jepa_checkpoint else None,
        seed=args.seed,
    )

    primary = run_diagnostic(
        config,
        jepa_ckpt=jepa_ckpt,
        data_root=data_root,
        device=device,
        num_pairs=args.num_pairs,
        seed=args.seed,
        label="dp_fixed",
    )

    report: dict[str, Any] = {"primary": primary, "comparison": None}
    old_path = Path(args.compare_old_checkpoint)
    if not old_path.is_absolute():
        old_path = root / old_path
    if old_path.exists():
        old_config = load_config("configs/ts_jepa_baseline.yaml")
        old_data_root = root / old_config["paths"]["data_root"]
        report["comparison"] = run_diagnostic(
            old_config,
            jepa_ckpt=old_path,
            data_root=old_data_root,
            device=device,
            num_pairs=args.num_pairs,
            seed=args.seed,
            label="old_baseline",
        )

    out = (
        Path(args.out)
        if args.out
        else runs_root / "eval" / "jepa_embedding_diagnostics_dp_fixed_seed0.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    _print_summary(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
