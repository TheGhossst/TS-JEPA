#!/usr/bin/env python
"""
Distinguish z-collapse observability bottleneck (A) vs encoder discarding signal (B).

Uses the exact ActorEmbeddingDataset path on actor/train:
  raw RGB -> PreprocessPipeline(training=False) -> process_frame(stochastic=False)
  -> assemble_context -> jepa.context_encoder -> z

Groups contexts by np.round(z, 6) (same precision as prior ~125-unique-z diagnostics).
Within same-z groups, flags groups with meaningfully differing teacher commands and
compares pixel / encoder-input distances against random non-colliding pairs.

Does NOT retrain JEPA or modify actor training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_jepa.config import jepa_run_dirname, load_config, project_root
from ts_jepa.data.datasets import load_command_normalizer
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.preprocessing.pipeline import PreprocessPipeline


Z_ROUND_DECIMALS = 6
CMD_NORM_DIFF_TOL = 1e-4  # normalized command range above this => "meaningfully different"


def _hash_raw_context(frames: list[np.ndarray]) -> str:
    h = hashlib.sha1()
    for frame in frames:
        arr = np.ascontiguousarray(frame)
        h.update(np.array(arr.shape, dtype=np.int64).tobytes())
        h.update(arr.tobytes())
    return h.hexdigest()


def _raw_context_frames(frames: np.ndarray, time_index: int, kappa: int) -> list[np.ndarray]:
    start = max(0, time_index - kappa + 1)
    selected = [frames[t] for t in range(start, time_index + 1)]
    while len(selected) < kappa:
        selected.insert(0, selected[0])
    return selected


def _z_group_key(z: np.ndarray, decimals: int = Z_ROUND_DECIMALS) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(z, dtype=np.float64), decimals))


def _grayscale_mean_image(frames: list[np.ndarray]) -> np.ndarray:
    """Mean grayscale over κ raw RGB uint8 frames, shape [H, W]."""
    stacked = np.stack(frames, axis=0).astype(np.float64)
    gray = 0.299 * stacked[:, :, :, 0] + 0.587 * stacked[:, :, :, 1] + 0.114 * stacked[:, :, :, 2]
    return gray.mean(axis=0)


def _ssim_global(img_a: np.ndarray, img_b: np.ndarray, *, c1: float = 6.5025, c2: float = 58.5225) -> float:
    """
    Global (single-window) SSIM on float images in [0, 255].
    Lightweight stand-in for skimage.metrics.structural_similarity (not a project dep).
    """
    a = np.asarray(img_a, dtype=np.float64)
    b = np.asarray(img_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"SSIM shape mismatch: {a.shape} vs {b.shape}")
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if den <= 0:
        return 1.0
    return float(num / den)


def _pairwise_stats(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "n_pairs": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    return {
        "n_pairs": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _histogram(values: np.ndarray, *, bins: int = 20) -> list[dict[str, float]]:
    if values.size == 0:
        return []
    counts, edges = np.histogram(values, bins=bins)
    return [
        {"bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]), "count": int(counts[i])}
        for i in range(len(counts))
    ]


@torch.no_grad()
def collect_actor_train_samples(
    *,
    train_dir: Path,
    config: dict[str, Any],
    normalizer,
    encoder: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Mirror ActorEmbeddingDataset sample order while retaining contexts and raw frames."""
    pipeline = PreprocessPipeline(config, training=False)
    kappa = int(config["input"]["kappa"])
    files = sorted(train_dir.glob("*.npz"))

    z_list: list[np.ndarray] = []
    command_norm_list: list[float] = []
    command_phys_list: list[float] = []
    raw_flat_u8: list[np.ndarray] = []
    raw_hashes: list[str] = []
    encoder_input_flat: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    encoder.eval()
    pending_contexts: list[torch.Tensor] = []
    pending_meta: list[dict[str, Any]] = []
    pending_raw_flat: list[np.ndarray] = []
    pending_raw_hash: list[str] = []
    pending_cmd_norm: list[float] = []
    pending_cmd_phys: list[float] = []
    pending_enc_flat: list[np.ndarray] = []

    def _flush_batch() -> None:
        nonlocal pending_contexts, pending_meta, pending_raw_flat, pending_raw_hash
        nonlocal pending_cmd_norm, pending_cmd_phys, pending_enc_flat
        if not pending_contexts:
            return
        chunk = torch.stack(pending_contexts, dim=0).to(device)
        emb = encoder(chunk).cpu().numpy().astype(np.float32)
        for i in range(emb.shape[0]):
            z_list.append(emb[i])
            command_norm_list.append(pending_cmd_norm[i])
            command_phys_list.append(pending_cmd_phys[i])
            raw_flat_u8.append(pending_raw_flat[i])
            raw_hashes.append(pending_raw_hash[i])
            encoder_input_flat.append(pending_enc_flat[i])
            meta.append(pending_meta[i])
        pending_contexts = []
        pending_meta = []
        pending_raw_flat = []
        pending_raw_hash = []
        pending_cmd_norm = []
        pending_cmd_phys = []
        pending_enc_flat = []

    bs = 64
    for file_idx, file_path in enumerate(files):
        with np.load(file_path) as data:
            frames = np.asarray(data["frames"])
            commands = np.asarray(data["commands"], dtype=np.float32)
        cached = [pipeline.process_frame(frames[t], stochastic=False) for t in range(len(commands))]
        for time_index in range(len(commands)):
            processed = {t: cached[t] for t in range(len(cached))}
            context = pipeline.assemble_context(processed, time_index, kappa=kappa)
            ctx_frames = _raw_context_frames(frames, time_index, kappa)
            cmd_phys = float(commands[time_index])
            cmd_norm = float(normalizer.normalize(np.array([cmd_phys], dtype=np.float32))[0])

            pending_contexts.append(context)
            pending_meta.append(
                {
                    "trajectory_id": file_path.stem,
                    "file_idx": file_idx,
                    "time_index": int(time_index),
                }
            )
            pending_raw_flat.append(
                np.concatenate([np.ascontiguousarray(f).reshape(-1) for f in ctx_frames])
            )
            pending_raw_hash.append(_hash_raw_context(ctx_frames))
            pending_cmd_norm.append(cmd_norm)
            pending_cmd_phys.append(cmd_phys)
            pending_enc_flat.append(context.detach().cpu().numpy().astype(np.float32).reshape(-1))

            if len(pending_contexts) >= bs:
                _flush_batch()
    _flush_batch()

    z_all = np.stack(z_list, axis=0)
    cmd_norm = np.asarray(command_norm_list, dtype=np.float64)
    cmd_phys = np.asarray(command_phys_list, dtype=np.float64)
    return {
        "z_all": z_all,
        "command_norm": cmd_norm,
        "command_phys": cmd_phys,
        "raw_flat_u8": raw_flat_u8,
        "raw_hashes": raw_hashes,
        "encoder_input_flat": encoder_input_flat,
        "meta": meta,
        "kappa": kappa,
        "num_samples": int(z_all.shape[0]),
    }


def analyze_z_command_groups(
    z_all: np.ndarray,
    command_norm: np.ndarray,
    command_phys: np.ndarray,
    *,
    decimals: int = Z_ROUND_DECIMALS,
    cmd_tol: float = CMD_NORM_DIFF_TOL,
) -> dict[str, Any]:
    groups: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for i in range(z_all.shape[0]):
        groups[_z_group_key(z_all[i], decimals)].append(i)

    multi_member = {k: idxs for k, idxs in groups.items() if len(idxs) > 1}
    diff_command_groups: dict[tuple[float, ...], list[int]] = {}
    group_command_stats: list[dict[str, Any]] = []

    for key, idxs in multi_member.items():
        cn = command_norm[idxs]
        cp = command_phys[idxs]
        cn_range = float(cn.max() - cn.min())
        cp_range = float(cp.max() - cp.min())
        n_unique_norm_6dp = int(len({round(float(v), 6) for v in cn}))
        n_unique_phys = int(len(np.unique(np.round(cp, 8))))
        differs = cn_range > cmd_tol
        row = {
            "group_size": len(idxs),
            "command_norm_range": cn_range,
            "command_norm_std": float(cn.std()),
            "command_phys_range": cp_range,
            "command_phys_std": float(cp.std()),
            "n_unique_command_norm_6dp": n_unique_norm_6dp,
            "n_unique_command_phys": n_unique_phys,
            "differs_by_tol": bool(differs),
        }
        group_command_stats.append(row)
        if differs:
            diff_command_groups[key] = idxs

    ranges = np.asarray([r["command_norm_range"] for r in group_command_stats], dtype=np.float64)
    diff_ranges = np.asarray(
        [r["command_norm_range"] for r in group_command_stats if r["differs_by_tol"]],
        dtype=np.float64,
    )

    return {
        "grouping_method": f"exact match on np.round(z, {decimals}) tuple",
        "grouping_justification": (
            f"Matches prior diagnostics (diagnose_jepa_embeddings.py, diagnose_actor_training.py) "
            f"that reported ~125 unique z at {decimals} decimal places."
        ),
        "command_diff_criterion": (
            f"max(command_norm) - min(command_norm) > {cmd_tol} within same-z group"
        ),
        "num_total_samples": int(z_all.shape[0]),
        "num_unique_z_groups": int(len(groups)),
        "num_groups_size_gt_1": int(len(multi_member)),
        "num_groups_size_gt_1_with_diff_commands": int(len(diff_command_groups)),
        "fraction_multi_groups_with_diff_commands": float(
            len(diff_command_groups) / max(len(multi_member), 1)
        ),
        "samples_in_diff_command_groups": int(sum(len(v) for v in diff_command_groups.values())),
        "command_norm_range_across_multi_groups": _pairwise_stats(ranges),
        "command_norm_range_diff_command_groups_only": _pairwise_stats(diff_ranges),
        "diff_command_groups": diff_command_groups,
        "group_command_stats_summary": {
            "multi_member_group_sizes": _pairwise_stats(
                np.asarray([len(v) for v in multi_member.values()], dtype=np.float64)
            ),
            "diff_command_group_sizes": _pairwise_stats(
                np.asarray([len(v) for v in diff_command_groups.values()], dtype=np.float64)
            ),
        },
    }


def compute_pair_distances(
    pairs: list[tuple[int, int]],
    *,
    raw_flat_u8: list[np.ndarray],
    encoder_input_flat: list[np.ndarray],
    raw_hashes: list[str],
    index_to_frames: dict[int, list[np.ndarray]] | None = None,
) -> dict[str, Any]:
    raw_l2: list[float] = []
    enc_l2: list[float] = []
    ssim_vals: list[float] = []
    raw_identical: list[bool] = []

    # Lazy cache for grayscale SSIM inputs
    gray_cache: dict[int, np.ndarray] = {}

    def _gray(i: int) -> np.ndarray:
        if i not in gray_cache:
            if index_to_frames is None:
                raise ValueError("index_to_frames required for SSIM")
            gray_cache[i] = _grayscale_mean_image(index_to_frames[i])
        return gray_cache[i]

    for i, j in pairs:
        diff_raw = raw_flat_u8[i].astype(np.float32) - raw_flat_u8[j].astype(np.float32)
        raw_l2.append(float(np.linalg.norm(diff_raw)))
        diff_enc = encoder_input_flat[i].astype(np.float64) - encoder_input_flat[j].astype(np.float64)
        enc_l2.append(float(np.linalg.norm(diff_enc)))
        raw_identical.append(raw_hashes[i] == raw_hashes[j])
        if index_to_frames is not None:
            ssim_vals.append(_ssim_global(_gray(i), _gray(j)))

    raw_arr = np.asarray(raw_l2, dtype=np.float64)
    enc_arr = np.asarray(enc_l2, dtype=np.float64)
    ssim_arr = np.asarray(ssim_vals, dtype=np.float64) if ssim_vals else np.asarray([], dtype=np.float64)
    raw_id = np.asarray(raw_identical, dtype=bool)

    out: dict[str, Any] = {
        "raw_pixel_l2": _pairwise_stats(raw_arr),
        "encoder_input_l2": _pairwise_stats(enc_arr),
        "fraction_raw_hash_identical": float(raw_id.mean()) if raw_id.size else float("nan"),
        "raw_pixel_l2_histogram": _histogram(raw_arr),
        "encoder_input_l2_histogram": _histogram(enc_arr),
    }
    if ssim_arr.size:
        out["raw_grayscale_ssim"] = _pairwise_stats(ssim_arr)
        out["raw_grayscale_ssim_histogram"] = _histogram(ssim_arr)
    if raw_id.any():
        out["raw_hash_identical_subset"] = {
            "raw_pixel_l2": _pairwise_stats(raw_arr[raw_id]),
            "encoder_input_l2": _pairwise_stats(enc_arr[raw_id]),
            "raw_grayscale_ssim": _pairwise_stats(ssim_arr[raw_id]) if ssim_arr.size else None,
        }
    if (~raw_id).any() and raw_id.size:
        out["raw_hash_different_subset"] = {
            "raw_pixel_l2": _pairwise_stats(raw_arr[~raw_id]),
            "encoder_input_l2": _pairwise_stats(enc_arr[~raw_id]),
            "raw_grayscale_ssim": _pairwise_stats(ssim_arr[~raw_id]) if ssim_arr.size else None,
        }
    return out


def build_index_to_frames(
    train_dir: Path,
    meta: list[dict[str, Any]],
    kappa: int,
) -> dict[int, list[np.ndarray]]:
    """Map global sample index -> raw κ-frame list (for SSIM only)."""
    files = sorted(train_dir.glob("*.npz"))
    frames_cache: dict[int, np.ndarray] = {}
    index_to_frames: dict[int, list[np.ndarray]] = {}
    for i, m in enumerate(meta):
        fi = int(m["file_idx"])
        if fi not in frames_cache:
            with np.load(files[fi]) as data:
                frames_cache[fi] = np.asarray(data["frames"])
        frames = frames_cache[fi]
        index_to_frames[i] = _raw_context_frames(frames, int(m["time_index"]), kappa)
    return index_to_frames


def sample_baseline_pairs(
    z_all: np.ndarray,
    *,
    target_n: int,
    seed: int,
    decimals: int = Z_ROUND_DECIMALS,
) -> list[tuple[int, int]]:
    """Random pairs from different z groups (non-colliding baseline)."""
    n = z_all.shape[0]
    keys = [_z_group_key(z_all[i], decimals) for i in range(n)]
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = max(target_n * 50, 10000)
    while len(pairs) < target_n and attempts < max_attempts:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        attempts += 1
        if i == j or keys[i] == keys[j]:
            continue
        a, b = (i, j) if i < j else (j, i)
        pairs.append((a, b))
    # Deduplicate while preserving order
    seen: set[tuple[int, int]] = set()
    unique_pairs: list[tuple[int, int]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            unique_pairs.append(p)
    return unique_pairs


def self_baseline_pairs(
    raw_flat_u8: list[np.ndarray],
    *,
    target_n: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Near-identical baseline: pair each sample with itself (L2=0 reference)."""
    n = len(raw_flat_u8)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(n, size=min(target_n, n), replace=False)
    return [(int(i), int(i)) for i in idxs]


@torch.no_grad()
def run_diagnostic(
    config: dict[str, Any],
    *,
    jepa_ckpt: Path,
    data_root: Path,
    device: torch.device,
    seed: int,
    max_pairs_per_group: int,
    cmd_tol: float,
) -> dict[str, Any]:
    payload = torch.load(jepa_ckpt, map_location=device, weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(payload["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)

    normalizer = load_command_normalizer(config, data_root=data_root)
    train_dir = data_root / "trajectories" / "actor" / "train"
    kappa = int(config["input"]["kappa"])
    in_channels = int(config["input"]["channels_per_rgb_frame"]) * kappa

    samples = collect_actor_train_samples(
        train_dir=train_dir,
        config=config,
        normalizer=normalizer,
        encoder=jepa.context_encoder,
        device=device,
    )
    z_all = samples["z_all"]
    group_report = analyze_z_command_groups(
        z_all,
        samples["command_norm"],
        samples["command_phys"],
        cmd_tol=cmd_tol,
    )
    diff_groups: dict[tuple[float, ...], list[int]] = group_report.pop("diff_command_groups")

    # All pairs within same-z / diff-command groups (cap per group for tractability)
    collision_pairs: list[tuple[int, int]] = []
    for idxs in diff_groups.values():
        group_pairs = list(combinations(idxs, 2))
        if len(group_pairs) > max_pairs_per_group:
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(group_pairs), size=max_pairs_per_group, replace=False)
            group_pairs = [group_pairs[int(k)] for k in pick]
        collision_pairs.extend(group_pairs)

    index_to_frames = build_index_to_frames(train_dir, samples["meta"], kappa)
    collision_dist = compute_pair_distances(
        collision_pairs,
        raw_flat_u8=samples["raw_flat_u8"],
        encoder_input_flat=samples["encoder_input_flat"],
        raw_hashes=samples["raw_hashes"],
        index_to_frames=index_to_frames,
    )

    target_n = max(len(collision_pairs), 1)
    baseline_pairs = sample_baseline_pairs(z_all, target_n=target_n, seed=seed)
    baseline_dist = compute_pair_distances(
        baseline_pairs,
        raw_flat_u8=samples["raw_flat_u8"],
        encoder_input_flat=samples["encoder_input_flat"],
        raw_hashes=samples["raw_hashes"],
        index_to_frames=index_to_frames,
    )

    self_pairs = self_baseline_pairs(samples["raw_flat_u8"], target_n=min(1000, len(samples["raw_flat_u8"])), seed=seed)
    self_dist = compute_pair_distances(
        self_pairs,
        raw_flat_u8=samples["raw_flat_u8"],
        encoder_input_flat=samples["encoder_input_flat"],
        raw_hashes=samples["raw_hashes"],
        index_to_frames=index_to_frames,
    )

    # Summary table for user
    def _tbl(dist: dict[str, Any]) -> dict[str, float | int]:
        return {
            "mean_pixel_l2": dist["raw_pixel_l2"]["mean"],
            "mean_pixel_ssim": dist.get("raw_grayscale_ssim", {}).get("mean", float("nan")),
            "mean_encoder_input_l2": dist["encoder_input_l2"]["mean"],
            "n_pairs": dist["raw_pixel_l2"]["n_pairs"],
        }

    summary_table = {
        "same_z_diff_command_pairs": _tbl(collision_dist),
        "random_non_colliding_pairs_baseline": _tbl(baseline_dist),
        "self_pairs_identical_reference": _tbl(self_dist),
    }

    # Interpretation heuristics
    coll_l2 = collision_dist["raw_pixel_l2"]["mean"]
    base_l2 = baseline_dist["raw_pixel_l2"]["mean"]
    self_l2 = self_dist["raw_pixel_l2"]["mean"]
    coll_ssim = collision_dist.get("raw_grayscale_ssim", {}).get("mean", float("nan"))
    base_ssim = baseline_dist.get("raw_grayscale_ssim", {}).get("mean", float("nan"))

    if np.isfinite(coll_l2) and np.isfinite(base_l2) and base_l2 > 0:
        l2_ratio = coll_l2 / base_l2
    else:
        l2_ratio = float("nan")

    if l2_ratio < 0.05 or (np.isfinite(coll_ssim) and coll_ssim > 0.995):
        verdict = "A_observability_bottleneck"
        verdict_detail = (
            "Same-z/diff-command pairs have near-zero raw pixel L2 (or SSIM~1), "
            "comparable to self-pairs — physically distinct states are already visually "
            "indistinguishable in the k-frame raw render."
        )
    elif l2_ratio > 0.5 and collision_dist["fraction_raw_hash_identical"] < 0.5:
        verdict = "B_encoder_discards_recoverable_signal"
        verdict_detail = (
            "Same-z/diff-command pairs have raw pixel L2 comparable to random non-colliding "
            "baseline while z still collides — pixels differ but encoder maps them to the same z."
        )
    else:
        verdict = "mixed_or_intermediate"
        verdict_detail = (
            "Pixel distances fall between identical-reference and random-baseline regimes, "
            "or many collisions share identical raw hashes — both observability limits and "
            "encoder compression may contribute."
        )

    # Step 4: kappa sensitivity — encoder in_channels fixed at training kappa
    kappa_secondary = {
        "tested": False,
        "reason": (
            f"Frozen context_encoder expects in_channels={in_channels} "
            f"(kappa={kappa} × 3 RGB). assemble_context accepts kappa overrides, but "
            "feeding kappa>2 would change channel count and cannot run through this "
            "checkpoint without retraining. Skipped."
        ),
        "current_kappa": kappa,
        "encoder_in_channels": in_channels,
    }

    # Representative diff-command collision examples
    examples: list[dict[str, Any]] = []
    sorted_groups = sorted(diff_groups.items(), key=lambda kv: -len(kv[1]))
    for key, idxs in sorted_groups[:10]:
        cn = samples["command_norm"][idxs]
        cp = samples["command_phys"][idxs]
        raw_set = {samples["raw_hashes"][i] for i in idxs}
        examples.append(
            {
                "group_size": len(idxs),
                "z_prefix": [float(x) for x in key[:4]],
                "num_distinct_raw_hashes": len(raw_set),
                "command_norm_range": float(cn.max() - cn.min()),
                "command_phys_range": float(cp.max() - cp.min()),
                "sample_indices": idxs[:8],
            }
        )

    return {
        "jepa_checkpoint": str(jepa_ckpt),
        "data_root": str(data_root),
        "train_dir": str(train_dir),
        "path_note": (
            "Identical to ActorEmbeddingDataset / linear_probe_force.py: "
            "PreprocessPipeline(training=False), process_frame(stochastic=False), "
            "assemble_context, jepa.context_encoder"
        ),
        "num_samples": samples["num_samples"],
        "kappa": kappa,
        "z_grouping": {
            k: v
            for k, v in group_report.items()
            if k != "group_command_stats_summary"
        },
        "group_command_stats_summary": group_report["group_command_stats_summary"],
        "collision_pair_analysis": {
            "num_diff_command_groups": len(diff_groups),
            "num_pairs_sampled": len(collision_pairs),
            "max_pairs_per_group_cap": max_pairs_per_group,
            "distances": collision_dist,
        },
        "baseline_pair_analysis": {
            "num_pairs_sampled": len(baseline_pairs),
            "distances": baseline_dist,
        },
        "self_reference_analysis": {
            "distances": self_dist,
        },
        "summary_table": summary_table,
        "interpretation": {
            "pixel_l2_ratio_collision_over_baseline": float(l2_ratio),
            "verdict": verdict,
            "detail": verdict_detail,
        },
        "kappa_secondary_check": kappa_secondary,
        "representative_diff_command_groups": examples,
    }


def _print_report(report: dict[str, Any]) -> None:
    print("=== Z-collapse observability vs encoder diagnostic ===")
    print(f"checkpoint: {report['jepa_checkpoint']}")
    print(f"samples: {report['num_samples']} (actor/train, all timesteps)")
    zg = report["z_grouping"]
    print(
        f"unique z groups (round 6dp): {zg['num_unique_z_groups']} | "
        f"groups size>1: {zg['num_groups_size_gt_1']} | "
        f"size>1 with diff commands: {zg['num_groups_size_gt_1_with_diff_commands']}"
    )
    print(f"criterion: {zg['command_diff_criterion']}")
    print()
    print("| | same-z diff-cmd | random baseline | self (identical) |")
    print("|---|---|---|---|")
    t = report["summary_table"]
    for label, key in [
        ("mean pixel L2", "mean_pixel_l2"),
        ("mean pixel SSIM", "mean_pixel_ssim"),
        ("mean encoder-input L2", "mean_encoder_input_l2"),
        ("n pairs", "n_pairs"),
    ]:
        a = t["same_z_diff_command_pairs"][key]
        b = t["random_non_colliding_pairs_baseline"][key]
        c = t["self_pairs_identical_reference"][key]
        if isinstance(a, float):
            print(f"| {label} | {a:.4g} | {b:.4g} | {c:.4g} |")
        else:
            print(f"| {label} | {a} | {b} | {c} |")
    print()
    interp = report["interpretation"]
    print(f"pixel L2 ratio (collision/baseline): {interp['pixel_l2_ratio_collision_over_baseline']:.4f}")
    print(f"verdict: {interp['verdict']}")
    print(f"  {interp['detail']}")
    print()
    k = report["kappa_secondary_check"]
    print(f"kappa secondary check: tested={k['tested']}")
    print(f"  {k['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_fixed.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cmd-tol", type=float, default=CMD_NORM_DIFF_TOL)
    parser.add_argument("--max-pairs-per-group", type=int, default=500)
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

    report = run_diagnostic(
        config,
        jepa_ckpt=jepa_ckpt,
        data_root=data_root,
        device=device,
        seed=args.seed,
        max_pairs_per_group=args.max_pairs_per_group,
        cmd_tol=args.cmd_tol,
    )

    out = (
        Path(args.out)
        if args.out
        else runs_root / "eval" / "z_collapse_observability_dp_fixed_seed0.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    _print_report(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
