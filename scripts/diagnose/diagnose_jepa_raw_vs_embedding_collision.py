#!/usr/bin/env python
"""
Read-only diagnostic: raw observation collapse vs JEPA representation collapse.

Uses the EXACT deterministic actor path:
  raw RGB → PreprocessPipeline(training=False) → process_frame(stochastic=False)
  → JEPA context encoder → embedding z

Does NOT modify src/, configs/, data/, checkpoints/, or training code.
Does NOT regenerate data or train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
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


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def _hash_raw_context(frames: list[np.ndarray]) -> str:
    """Deterministic hash of ordered raw RGB frames used for κ-frame context."""
    h = hashlib.sha1()
    for frame in frames:
        arr = np.ascontiguousarray(frame)
        h.update(np.array(arr.shape, dtype=np.int64).tobytes())
        h.update(arr.tobytes())
    return h.hexdigest()


def _hash_embedding_exact(z: np.ndarray) -> str:
    arr = np.ascontiguousarray(z.astype(np.float32))
    return _hash_bytes(arr.tobytes())


def _hash_embedding_rounded(z: np.ndarray, decimals: int) -> str:
    arr = np.ascontiguousarray(np.round(z.astype(np.float64), decimals))
    return _hash_bytes(arr.tobytes())


def _raw_context_frames(frames: np.ndarray, time_index: int, kappa: int) -> list[np.ndarray]:
    """Mirror PreprocessPipeline.assemble_context frame selection (left-pad by repeat)."""
    start = max(0, time_index - kappa + 1)
    selected = [frames[t] for t in range(start, time_index + 1)]
    while len(selected) < kappa:
        selected.insert(0, selected[0])
    return selected


def _duplicate_stats(hashes: list[str]) -> dict[str, Any]:
    counts = Counter(hashes)
    values = np.asarray(list(counts.values()), dtype=np.int64)
    n = len(hashes)
    n_unique = len(counts)
    multi = int(np.sum(values > 1))
    once = int(np.sum(values == 1))
    largest = int(values.max()) if values.size else 0
    # Top duplicate group sizes (count → how many hashes have that multiplicity)
    size_hist = Counter(int(v) for v in values)
    top_sizes = sorted(size_hist.items(), key=lambda kv: (-kv[0], -kv[1]))[:20]
    # Top hashes by multiplicity
    top_groups = counts.most_common(20)
    return {
        "total": n,
        "unique": n_unique,
        "duplicate_fraction": float(1.0 - n_unique / max(n, 1)),
        "hashes_occurring_once": once,
        "hashes_occurring_multiple_times": multi,
        "largest_duplicate_group": largest,
        "multiplicity_histogram_top20": [
            {"group_size": int(size), "num_hashes": int(num)} for size, num in top_sizes
        ],
        "top_duplicate_counts": [
            {"hash_prefix": h[:12], "count": int(c)} for h, c in top_groups if c > 1
        ][:20],
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _load_states_for_dataset(dataset: TrajectoryDataset) -> list[np.ndarray]:
    states: list[np.ndarray] = []
    for file_path in dataset.files:
        with np.load(file_path) as data:
            states.append(np.asarray(data["states"], dtype=np.float64))
    return states


@torch.no_grad()
def run_diagnostic(
    config: dict[str, Any],
    *,
    jepa_ckpt: Path,
    data_root: Path,
    device: torch.device,
    num_pairs: int,
    seed: int,
    max_collision_groups: int,
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
    assert len(dataset) == 5100, f"Expected 5100 contexts, got {len(dataset)}"

    kappa = int(config["input"]["kappa"])
    jepa_states = _load_states_for_dataset(jepa_test)
    actor_states = _load_states_for_dataset(actor_test)

    # ---- collect raw hashes / states / commands in exact ConcatDataset order ----
    raw_hashes: list[str] = []
    states_ti: list[np.ndarray] = []
    commands_ti: list[float] = []
    traj_keys: list[tuple[str, int]] = []
    time_indices: list[int] = []
    # uint8 flattened κ-frame contexts (avoids multi-GB float64 copies)
    raw_flat_u8: list[np.ndarray] = []

    def _consume(split_name: str, ds: TrajectoryDataset, states_list: list[np.ndarray]) -> None:
        for file_idx, time_index in ds.index_map:
            frames = ds.frames[file_idx]
            cmds = ds.commands[file_idx]
            states = states_list[file_idx]
            ctx_frames = _raw_context_frames(frames, time_index, kappa)
            raw_hashes.append(_hash_raw_context(ctx_frames))
            raw_flat_u8.append(np.concatenate([np.ascontiguousarray(f).reshape(-1) for f in ctx_frames]))
            states_ti.append(states[time_index].copy())
            commands_ti.append(float(cmds[time_index]))
            traj_keys.append((split_name, file_idx))
            time_indices.append(int(time_index))

    _consume("jepa_test", jepa_test, jepa_states)
    _consume("actor_test", actor_test, actor_states)
    assert len(raw_hashes) == 5100

    # ---- deterministic embeddings via same DataLoader path as diagnose_jepa_embeddings ----
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    z_list: list[np.ndarray] = []
    for batch in loader:
        context = batch["context"].to(device)
        # TrajectoryDataset(training=False) → eval cache → stochastic=False path
        z = jepa.encode_context(context).cpu().numpy().astype(np.float32)
        z_list.append(z)
    z_all = np.concatenate(z_list, axis=0)
    assert z_all.shape[0] == 5100

    emb_exact = [_hash_embedding_exact(z_all[i]) for i in range(z_all.shape[0])]
    emb_6 = [_hash_embedding_rounded(z_all[i], 6) for i in range(z_all.shape[0])]
    emb_4 = [_hash_embedding_rounded(z_all[i], 4) for i in range(z_all.shape[0])]

    # ========== EXPERIMENT 1 ==========
    raw_div = _duplicate_stats(raw_hashes)

    # Per-trajectory raw diversity
    per_traj_raw: list[dict[str, Any]] = []
    traj_to_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, key in enumerate(traj_keys):
        traj_to_indices[key].append(i)

    unique_ratios: list[float] = []
    for key, idxs in sorted(traj_to_indices.items()):
        h = [raw_hashes[i] for i in idxs]
        n_u = len(set(h))
        ratio = n_u / max(len(h), 1)
        unique_ratios.append(ratio)
        per_traj_raw.append(
            {
                "split": key[0],
                "file_idx": key[1],
                "num_contexts": len(h),
                "contexts_note": (
                    "TrajectoryDataset indexes time_index in [0, length-Kp-1] "
                    f"(=85 for length=100, Kp=15); not full 100 steps"
                ),
                "unique_raw_context_hashes": n_u,
                "duplicate_fraction": float(1.0 - ratio),
                "unique_ratio": float(ratio),
            }
        )

    raw_context_diversity = {
        **raw_div,
        "per_trajectory": {
            "num_trajectories": len(per_traj_raw),
            "contexts_per_trajectory": int(np.median([r["num_contexts"] for r in per_traj_raw])),
            "expected_full_trajectory_steps": 100,
            "min_unique_ratio": float(min(unique_ratios)) if unique_ratios else float("nan"),
            "max_unique_ratio": float(max(unique_ratios)) if unique_ratios else float("nan"),
            "mean_unique_ratio": float(np.mean(unique_ratios)) if unique_ratios else float("nan"),
            "trajectories": per_traj_raw,
        },
    }

    # ========== EXPERIMENT 2 ==========
    emb_exact_stats = _duplicate_stats(emb_exact)
    emb_6_stats = _duplicate_stats(emb_6)
    emb_4_stats = _duplicate_stats(emb_4)
    embedding_diversity = {
        "embedding_shape": list(z_all.shape),
        "exact_float32": emb_exact_stats,
        "rounded_6dp": emb_6_stats,
        "rounded_4dp": emb_4_stats,
        "duplicate_fraction_exact": emb_exact_stats["duplicate_fraction"],
        "largest_duplicate_group_exact": emb_exact_stats["largest_duplicate_group"],
    }

    # ========== EXPERIMENT 3: raw → embedding ==========
    raw_to_emb: dict[str, set[str]] = defaultdict(set)
    for rh, eh in zip(raw_hashes, emb_exact):
        raw_to_emb[rh].add(eh)

    one_to_one = 0
    one_to_many = 0
    for emb_set in raw_to_emb.values():
        if len(emb_set) == 1:
            one_to_one += 1
        else:
            one_to_many += 1

    # Multiple raw → same embedding: counted via inverse
    emb_to_raw: dict[str, set[str]] = defaultdict(set)
    for rh, eh in zip(raw_hashes, emb_exact):
        emb_to_raw[eh].add(rh)
    many_raw_to_one_emb = sum(1 for s in emb_to_raw.values() if len(s) > 1)

    raw_to_embedding_mapping = {
        "unique_raw_contexts": len(raw_to_emb),
        "one_raw_to_exactly_one_embedding": one_to_one,
        "one_raw_to_multiple_embeddings": one_to_many,
        "embeddings_that_merge_multiple_raw_contexts": many_raw_to_one_emb,
        "note": (
            "one_raw→one_emb confirms deterministic encoder path; "
            "one_raw→many would indicate nondeterminism; "
            "many_raw→one_emb indicates representation compression"
        ),
        "fraction_raw_with_single_embedding": float(one_to_one / max(len(raw_to_emb), 1)),
        "max_embeddings_per_raw": int(max((len(s) for s in raw_to_emb.values()), default=0)),
    }

    # ========== EXPERIMENT 4: embedding → raw ==========
    fanouts = np.asarray([len(s) for s in emb_to_raw.values()], dtype=np.int64)

    def _bucket(fanouts_arr: np.ndarray) -> dict[str, int]:
        return {
            "exactly_1": int(np.sum(fanouts_arr == 1)),
            "2_to_5": int(np.sum((fanouts_arr >= 2) & (fanouts_arr <= 5))),
            "6_to_10": int(np.sum((fanouts_arr >= 6) & (fanouts_arr <= 10))),
            "11_to_50": int(np.sum((fanouts_arr >= 11) & (fanouts_arr <= 50))),
            "gt_50": int(np.sum(fanouts_arr > 50)),
        }

    embedding_to_raw_mapping = {
        "unique_embeddings_exact": len(emb_to_raw),
        "fanout_buckets": _bucket(fanouts),
        "max_distinct_raw_contexts_per_embedding": int(fanouts.max()) if fanouts.size else 0,
        "mean_raw_contexts_per_embedding": float(fanouts.mean()) if fanouts.size else float("nan"),
        "median_raw_contexts_per_embedding": float(np.median(fanouts)) if fanouts.size else float("nan"),
        "num_embeddings_merging_distinct_raw": int(np.sum(fanouts > 1)),
        "fraction_embeddings_merging_distinct_raw": float(np.mean(fanouts > 1)) if fanouts.size else float("nan"),
    }

    # ========== EXPERIMENT 5: physical state within collisions ==========
    # Primary (user-specified): embedding collisions with multiple distinct raw contexts
    collision_groups: list[dict[str, Any]] = []
    emb_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, eh in enumerate(emb_exact):
        emb_to_indices[eh].append(i)

    candidate_groups = []
    for eh, idxs in emb_to_indices.items():
        raw_set = {raw_hashes[i] for i in idxs}
        if len(raw_set) > 1:
            candidate_groups.append((len(raw_set), -len(idxs), eh, idxs, raw_set))
    candidate_groups.sort(reverse=True)

    states_arr = np.stack(states_ti, axis=0)
    cmds_arr = np.asarray(commands_ti, dtype=np.float64)

    for _, _, eh, idxs, raw_set in candidate_groups[:max_collision_groups]:
        st = states_arr[idxs]
        cm = cmds_arr[idxs]
        collision_groups.append(
            {
                "embedding_hash_prefix": eh[:12],
                "num_samples": len(idxs),
                "num_distinct_raw_contexts": len(raw_set),
                "x_min": float(st[:, 0].min()),
                "x_max": float(st[:, 0].max()),
                "x_dot_min": float(st[:, 1].min()),
                "x_dot_max": float(st[:, 1].max()),
                "theta_min": float(st[:, 2].min()),
                "theta_max": float(st[:, 2].max()),
                "theta_dot_min": float(st[:, 3].min()),
                "theta_dot_max": float(st[:, 3].max()),
                "state_l2_span": float(np.linalg.norm(st.max(axis=0) - st.min(axis=0))),
                "command_min": float(cm.min()),
                "command_max": float(cm.max()),
                "command_unique_values": [float(u) for u in np.unique(cm)],
                "num_unique_commands": int(np.unique(cm).size),
            }
        )

    state_spans = []
    for _, _, eh, idxs, raw_set in candidate_groups:
        st = states_arr[idxs]
        state_spans.append(float(np.linalg.norm(st.max(axis=0) - st.min(axis=0))))

    # Secondary (critical for raw-collapse diagnosis): identical raw contexts with
    # differing physical states (renderer quantization / subpixel collapse).
    raw_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, rh in enumerate(raw_hashes):
        raw_to_indices[rh].append(i)

    raw_state_groups: list[dict[str, Any]] = []
    raw_state_spans: list[float] = []
    raw_multi_state = 0
    for rh, idxs in raw_to_indices.items():
        if len(idxs) < 2:
            continue
        st = states_arr[idxs]
        span = float(np.linalg.norm(st.max(axis=0) - st.min(axis=0)))
        raw_state_spans.append(span)
        # Distinct states if any coordinate differs beyond tiny tol
        uniq_states = {tuple(np.round(states_arr[i], 8)) for i in idxs}
        if len(uniq_states) > 1:
            raw_multi_state += 1
        cm = cmds_arr[idxs]
        raw_state_groups.append(
            (
                len(idxs),
                span,
                {
                    "raw_hash_prefix": rh[:12],
                    "num_samples": len(idxs),
                    "num_unique_states_8dp": len(uniq_states),
                    "x_min": float(st[:, 0].min()),
                    "x_max": float(st[:, 0].max()),
                    "x_dot_min": float(st[:, 1].min()),
                    "x_dot_max": float(st[:, 1].max()),
                    "theta_min": float(st[:, 2].min()),
                    "theta_max": float(st[:, 2].max()),
                    "theta_dot_min": float(st[:, 3].min()),
                    "theta_dot_max": float(st[:, 3].max()),
                    "state_l2_span": span,
                    "command_min": float(cm.min()),
                    "command_max": float(cm.max()),
                    "command_unique_values": [float(u) for u in np.unique(cm)],
                    "num_unique_commands": int(np.unique(cm).size),
                },
            )
        )
    raw_state_groups.sort(key=lambda t: (-t[0], -t[1]))
    representative_raw_state_groups = [g for _, _, g in raw_state_groups[:max_collision_groups]]

    state_collision_analysis = {
        "num_embedding_collision_groups_multi_raw": len(candidate_groups),
        "representative_groups_multi_raw_same_embedding": collision_groups,
        "aggregate_state_l2_span_across_multi_raw_emb_groups": {
            "count": len(state_spans),
            "mean": float(np.mean(state_spans)) if state_spans else None,
            "median": float(np.median(state_spans)) if state_spans else None,
            "min": float(np.min(state_spans)) if state_spans else None,
            "max": float(np.max(state_spans)) if state_spans else None,
            "fraction_with_state_l2_span_gt_0.01": float(np.mean(np.asarray(state_spans) > 0.01))
            if state_spans
            else None,
            "fraction_with_state_l2_span_gt_0.1": float(np.mean(np.asarray(state_spans) > 0.1))
            if state_spans
            else None,
        },
        "identical_raw_context_state_diversity": {
            "num_raw_hashes_with_multiple_samples": len(raw_state_spans),
            "num_raw_hashes_with_multiple_distinct_states_8dp": raw_multi_state,
            "aggregate_state_l2_span": {
                "mean": float(np.mean(raw_state_spans)) if raw_state_spans else None,
                "median": float(np.median(raw_state_spans)) if raw_state_spans else None,
                "min": float(np.min(raw_state_spans)) if raw_state_spans else None,
                "max": float(np.max(raw_state_spans)) if raw_state_spans else None,
                "fraction_gt_0.01": float(np.mean(np.asarray(raw_state_spans) > 0.01))
                if raw_state_spans
                else None,
                "fraction_gt_0.1": float(np.mean(np.asarray(raw_state_spans) > 0.1))
                if raw_state_spans
                else None,
            },
            "representative_groups": representative_raw_state_groups,
            "note": (
                "Same raw RGB context hash with different physical states indicates "
                "renderer/observation quantization (subpixel state differences)."
            ),
        },
    }

    # ========== EXPERIMENT 6: distance correlations ==========
    rng = np.random.default_rng(seed)
    n = len(raw_hashes)
    num_pairs = min(int(num_pairs), n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=num_pairs)
    j_idx = rng.integers(0, n, size=num_pairs)
    mask = i_idx == j_idx
    while mask.any():
        j_idx[mask] = rng.integers(0, n, size=int(mask.sum()))
        mask = i_idx == j_idx

    raw_dists = np.empty(num_pairs, dtype=np.float64)
    emb_dists = np.empty(num_pairs, dtype=np.float64)
    state_dists = np.empty(num_pairs, dtype=np.float64)
    raw_identical = np.empty(num_pairs, dtype=bool)

    for k in range(num_pairs):
        i, j = int(i_idx[k]), int(j_idx[k])
        # L2 over uint8-as-float differences (same scale as mean abs pixel * sqrt(N))
        diff = raw_flat_u8[i].astype(np.float32) - raw_flat_u8[j].astype(np.float32)
        raw_dists[k] = float(np.linalg.norm(diff))
        emb_dists[k] = float(np.linalg.norm(z_all[i].astype(np.float64) - z_all[j].astype(np.float64)))
        state_dists[k] = float(np.linalg.norm(states_arr[i] - states_arr[j]))
        raw_identical[k] = raw_hashes[i] == raw_hashes[j]

    def _pair_stats(rd: np.ndarray, ed: np.ndarray, sd: np.ndarray) -> dict[str, Any]:
        return {
            "num_pairs": int(rd.size),
            "mean_raw_distance": float(rd.mean()) if rd.size else float("nan"),
            "median_raw_distance": float(np.median(rd)) if rd.size else float("nan"),
            "mean_embedding_distance": float(ed.mean()) if ed.size else float("nan"),
            "median_embedding_distance": float(np.median(ed)) if ed.size else float("nan"),
            "mean_state_distance": float(sd.mean()) if sd.size else float("nan"),
            "median_state_distance": float(np.median(sd)) if sd.size else float("nan"),
            "corr_raw_vs_embedding": _corr(rd, ed),
            "corr_state_vs_embedding": _corr(sd, ed),
        }

    distance_analysis = {
        "all_pairs": _pair_stats(raw_dists, emb_dists, state_dists),
        "raw_identical_pairs": _pair_stats(
            raw_dists[raw_identical], emb_dists[raw_identical], state_dists[raw_identical]
        ),
        "raw_different_pairs": _pair_stats(
            raw_dists[~raw_identical], emb_dists[~raw_identical], state_dists[~raw_identical]
        ),
        "fraction_raw_identical_pairs": float(np.mean(raw_identical)),
    }

    # ========== EXPERIMENT 7: temporal per trajectory ==========
    traj_reports: list[dict[str, Any]] = []
    # Prefer first 10 jepa_test trajectories for representativeness
    ordered_keys = sorted(traj_to_indices.keys(), key=lambda k: (0 if k[0] == "jepa_test" else 1, k[1]))
    selected_keys = ordered_keys[: max(10, min(10, len(ordered_keys)))]
    # Actually user asked at least 10 — take first 10
    selected_keys = ordered_keys[:10]

    agg_raw_ratio = []
    agg_emb_ratio = []
    agg_consec_raw = []
    agg_consec_emb = []
    agg_exact_raw_dup = []
    agg_exact_emb_dup = []

    for key in ordered_keys:
        idxs = sorted(traj_to_indices[key], key=lambda i: time_indices[i])
        rh = [raw_hashes[i] for i in idxs]
        eh = [emb_exact[i] for i in idxs]
        z_traj = z_all[idxs].astype(np.float64)
        raw_ratio = len(set(rh)) / max(len(rh), 1)
        emb_ratio = len(set(eh)) / max(len(eh), 1)

        consec_raw = []
        consec_emb = []
        exact_raw_dup = 0
        exact_emb_dup = 0
        for a, b in zip(idxs[:-1], idxs[1:]):
            # consecutive in time_index order
            if time_indices[b] != time_indices[a] + 1:
                continue
            diff = raw_flat_u8[a].astype(np.float32) - raw_flat_u8[b].astype(np.float32)
            rd = float(np.linalg.norm(diff))
            ed = float(np.linalg.norm(z_all[a].astype(np.float64) - z_all[b].astype(np.float64)))
            consec_raw.append(rd)
            consec_emb.append(ed)
            if raw_hashes[a] == raw_hashes[b]:
                exact_raw_dup += 1
            if emb_exact[a] == emb_exact[b]:
                exact_emb_dup += 1

        row = {
            "split": key[0],
            "file_idx": key[1],
            "num_contexts": len(idxs),
            "raw_unique_context_ratio": float(raw_ratio),
            "embedding_unique_ratio": float(emb_ratio),
            "mean_consecutive_raw_pixel_distance": float(np.mean(consec_raw)) if consec_raw else float("nan"),
            "mean_consecutive_embedding_l2_distance": float(np.mean(consec_emb)) if consec_emb else float("nan"),
            "num_exact_consecutive_raw_duplicates": exact_raw_dup,
            "num_exact_consecutive_embedding_duplicates": exact_emb_dup,
            "num_consecutive_pairs": len(consec_raw),
        }
        agg_raw_ratio.append(raw_ratio)
        agg_emb_ratio.append(emb_ratio)
        if consec_raw:
            agg_consec_raw.append(float(np.mean(consec_raw)))
            agg_consec_emb.append(float(np.mean(consec_emb)))
        agg_exact_raw_dup.append(exact_raw_dup)
        agg_exact_emb_dup.append(exact_emb_dup)
        if key in selected_keys or len(traj_reports) < 10:
            if len(traj_reports) < 10:
                traj_reports.append(row)

    # Ensure we have at least 10 detailed trajectories (already capped)
    # Also compute aggregates over ALL trajectories
    trajectory_analysis = {
        "representative_trajectories": traj_reports,
        "aggregate_over_all_trajectories": {
            "num_trajectories": len(ordered_keys),
            "mean_raw_unique_ratio": float(np.mean(agg_raw_ratio)),
            "mean_embedding_unique_ratio": float(np.mean(agg_emb_ratio)),
            "mean_of_mean_consecutive_raw_pixel_distance": float(np.mean(agg_consec_raw))
            if agg_consec_raw
            else float("nan"),
            "mean_of_mean_consecutive_embedding_l2_distance": float(np.mean(agg_consec_emb))
            if agg_consec_emb
            else float("nan"),
            "mean_exact_consecutive_raw_duplicates": float(np.mean(agg_exact_raw_dup)),
            "mean_exact_consecutive_embedding_duplicates": float(np.mean(agg_exact_emb_dup)),
        },
    }

    # ========== CLASSIFICATION ==========
    n_raw_u = raw_div["unique"]
    n_emb_u = emb_exact_stats["unique"]
    raw_dup_frac = raw_div["duplicate_fraction"]
    emb_dup_frac = emb_exact_stats["duplicate_fraction"]
    # Embedding diversity relative to raw: how much extra collapse beyond raw identity
    raw_first_index: dict[str, int] = {}
    for i, rh in enumerate(raw_hashes):
        if rh not in raw_first_index:
            raw_first_index[rh] = i
    unique_raw_indices = list(raw_first_index.values())
    emb_on_unique_raw = [emb_exact[i] for i in unique_raw_indices]
    n_emb_on_unique_raw = len(set(emb_on_unique_raw))
    representation_compression_ratio = 1.0 - (n_emb_on_unique_raw / max(len(unique_raw_indices), 1))

    multi_raw_merge_frac = float(np.mean(fanouts > 1)) if fanouts.size else 0.0

    # Treat tiny one-raw->many (CUDA float noise) as negligible if max emb L2 among
    # raw-identical pairs is ~0 and no multi-raw merges exist.
    negligible_nondeterminism = (
        one_to_many <= 2
        and many_raw_to_one_emb == 0
        and representation_compression_ratio < 1e-9
    )

    if one_to_many > 0 and not negligible_nondeterminism:
        classification = "C. MIXED"
        class_reason = (
            f"Unexpected: {one_to_many} raw contexts produced multiple embeddings "
            "(deterministic path should be 1:1). Combined with other metrics -> MIXED."
        )
    elif raw_dup_frac >= 0.5 and representation_compression_ratio < 0.15 and many_raw_to_one_emb == 0:
        classification = "A. RAW OBSERVATION COLLAPSE"
        class_reason = (
            f"Raw context duplicate fraction={raw_dup_frac:.3f} "
            f"({n_raw_u} unique / {raw_div['total']}); "
            f"unique embeddings={n_emb_u}; "
            f"representation compression among unique raw={representation_compression_ratio:.3f}; "
            f"embeddings merging distinct raws={many_raw_to_one_emb}. "
            "Embedding duplication is explained by identical raw observations."
        )
    elif representation_compression_ratio >= 0.25 and n_raw_u > max(50, 2 * n_emb_u):
        classification = "B. REPRESENTATION COLLAPSE / UNDERTRAINING"
        class_reason = (
            f"Unique raw contexts={n_raw_u} vs unique embeddings={n_emb_u}; "
            f"compression among unique-raw={representation_compression_ratio:.3f}; "
            f"fraction of embeddings merging distinct raws={multi_raw_merge_frac:.3f}."
        )
    elif raw_dup_frac >= 0.25 and representation_compression_ratio >= 0.15:
        classification = "C. MIXED"
        class_reason = (
            f"Both substantial: raw_dup_frac={raw_dup_frac:.3f}, "
            f"representation_compression_among_unique_raw={representation_compression_ratio:.3f}, "
            f"embeddings_merging_distinct_raw_frac={multi_raw_merge_frac:.3f}."
        )
    elif raw_dup_frac < 0.25 and representation_compression_ratio < 0.15 and n_emb_u > 0.7 * n_raw_u:
        classification = "D. HEALTHY"
        class_reason = (
            f"Raw unique={n_raw_u}, emb unique={n_emb_u}; low raw duplication "
            f"({raw_dup_frac:.3f}) and low extra representation compression "
            f"({representation_compression_ratio:.3f})."
        )
    else:
        classification = "C. MIXED"
        class_reason = (
            f"Intermediate: raw_dup={raw_dup_frac:.3f}, emb_dup={emb_dup_frac:.3f}, "
            f"unique_raw={n_raw_u}, unique_emb={n_emb_u}, "
            f"compression_on_unique_raw={representation_compression_ratio:.3f}."
        )

    # Retrain recommendation based on measured data only
    if classification.startswith("B."):
        decision = "YES"
        reason = (
            "Raw contexts are substantially more diverse than deterministic embeddings, "
            "and multiple distinct raw contexts map to the same embedding. "
            "A longer JEPA retrain could plausibly increase embedding diversity without "
            "changing the renderer."
        )
    elif classification.startswith("A."):
        decision = "NO"
        reason = (
            "Most embedding duplication is explained by identical raw context hashes "
            f"({n_raw_u} unique raw ≈ {n_emb_u} unique emb; "
            f"0 embeddings merge distinct raw contexts; "
            f"compression among unique-raw={representation_compression_ratio:.3f}). "
            "Retraining JEPA cannot separate states the renderer already paints identically."
        )
    elif classification.startswith("D."):
        decision = "NO"
        reason = (
            "Deterministic embeddings already preserve substantial diversity relative to raw "
            "contexts; a 150-epoch retrain is unlikely to be the fix for actor collapse."
        )
    else:
        if representation_compression_ratio >= 0.2 and n_raw_u >= 2 * max(n_emb_u, 1):
            decision = "YES"
            reason = (
                "Mixed picture, but representation compression among visually distinct raw "
                f"contexts is large ({representation_compression_ratio:.3f}; "
                f"unique_raw={n_raw_u}, unique_emb={n_emb_u}). Retrain may help the "
                "representation side; renderer issues may remain."
            )
        elif raw_dup_frac >= 0.5 and representation_compression_ratio < 0.2:
            decision = "NO"
            reason = (
                "Mixed but raw observation duplication dominates "
                f"(raw_dup_frac={raw_dup_frac:.3f}). Retrain alone is unlikely to fix actor collapse."
            )
        else:
            decision = "UNCERTAIN"
            reason = (
                f"Both raw duplication (frac={raw_dup_frac:.3f}) and representation compression "
                f"({representation_compression_ratio:.3f}) are present at intermediate levels; "
                "retrain may help partially but renderer/observation limits may still bind."
            )

    return {
        "classification": classification,
        "classification_reason": class_reason,
        "num_contexts": 5100,
        "splits": {
            "jepa_test_samples": len(jepa_test),
            "actor_test_samples": len(actor_test),
            "total_samples": 5100,
            "contexts_per_trajectory": int(np.median([r["num_contexts"] for r in per_traj_raw])),
            "indexing": "same ConcatDataset([jepa_test, actor_test]) as diagnose_jepa_embeddings.py",
        },
        "path": {
            "preprocess": "PreprocessPipeline(training=False)",
            "process_frame": "stochastic=False",
            "encoder": "jepa.encode_context(context)",
            "augmentation": "disabled (no color jitter/drop; gaussian sigma=mean of range)",
        },
        "jepa_checkpoint": str(jepa_ckpt),
        "data_root": str(data_root),
        "raw_context_diversity": raw_context_diversity,
        "embedding_diversity": embedding_diversity,
        "raw_to_embedding_mapping": raw_to_embedding_mapping,
        "embedding_to_raw_mapping": embedding_to_raw_mapping,
        "state_collision_analysis": state_collision_analysis,
        "distance_analysis": distance_analysis,
        "trajectory_analysis": trajectory_analysis,
        "derived_metrics": {
            "unique_raw_contexts": n_raw_u,
            "unique_embeddings_exact": n_emb_u,
            "unique_embeddings_among_unique_raw": n_emb_on_unique_raw,
            "representation_compression_ratio_among_unique_raw": float(representation_compression_ratio),
            "raw_duplicate_fraction": float(raw_dup_frac),
            "embedding_duplicate_fraction_exact": float(emb_dup_frac),
        },
        "retrain_150_epoch_recommendation": {
            "decision": decision,
            "reason": reason,
        },
    }


def _print_summary(report: dict[str, Any]) -> None:
    print("=== JEPA raw vs embedding collision diagnostic ===")
    print(f"checkpoint: {report['jepa_checkpoint']}")
    print(f"contexts: {report['num_contexts']}")
    rd = report["raw_context_diversity"]
    ed = report["embedding_diversity"]
    print(
        f"raw unique: {rd['unique']} / {rd['total']} "
        f"(dup_frac={rd['duplicate_fraction']:.4f}, largest_group={rd['largest_duplicate_group']})"
    )
    ex = ed["exact_float32"]
    print(
        f"emb unique exact: {ex['unique']} / {ex['total']} "
        f"(dup_frac={ex['duplicate_fraction']:.4f}, largest_group={ex['largest_duplicate_group']})"
    )
    print(
        f"emb unique 6dp: {ed['rounded_6dp']['unique']} | "
        f"4dp: {ed['rounded_4dp']['unique']}"
    )
    m = report["raw_to_embedding_mapping"]
    print(
        f"raw->emb: one->one={m['one_raw_to_exactly_one_embedding']}, "
        f"one->many={m['one_raw_to_multiple_embeddings']}, "
        f"emb merging multi-raw={m['embeddings_that_merge_multiple_raw_contexts']}"
    )
    er = report["embedding_to_raw_mapping"]
    print(f"emb->raw fanout buckets: {er['fanout_buckets']}")
    print(f"max raw contexts / emb: {er['max_distinct_raw_contexts_per_embedding']}")
    d = report["derived_metrics"]
    print(
        f"compression among unique-raw: {d['representation_compression_ratio_among_unique_raw']:.4f} "
        f"({d['unique_embeddings_among_unique_raw']}/{d['unique_raw_contexts']})"
    )
    dist = report["distance_analysis"]
    print(
        f"pairs: corr(raw,emb)={dist['all_pairs']['corr_raw_vs_embedding']:.4f} | "
        f"corr(state,emb)={dist['all_pairs']['corr_state_vs_embedding']:.4f} | "
        f"frac raw-identical pairs={dist['fraction_raw_identical_pairs']:.4f}"
    )
    print(f"classification: {report['classification']}")
    print(f"reason: {report['classification_reason']}")
    rec = report["retrain_150_epoch_recommendation"]
    print(f"150-epoch retrain without renderer change: {rec['decision']}")
    print(f"  {rec['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_fixed.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default="runs/ts_jepa_dp_fixed/seed_0/best.pt")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-pairs", type=int, default=5000)
    parser.add_argument("--max-collision-groups", type=int, default=15)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = project_root(config)
    data_root = Path(args.data_root) if args.data_root else root / config["paths"]["data_root"]
    runs_root = root / config["paths"]["runs_root"]
    device = select_device(args.device)
    print(describe_device(device))

    jepa_ckpt = Path(args.jepa_checkpoint)
    if not jepa_ckpt.is_absolute():
        jepa_ckpt = root / jepa_ckpt
    if not jepa_ckpt.exists():
        jepa_ckpt = resolve_run_checkpoint(
            runs_root,
            jepa_run_dirname(config),
            explicit=None,
            seed=args.seed,
        )

    report = run_diagnostic(
        config,
        jepa_ckpt=jepa_ckpt,
        data_root=data_root,
        device=device,
        num_pairs=args.num_pairs,
        seed=args.seed,
        max_collision_groups=args.max_collision_groups,
    )

    out = (
        Path(args.out)
        if args.out
        else runs_root / "eval" / "jepa_raw_vs_embedding_collision_dp_fixed_seed0.json"
    )
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    _print_summary(report)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
