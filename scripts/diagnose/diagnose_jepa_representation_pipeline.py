#!/usr/bin/env python
"""
Read-only JEPA representation pipeline diagnostics.

Traces raw frames → context construction → encoder embeddings to locate
where representation repetition originates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ts_jepa.config import jepa_run_dirname, load_config, project_root
from ts_jepa.data.datasets import TrajectoryDataset, load_command_normalizer
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.preprocessing.pipeline import PreprocessPipeline

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


def _hash_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:12]


def _hash_array(arr: np.ndarray, *, decimals: int | None = None) -> str:
    x = np.asarray(arr)
    if decimals is not None:
        x = np.round(x, decimals)
    return _hash_bytes(x.tobytes())


def _consecutive_diffs(frames: np.ndarray) -> np.ndarray:
    if frames.shape[0] < 2:
        return np.array([], dtype=np.float64)
    a = frames[:-1].astype(np.float64)
    b = frames[1:].astype(np.float64)
    return np.mean(np.abs(b - a), axis=(1, 2, 3))


def _raw_frame_diversity(frames: np.ndarray, trajectory_id: str) -> dict[str, Any]:
    diffs = _consecutive_diffs(frames)
    exact_hashes = {_hash_array(frames[t]) for t in range(frames.shape[0])}
    rounded_hashes = {_hash_array(frames[t], decimals=0) for t in range(frames.shape[0])}
    downsampled = frames[:, ::4, ::4, :]
    ds_hashes = {_hash_array(downsampled[t], decimals=0) for t in range(downsampled.shape[0])}
    return {
        "trajectory_id": trajectory_id,
        "frame_shape": list(frames.shape[1:]),
        "dtype": str(frames.dtype),
        "min": int(frames.min()),
        "max": int(frames.max()),
        "num_frames": int(frames.shape[0]),
        "unique_raw_frames_exact": len(exact_hashes),
        "unique_raw_frames_rounded_int": len(rounded_hashes),
        "unique_raw_frames_downsampled_rounded": len(ds_hashes),
        "mean_consecutive_frame_difference": float(diffs.mean()) if diffs.size else 0.0,
        "median_consecutive_frame_difference": float(np.median(diffs)) if diffs.size else 0.0,
        "min_consecutive_frame_difference": float(diffs.min()) if diffs.size else 0.0,
        "max_consecutive_frame_difference": float(diffs.max()) if diffs.size else 0.0,
    }


def _build_contexts_for_trajectory(
    frames: np.ndarray,
    pipeline: PreprocessPipeline,
) -> list[torch.Tensor]:
    processed = [pipeline.process_frame(frames[t], stochastic=False) for t in range(frames.shape[0])]
    proc_dict = {t: processed[t] for t in range(len(processed))}
    contexts = []
    for time_index in range(frames.shape[0]):
        contexts.append(pipeline.assemble_jepa_frame(proc_dict, time_index))
    return contexts


def _context_diversity(contexts: list[torch.Tensor], trajectory_id: str) -> dict[str, Any]:
    arrs = [c.numpy() for c in contexts]
    rounded6 = [_hash_array(a, decimals=6) for a in arrs]
    unique6 = len(set(rounded6))
    dup_count = len(rounded6) - unique6

    adj = []
    for i in range(1, len(arrs)):
        adj.append(float(np.linalg.norm(arrs[i] - arrs[i - 1])))
    adj_arr = np.asarray(adj, dtype=np.float64) if adj else np.array([0.0])

    rng = np.random.default_rng(0)
    rand = []
    n = len(arrs)
    for _ in range(min(500, max(1, n * (n - 1) // 2))):
        i, j = rng.integers(0, n, size=2)
        while i == j:
            j = int(rng.integers(0, n))
        rand.append(float(np.linalg.norm(arrs[i] - arrs[j])))
    rand_arr = np.asarray(rand, dtype=np.float64)

    return {
        "trajectory_id": trajectory_id,
        "context_tensor_shape": list(contexts[0].shape) if contexts else None,
        "num_contexts": len(contexts),
        "unique_contexts_rounded_6dp": unique6,
        "duplicated_contexts": dup_count,
        "mean_l2_adjacent_contexts": float(adj_arr.mean()),
        "median_l2_adjacent_contexts": float(np.median(adj_arr)),
        "mean_l2_random_context_pairs": float(rand_arr.mean()),
        "min_l2_adjacent_contexts": float(adj_arr.min()),
        "max_l2_adjacent_contexts": float(adj_arr.max()),
    }


def _encode_trajectory(
    jepa: TSJEPA,
    contexts: list[torch.Tensor],
    commands: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    z_list = []
    table = []
    prev_z = None
    with torch.no_grad():
        for t, ctx in enumerate(contexts):
            z = jepa.encode_context(ctx.unsqueeze(0).to(device)).cpu().numpy()[0]
            z_list.append(z)
            delta = None if prev_z is None else float(np.linalg.norm(z - prev_z))
            table.append(
                {
                    "t": t,
                    "teacher_u_t": float(commands[t]),
                    "context_hash": _hash_array(ctx.numpy(), decimals=6),
                    "embedding_hash_6dp": _hash_array(z, decimals=6),
                    "embedding_hash_4dp": _hash_array(z, decimals=4),
                    "embedding_hash_3dp": _hash_array(z, decimals=3),
                    "l2_to_prev_embedding": delta,
                }
            )
            prev_z = z
    return np.stack(z_list, axis=0), table


def _embedding_uniqueness(z: np.ndarray) -> dict[str, Any]:
    deltas = []
    for i in range(1, z.shape[0]):
        deltas.append(float(np.linalg.norm(z[i] - z[i - 1])))
    d = np.asarray(deltas, dtype=np.float64) if deltas else np.array([0.0])
    return {
        "unique_embeddings_6dp": len({_hash_array(z[t], decimals=6) for t in range(z.shape[0])}),
        "unique_embeddings_4dp": len({_hash_array(z[t], decimals=4) for t in range(z.shape[0])}),
        "unique_embeddings_3dp": len({_hash_array(z[t], decimals=3) for t in range(z.shape[0])}),
        "consecutive_l2_mean": float(d.mean()),
        "consecutive_l2_median": float(np.median(d)),
        "consecutive_l2_min": float(d.min()),
        "consecutive_l2_max": float(d.max()),
    }


def _find_transition_examples(
    frames: np.ndarray,
    commands: np.ndarray,
    contexts: list[torch.Tensor],
    z: np.ndarray,
    *,
    max_per_type: int = 2,
) -> list[dict[str, Any]]:
    def _sign(u: float) -> str:
        if u < -1e-6:
            return "neg"
        if u > 1e-6:
            return "pos"
        return "zero"

    wanted = [
        ("negative_to_zero", "neg", "zero"),
        ("zero_to_positive", "zero", "pos"),
        ("positive_to_zero", "pos", "zero"),
        ("negative_to_positive", "neg", "pos"),
    ]
    found: dict[str, int] = {k: 0 for k, _, _ in wanted}
    examples: list[dict[str, Any]] = []

    for t in range(1, len(commands) - 1):
        s_prev, s_cur, s_next = _sign(commands[t - 1]), _sign(commands[t]), _sign(commands[t + 1])
        for label, a, b in wanted:
            if found[label] >= max_per_type:
                continue
            if s_prev == a and s_cur == b:
                rows = []
                for ti in (t - 1, t, t + 1):
                    raw_diff = None
                    if ti > 0:
                        raw_diff = float(
                            np.mean(
                                np.abs(
                                    frames[ti].astype(np.float64) - frames[ti - 1].astype(np.float64)
                                )
                            )
                        )
                    ctx_diff = None
                    if ti > 0:
                        ctx_diff = float(
                            np.linalg.norm(contexts[ti].numpy() - contexts[ti - 1].numpy())
                        )
                    emb_diff = None
                    if ti > 0:
                        emb_diff = float(np.linalg.norm(z[ti] - z[ti - 1]))
                    rows.append(
                        {
                            "t": ti,
                            "teacher_command": float(commands[ti]),
                            "raw_frame_mean_abs_diff_from_prev": raw_diff,
                            "context_l2_diff_from_prev": ctx_diff,
                            "embedding_l2_diff_from_prev": emb_diff,
                        }
                    )
                examples.append({"transition_type": label, "timesteps": rows})
                found[label] += 1
    return examples


def _state_correlations(z: np.ndarray, states: np.ndarray, commands: np.ndarray) -> dict[str, Any]:
    labels = ["x", "x_dot", "theta", "theta_dot", "teacher_command"]
    vars_ = [
        states[:, 0],
        states[:, 1],
        states[:, 2],
        states[:, 3],
        commands.reshape(-1),
    ]
    out: dict[str, Any] = {}
    for label, v in zip(labels, vars_):
        corrs = []
        for d in range(z.shape[1]):
            x = z[:, d]
            if np.std(x) < 1e-12 or np.std(v) < 1e-12:
                corrs.append(0.0)
            else:
                corrs.append(float(np.corrcoef(x, v)[0, 1]))
        ca = np.abs(np.asarray(corrs))
        out[label] = {
            "max_abs_correlation": float(ca.max()),
            "mean_abs_correlation": float(ca.mean()),
        }
    return out


def _theta_bucket_distances(
    z: np.ndarray,
    states: np.ndarray,
    *,
    targets: tuple[float, float, float] = (-0.25, 0.0, 0.25),
    tol: float = 0.05,
) -> dict[str, Any]:
    thetas = states[:, 2]
    buckets: dict[str, np.ndarray] = {}
    names = ("theta_neg", "theta_zero", "theta_pos")
    for name, target in zip(names, targets):
        mask = np.abs(thetas - target) <= tol
        if mask.any():
            buckets[name] = z[mask].mean(axis=0)
    dists = {}
    keys = list(buckets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            dists[f"{a}_to_{b}"] = float(np.linalg.norm(buckets[a] - buckets[b]))
    return {
        "bucket_targets": list(targets),
        "tolerance": tol,
        "bucket_counts": {k: int(np.sum(np.abs(thetas - t) <= tol)) for k, t in zip(names, targets)},
        "mean_embedding_distances": dists,
    }


def _trace_dataset_samples(
    dataset: TrajectoryDataset,
    jepa: TSJEPA,
    device: torch.device,
    *,
    sample_indices: list[int],
) -> list[dict[str, Any]]:
    traces = []
    for idx in sample_indices:
        file_idx, time_index = dataset.index_map[idx]
        file_path = dataset.files[file_idx]
        with np.load(file_path) as data:
            frames = np.asarray(data["frames"])
            commands = np.asarray(data["commands"])
            states = np.asarray(data["states"])
        item = dataset[idx]
        context = item["context"]
        with torch.no_grad():
            emb = jepa.encode_context(context.unsqueeze(0).to(device)).cpu().numpy()[0]
        frame_indices = [int(time_index)]
        traces.append(
            {
                "dataset_index": idx,
                "trajectory_file": file_path.name,
                "trajectory_id": file_path.stem,
                "time_index": time_index,
                "raw_frame_indices_in_context": frame_indices,
                "raw_frame_hashes": [_hash_array(frames[t]) for t in frame_indices],
                "teacher_command": float(commands[time_index]),
                "physical_state": [float(v) for v in states[time_index]],
                "context_hash_6dp": _hash_array(context.numpy(), decimals=6),
                "embedding_hash_6dp": _hash_array(emb, decimals=6),
            }
        )
    return traces


def _classify(report: dict[str, Any]) -> str:
    raw = report["raw_frame_diversity"]["summary"]
    ctx = report["context_input_diversity"]["summary"]
    emb = report["within_trajectory_temporal_collapse"]["summary"]

    raw_ratio = raw["mean_unique_raw_frames_exact"] / max(raw["mean_num_frames"], 1)
    ctx_ratio = ctx["mean_unique_contexts_rounded_6dp"] / max(ctx["mean_num_contexts"], 1)
    emb_ratio = emb["mean_unique_ratio"]

    if raw_ratio < 0.2:
        return "A. RAW DATA / OBSERVATION COLLAPSE"
    if raw_ratio > 0.5 and ctx_ratio < 0.2:
        return "B. CONTEXT-CONSTRUCTION COLLAPSE"
    if raw_ratio > 0.5 and ctx_ratio > 0.3 and emb_ratio < 0.2:
        return "C. ENCODER REPRESENTATION COLLAPSE"
    if raw_ratio > 0.5 and ctx_ratio > 0.3 and emb_ratio < 0.3:
        state = report["embedding_vs_physical_state"]["summary"]
        if state.get("max_state_correlation", 0) > 0.2:
            return "D. TEMPORAL REPRESENTATION COMPRESSION"
        return "C. ENCODER REPRESENTATION COLLAPSE"
    return "E. NO REPRESENTATION PIPELINE BUG"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_dp_fixed.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-trajectories", type=int, default=10)
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
    payload = torch.load(jepa_ckpt, map_location=device, weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(payload["model"])
    jepa.eval()

    pipeline = PreprocessPipeline(config, training=False)
    normalizer = load_command_normalizer(config, data_root=data_root)

    train_dir = data_root / "trajectories" / "jepa" / "train"
    test_dir = data_root / "trajectories" / "jepa" / "test"
    train_files = sorted(train_dir.glob("*.npz"))[: args.num_trajectories]
    test_files = sorted(test_dir.glob("*.npz"))[: max(3, args.num_trajectories // 3)]

    raw_reports = []
    ctx_reports = []
    emb_tables = []
    collapse_rows = []
    transition_examples = []
    state_corr_all = []
    theta_dist_all = []

    for file_path in list(train_files) + list(test_files):
        with np.load(file_path) as data:
            frames = np.asarray(data["frames"])
            commands = np.asarray(data["commands"], dtype=np.float32)
            states = np.asarray(data["states"], dtype=np.float32)
        tid = file_path.stem

        raw = _raw_frame_diversity(frames, tid)
        raw_reports.append(raw)

        contexts = _build_contexts_for_trajectory(frames, pipeline)
        ctx = _context_diversity(contexts, tid)
        ctx_reports.append(ctx)

        z, table = _encode_trajectory(jepa, contexts, commands, device)
        uniq = _embedding_uniqueness(z)
        collapse_rows.append(
            {
                "trajectory": tid,
                "contexts": len(contexts),
                "unique_z_6dp": uniq["unique_embeddings_6dp"],
                "unique_ratio": float(uniq["unique_embeddings_6dp"] / max(len(contexts), 1)),
                "mean_delta_z": uniq["consecutive_l2_mean"],
                "max_delta_z": uniq["consecutive_l2_max"],
            }
        )
        emb_tables.append(
            {
                "trajectory_id": tid,
                "per_timestep_compact_table": table,
                "embedding_uniqueness": uniq,
            }
        )
        transition_examples.extend(_find_transition_examples(frames, commands, contexts, z))
        state_corr_all.append(_state_correlations(z, states, commands))
        theta_dist_all.append(_theta_bucket_distances(z, states))

    # Dataset trace via TrajectoryDataset (training=False, same as JEPA test eval)
    test_dataset = TrajectoryDataset(test_dir, config, normalizer, training=False)
    trace_indices = [0, 1, 2, 10, 50, 100, 500, 1000, 2000, 3399]
    trace_indices = [i for i in trace_indices if i < len(test_dataset)]
    traces = _trace_dataset_samples(test_dataset, jepa, device, sample_indices=trace_indices)

    # Summaries
    raw_summary = {
        "num_trajectories": len(raw_reports),
        "mean_num_frames": float(np.mean([r["num_frames"] for r in raw_reports])),
        "mean_unique_raw_frames_exact": float(np.mean([r["unique_raw_frames_exact"] for r in raw_reports])),
        "mean_consecutive_frame_difference": float(np.mean([r["mean_consecutive_frame_difference"] for r in raw_reports])),
        "median_consecutive_frame_difference": float(np.median([r["median_consecutive_frame_difference"] for r in raw_reports])),
    }
    ctx_summary = {
        "mean_num_contexts": float(np.mean([r["num_contexts"] for r in ctx_reports])),
        "mean_unique_contexts_rounded_6dp": float(np.mean([r["unique_contexts_rounded_6dp"] for r in ctx_reports])),
        "mean_l2_adjacent_contexts": float(np.mean([r["mean_l2_adjacent_contexts"] for r in ctx_reports])),
        "mean_l2_random_context_pairs": float(np.mean([r["mean_l2_random_context_pairs"] for r in ctx_reports])),
    }
    collapse_summary = {
        "mean_unique_ratio": float(np.mean([r["unique_ratio"] for r in collapse_rows])),
        "min_unique_ratio": float(np.min([r["unique_ratio"] for r in collapse_rows])),
        "max_unique_ratio": float(np.max([r["unique_ratio"] for r in collapse_rows])),
        "mean_consecutive_embedding_l2": float(np.mean([r["mean_delta_z"] for r in collapse_rows])),
    }
    state_summary = {
        "max_state_correlation": float(
            max(
                sc[var]["max_abs_correlation"]
                for sc in state_corr_all
                for var in ("x", "x_dot", "theta", "theta_dot")
            )
        ),
        "mean_state_correlation": float(
            np.mean(
                [
                    sc[var]["mean_abs_correlation"]
                    for sc in state_corr_all
                    for var in ("x", "x_dot", "theta", "theta_dot")
                ]
            )
        ),
        "per_variable_mean_max_abs_correlation": {
            var: float(np.mean([sc[var]["max_abs_correlation"] for sc in state_corr_all]))
            for var in ("x", "x_dot", "theta", "theta_dot", "teacher_command")
        },
    }

    report: dict[str, Any] = {
        "jepa_checkpoint": str(jepa_ckpt),
        "data_root": str(data_root),
        "context_construction": {
            "pipeline_training_flag": False,
            "note": "Deterministic eval path (stochastic=False), matching TrajectoryDataset training=False and ActorEmbeddingDataset.",
            "kappa": int(config["input"]["kappa"]),
            "resize": list(config["input"]["resize"]),
        },
        "raw_frame_diversity": {
            "per_trajectory": raw_reports,
            "summary": raw_summary,
        },
        "context_input_diversity": {
            "per_trajectory": ctx_reports,
            "summary": ctx_summary,
            "distinct_observations_become_distinct_contexts": bool(
                raw_summary["mean_unique_raw_frames_exact"] > 0.8 * raw_summary["mean_num_frames"]
                and ctx_summary["mean_unique_contexts_rounded_6dp"] > 0.5 * ctx_summary["mean_num_contexts"]
            ),
        },
        "encoder_output_per_timestep": emb_tables[:3],
        "frame_context_embedding_chain": transition_examples,
        "embedding_vs_physical_state": {
            "per_trajectory_correlations": state_corr_all[:5],
            "theta_bucket_distances": theta_dist_all[:5],
            "summary": state_summary,
        },
        "within_trajectory_temporal_collapse": {
            "per_trajectory": collapse_rows,
            "summary": collapse_summary,
        },
        "dataset_sample_trace": {
            "dataset": "jepa/test TrajectoryDataset training=False",
            "num_dataset_samples": len(test_dataset),
            "traced_samples": traces,
            "indexing_note": "index_map is (file_idx, time_index); contexts use eval cache; no duplicate frame reference bug detected if hashes vary with time_index",
        },
        "classification": "",
    }
    report["classification"] = _classify(report)

    # #region agent log
    _agent_dbg(
        "H1",
        "diagnose_jepa_representation_pipeline.py:classify",
        "pipeline collapse classification",
        {
            "classification": report["classification"],
            "mean_unique_raw": raw_summary["mean_unique_raw_frames_exact"],
            "mean_num_frames": raw_summary["mean_num_frames"],
            "mean_unique_contexts": ctx_summary["mean_unique_contexts_rounded_6dp"],
            "mean_unique_z_ratio": collapse_summary["mean_unique_ratio"],
            "min_unique_z_ratio": collapse_summary["min_unique_ratio"],
            "mean_consecutive_z_l2": collapse_summary["mean_consecutive_embedding_l2"],
            "mean_consecutive_pixel_diff": raw_summary["mean_consecutive_frame_difference"],
        },
    )
    _agent_dbg(
        "H5",
        "diagnose_jepa_representation_pipeline.py:state_corr",
        "embedding vs physical-state correlations including velocity",
        {
            "max_state_correlation": state_summary["max_state_correlation"],
            "mean_state_correlation": state_summary["mean_state_correlation"],
            "per_variable_mean_max_abs": state_summary["per_variable_mean_max_abs_correlation"],
        },
    )
    # #endregion

    out = (
        Path(args.out)
        if args.out
        else runs_root / "eval" / "jepa_representation_pipeline_diagnostics_dp_fixed_seed0.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("=== JEPA representation pipeline diagnostic ===")
    print(
        f"raw: mean unique frames {raw_summary['mean_unique_raw_frames_exact']:.1f}/"
        f"{raw_summary['mean_num_frames']:.0f} | mean consecutive pixel diff {raw_summary['mean_consecutive_frame_difference']:.4g}"
    )
    print(
        f"context: mean unique {ctx_summary['mean_unique_contexts_rounded_6dp']:.1f}/"
        f"{ctx_summary['mean_num_contexts']:.0f} | mean adjacent L2 {ctx_summary['mean_l2_adjacent_contexts']:.4g}"
    )
    print(
        f"embedding: mean unique ratio {collapse_summary['mean_unique_ratio']:.3f} "
        f"(range {collapse_summary['min_unique_ratio']:.3f}-{collapse_summary['max_unique_ratio']:.3f}) | "
        f"mean consecutive L2 {collapse_summary['mean_consecutive_embedding_l2']:.4g}"
    )
    print(f"max state |correlation|: {state_summary['max_state_correlation']:.4g}")
    print(f"classification: {report['classification']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
