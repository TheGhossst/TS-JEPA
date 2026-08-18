#!/usr/bin/env python
"""Tier 1: nonlinear frozen-z → state MLP vs linear probe, then Probe-LQR.

Does not update JEPA. Compares θ̇ R² on actor test, then runs LQR on the MLP
decoder trained on near-upright LQR rollouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ts_jepa.config import apply_cli_path_overrides, load_config, project_root
from ts_jepa.data.datasets import ActorEmbeddingDataset, load_command_normalizer
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.command_linear_probe import collect_actor_arrays
from ts_jepa.evaluation.mlp_state_decoder import mlp_decoder_metrics, train_mlp_state_decoder
from ts_jepa.evaluation.state_probe import STATE_DIM_NAMES, resolve_jepa_ckpt
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.training.probe_lqr_actor import train_probe_lqr_actor


def _encode_actor_splits(config, jepa_ckpt, device, data_root):
    payload = __import__("torch").load(jepa_ckpt, map_location="cpu", weights_only=False)
    jepa = TSJEPA(config).to(device)
    jepa.load_state_dict(payload["model"])
    jepa.eval()
    for p in jepa.parameters():
        p.requires_grad_(False)
    normalizer = load_command_normalizer(config, data_root=data_root)
    train_dir = data_root / "trajectories" / "actor" / "train"
    test_dir = data_root / "trajectories" / "actor" / "test"
    print(f"encoding frozen z from {train_dir} and {test_dir} ...", flush=True)
    train_ds = ActorEmbeddingDataset(
        train_dir, config, normalizer, jepa.context_encoder, device, training=False
    )
    test_ds = ActorEmbeddingDataset(
        test_dir, config, normalizer, jepa.context_encoder, device, training=False
    )
    train = collect_actor_arrays(train_ds, train_dir)
    test = collect_actor_arrays(test_ds, test_dir)
    del jepa
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_working.yaml")
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--actor-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-lqr", action="store_true")
    parser.add_argument("--cache", type=str, default="runs/eval/actor_z_state_splits.npz")
    args = parser.parse_args()
    config = load_config(args.config)
    apply_cli_path_overrides(config)
    root = project_root(config)
    data_root = root / config["paths"]["data_root"]
    device = select_device(args.device)
    print(describe_device(device), flush=True)
    jepa_ckpt = resolve_jepa_ckpt(
        config,
        explicit=Path(args.jepa_checkpoint) if args.jepa_checkpoint else None,
        seed=0,
    )
    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = root / cache
    if cache.is_file():
        print(f"loading cached z,state from {cache}", flush=True)
        with np.load(cache) as payload:
            train = {"z": payload["z_train"], "state": payload["s_train"]}
            test = {"z": payload["z_test"], "state": payload["s_test"]}
    else:
        train, test = _encode_actor_splits(config, jepa_ckpt, device, data_root)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            z_train=train["z"],
            s_train=train["state"],
            z_test=test["z"],
            s_test=test["state"],
        )
        print(f"wrote {cache}", flush=True)

    print("fitting linear ridge probe on actor train (same protocol as frozen_jepa_state_probe)", flush=True)
    linear_report = None
    linear_json = root / "runs" / "eval" / "frozen_jepa_state_probe.json"
    if linear_json.is_file():
        linear_report = json.loads(linear_json.read_text(encoding="utf-8"))

    print("training MLP decoder on actor train", flush=True)
    probe = train_mlp_state_decoder(train["z"], train["state"], device=device)
    train_means = np.mean(np.asarray(train["state"][:, :4], dtype=np.float64), axis=0)
    mlp_test = mlp_decoder_metrics(test["z"], test["state"], probe, train_means=train_means)
    linear_r2 = (linear_report or {}).get("multivariate", {}).get("test_r2_per_dim")
    comparison = {
        "jepa_frozen": True,
        "jepa_checkpoint": str(jepa_ckpt),
        "linear_test_r2_per_dim": linear_r2,
        "mlp_test_r2_per_dim": mlp_test["test_r2_per_dim"],
        "mlp_test_mean_r2": mlp_test["test_mean_r2"],
        "mlp_test_mae_per_dim": mlp_test["test_mae_per_dim"],
        "state_dim_names": list(STATE_DIM_NAMES),
        "theta_dot_linear_r2": None if linear_r2 is None else float(linear_r2[3]),
        "theta_dot_mlp_r2": float(mlp_test["test_r2_per_dim"][3]),
        "theta_dot_r2_delta": (
            None if linear_r2 is None else float(mlp_test["test_r2_per_dim"][3] - linear_r2[3])
        ),
        "mlp_test_per_dim": mlp_test["per_dim"],
    }
    print(json.dumps({k: comparison[k] for k in comparison if k != "mlp_test_per_dim"}, indent=2), flush=True)

    out = root / "runs" / "eval" / "frozen_jepa_state_probe_mlp.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    if not args.skip_lqr:
        print("Probe-LQR with MLP decoder on balanced LQR rollouts", flush=True)
        lqr_report = train_probe_lqr_actor(
            config,
            jepa_checkpoint=jepa_ckpt,
            actor_checkpoint=Path(args.actor_checkpoint) if args.actor_checkpoint else None,
            device=device,
            decoder="mlp",
        )
        comparison["probe_lqr_mlp"] = {
            "full_information": lqr_report["probe_lqr_full_information"],
            "working_gate": lqr_report["working_gate"],
            "probe_mae_state": lqr_report["probe_mae_state"],
        }
        out.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
        print(f"Updated {out} with Probe-LQR", flush=True)


if __name__ == "__main__":
    main()
