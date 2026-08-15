#!/usr/bin/env python
"""
Working-overlay pipeline (not paper-faithful).

1) Generate 20 ms / κ-stack trajectories into data_working/
2) Train TS-JEPA (cosine + VICReg, AdamW)
3) JEPA gates: predictor uses u, encoder rank
4) Train semantic actor
5) Actor + closed-loop gates vs hold-last / zero-action
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, load_config, project_root
from ts_jepa.data.dataset_sanity import sanity_check_trajectory_root
from ts_jepa.data.trajectory_generator import generate_all_trajectories
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.working_gates import run_working_gates
from ts_jepa.plan.enforce import is_working_mode
from ts_jepa.training.train_actor import train_semantic_actor, train_semantic_actor_repetitions
from ts_jepa.training.train_jepa import train_ts_jepa, train_ts_jepa_repetitions


def _count_npz(path: Path) -> int:
    return len(list(path.glob("*.npz"))) if path.exists() else 0


def verify_dataset_counts(config: dict, data_root: Path) -> None:
    expected = {
        data_root / "trajectories" / "jepa" / "train": config["ts_jepa"]["dataset"]["train_trajectories"],
        data_root / "trajectories" / "jepa" / "test": config["ts_jepa"]["dataset"]["test_trajectories"],
        data_root / "trajectories" / "actor" / "train": config["semantic_actor"]["dataset"]["train_trajectories"],
        data_root / "trajectories" / "actor" / "test": config["semantic_actor"]["dataset"]["test_trajectories"],
    }
    for path, count in expected.items():
        got = _count_npz(path)
        if got != count:
            raise RuntimeError(f"Dataset count mismatch at {path}: got {got}, expected {count}")
        print(f"OK {path}: {got}")
    report = sanity_check_trajectory_root(config, data_root, fit_normalizer=False)
    if not report["overall_pass"]:
        raise RuntimeError(
            "Trajectory sanity failed (native resolution, D_s/D_a overlap, or teacher checks). "
            f"pass={report['pass']} id_overlap={report.get('id_overlap')}"
        )
    print("OK trajectory sanity")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/ts_jepa_working.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-jepa", action="store_true")
    parser.add_argument("--skip-actor", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--jepa-epochs", type=int, default=None)
    parser.add_argument("--actor-epochs", type=int, default=None)
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Train the full 5-seed protocol. Default is seed 0 only.",
    )
    args = parser.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    config = load_config(args.config)
    if not is_working_mode(config):
        raise SystemExit(
            "run_working.py requires plan.mode=working (use configs/ts_jepa_working.yaml). "
            "Paper-faithful runs use scripts/pipeline/run_full_baseline.py."
        )

    root = project_root(config)
    data_root = root / config["paths"]["data_root"]
    runs_root = root / config["paths"]["runs_root"]
    jepa_family = jepa_run_dirname(config)
    actor_family = actor_run_dirname(config)
    device = select_device(args.device)
    print(f"device={device} cpu_threads={torch.get_num_threads()} mode=working")
    print(json.dumps(describe_device(device), indent=2))

    if not args.skip_generate:
        t0 = time.time()
        print("Generating working datasets (20 ms stride)...")
        generate_all_trajectories(config, data_root=data_root)
        verify_dataset_counts(config, data_root)
        print(f"Generation done in {time.time() - t0:.1f}s")
    else:
        verify_dataset_counts(config, data_root)

    if not args.skip_jepa:
        t0 = time.time()
        if args.all_seeds:
            print("Training 5 JEPA seeds (working overlay)...")
            jepa_summary = train_ts_jepa_repetitions(
                config,
                device=device,
                max_epochs=args.jepa_epochs,
                data_root=data_root,
            )
        else:
            print("Training JEPA seed 0 (working overlay)...")
            jepa_summary = train_ts_jepa(
                config,
                device=device,
                max_epochs=args.jepa_epochs,
                data_root=data_root,
                seed=0,
            )
        print(json.dumps(jepa_summary, indent=2, default=str))
        print(f"JEPA training done in {time.time() - t0:.1f}s")

    if not args.skip_eval:
        t0 = time.time()
        jepa_gates = run_working_gates(
            config,
            device=device,
            data_root=data_root,
            include_actor=False,
            include_closed_loop=False,
        )
        print(json.dumps({"jepa_gates": jepa_gates["jepa"]}, indent=2))
        if not jepa_gates["pass"]:
            raise SystemExit(
                "JEPA working gates failed (predictor still ignores u or rank too low). "
                "Try observation_stride_steps=50 before a full 150-epoch run."
            )
        print(f"JEPA gates passed in {time.time() - t0:.1f}s")

    if not args.skip_actor:
        t0 = time.time()
        jepa_ckpt = runs_root / jepa_family / "best.pt"
        if not jepa_ckpt.is_file():
            jepa_ckpt = runs_root / jepa_family / "seed_0" / "best.pt"
        if args.all_seeds:
            print("Training 5 actor seeds (working overlay)...")
            actor_summary = train_semantic_actor_repetitions(
                config,
                jepa_checkpoint=jepa_ckpt,
                device=device,
                max_epochs=args.actor_epochs,
                data_root=data_root,
            )
        else:
            print("Training actor seed 0 (working overlay)...")
            actor_summary = train_semantic_actor(
                config,
                jepa_checkpoint=jepa_ckpt,
                device=device,
                max_epochs=args.actor_epochs,
                data_root=data_root,
                seed=0,
            )
        print(json.dumps(actor_summary, indent=2, default=str))
        print(f"Actor training done in {time.time() - t0:.1f}s")

    if not args.skip_eval and not args.skip_actor:
        t0 = time.time()
        report = run_working_gates(
            config,
            device=device,
            data_root=data_root,
            include_actor=True,
            include_closed_loop=True,
        )
        out = runs_root / "eval" / "working_gates.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2, default=str))
        print(f"Wrote {out} in {time.time() - t0:.1f}s")
        if not report["pass"]:
            raise SystemExit("Working overlay gates failed.")

    print("Working pipeline complete.")


if __name__ == "__main__":
    main()
