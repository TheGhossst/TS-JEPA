#!/usr/bin/env python
"""
Full paper-scale baseline pipeline:

1) Generate JEPA 200/40 and actor 100/20 trajectories
2) Train 5 JEPA seeds (150 epochs), select best by validation
3) Train 5 actor seeds (300 epochs), select best by validation
4) Evaluate NMAE (Kp=1..15) + closed-loop + wireless; write runs/eval artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from ts_jepa.config import load_config, project_root
from ts_jepa.data.trajectory_generator import generate_all_trajectories
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.evaluate import baseline_report, write_evaluation_artifacts
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.training.train_actor import train_semantic_actor_repetitions
from ts_jepa.training.train_jepa import train_ts_jepa_repetitions
from ts_jepa.evaluation.checkpoints import resolve_run_checkpoint


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-jepa", action="store_true")
    parser.add_argument("--skip-actor", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--jepa-epochs", type=int, default=None)
    parser.add_argument("--actor-epochs", type=int, default=None)
    args = parser.parse_args()

    # Prefer all CPU cores for the long paper-scale run.
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    config = load_config(args.config)
    root = project_root(config)
    data_root = root / config["paths"]["data_root"]
    device = select_device(args.device)
    print(f"device={device} cpu_threads={torch.get_num_threads()}")
    print(json.dumps(describe_device(device), indent=2))

    if not args.skip_generate:
        t0 = time.time()
        print("Generating full datasets...")
        generate_all_trajectories(config, data_root=data_root)
        verify_dataset_counts(config, data_root)
        print(f"Generation done in {time.time() - t0:.1f}s")
    else:
        verify_dataset_counts(config, data_root)

    if not args.skip_jepa:
        t0 = time.time()
        print("Training 5 JEPA seeds...")
        jepa_summary = train_ts_jepa_repetitions(
            config,
            device=device,
            max_epochs=args.jepa_epochs,
            data_root=data_root,
        )
        print(json.dumps(jepa_summary, indent=2))
        print(f"JEPA training done in {time.time() - t0:.1f}s")
    else:
        jepa_summary = json.loads((root / "runs" / "ts_jepa" / "repetition_summary.json").read_text(encoding="utf-8"))

    if not args.skip_actor:
        t0 = time.time()
        print("Training 5 actor seeds...")
        actor_summary = train_semantic_actor_repetitions(
            config,
            jepa_checkpoint=root / "runs" / "ts_jepa" / "best.pt",
            device=device,
            max_epochs=args.actor_epochs,
            data_root=data_root,
        )
        print(json.dumps(actor_summary, indent=2))
        print(f"Actor training done in {time.time() - t0:.1f}s")
    else:
        actor_summary = json.loads(
            (root / "runs" / "semantic_actor" / "repetition_summary.json").read_text(encoding="utf-8")
        )

    if not args.skip_eval:
        t0 = time.time()
        print("Evaluating untouched test sets + baseline report...")
        jepa_ckpt = resolve_run_checkpoint(root / "runs", "ts_jepa")
        actor_ckpt = resolve_run_checkpoint(root / "runs", "semantic_actor")
        controller = FrozenRuntimeController.from_checkpoints(
            config,
            jepa_ckpt,
            actor_ckpt,
            device=device,
        )
        report = baseline_report(config, controller, data_root=data_root)
        out_dir = root / "runs" / "eval"
        paths = write_evaluation_artifacts(report, out_dir)
        print(json.dumps({"artifacts": paths, "report": report}, indent=2, default=str))
        print(f"Wrote eval artifacts under {out_dir} in {time.time() - t0:.1f}s")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
