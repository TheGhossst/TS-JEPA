#!/usr/bin/env python
"""
Full paper-scale baseline pipeline:

1) Generate JEPA 200/40 and actor 100/20 trajectories
2) Train 5 JEPA seeds (150 epochs), select best by validation
3) Train 5 actor seeds (300 epochs), select best by validation
4) Evaluate baseline (plan §16): NMAE, closed-loop, t-SNE, validation gate
5) Optional: wireless eval only after baseline validation passes (--include-wireless)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, load_config, project_root
from ts_jepa.data.trajectory_generator import generate_all_trajectories
from ts_jepa.data.temporal_plan import assert_plan_temporal_config
from ts_jepa.env.env_validation import assert_plan_environment_validated
from ts_jepa.losses.loss_plan import assert_plan_jepa_loss_config
from ts_jepa.models.predictor_command_resolution import assert_plan_predictor_command_resolution
from ts_jepa.models.predictor_plan import assert_plan_predictor_config
from ts_jepa.device import describe_device, select_device
from ts_jepa.evaluation.evaluate import baseline_report, write_evaluation_artifacts
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.training.jepa_procedure import assert_plan_jepa_procedure_config
from ts_jepa.training.jepa_training_plan import assert_plan_jepa_training_config
from ts_jepa.plan.actor import (
    assert_plan_semantic_actor_config,
    assert_plan_semantic_actor_training_config,
)
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
    parser.add_argument("--include-wireless", action="store_true", help="Run wireless eval only after baseline validation passes.")
    parser.add_argument("--jepa-epochs", type=int, default=None)
    parser.add_argument("--actor-epochs", type=int, default=None)
    args = parser.parse_args()

    # Prefer all CPU cores for the long paper-scale run.
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    config = load_config(args.config)
    assert_plan_environment_validated(config)
    assert_plan_temporal_config(config)
    assert_plan_predictor_config(config)
    assert_plan_predictor_command_resolution(config)
    assert_plan_jepa_loss_config(config)
    assert_plan_jepa_training_config(config)
    assert_plan_jepa_procedure_config(config)
    assert_plan_semantic_actor_config(config)
    if args.actor_epochs is None:
        assert_plan_semantic_actor_training_config(config)
    root = project_root(config)
    data_root = root / config["paths"]["data_root"]
    runs_root = root / config["paths"]["runs_root"]
    jepa_family = jepa_run_dirname(config)
    actor_family = actor_run_dirname(config)
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
        jepa_summary = json.loads((runs_root / jepa_family / "repetition_summary.json").read_text(encoding="utf-8"))

    if not args.skip_actor:
        t0 = time.time()
        print("Training 5 actor seeds...")
        actor_summary = train_semantic_actor_repetitions(
            config,
            jepa_checkpoint=runs_root / jepa_family / "best.pt",
            device=device,
            max_epochs=args.actor_epochs,
            data_root=data_root,
        )
        print(json.dumps(actor_summary, indent=2))
        print(f"Actor training done in {time.time() - t0:.1f}s")
    else:
        actor_summary = json.loads(
            (runs_root / actor_family / "repetition_summary.json").read_text(encoding="utf-8")
        )

    if not args.skip_eval:
        t0 = time.time()
        print("Evaluating untouched test sets + baseline report (plan §16)...")
        jepa_ckpt = resolve_run_checkpoint(runs_root, jepa_family)
        actor_ckpt = resolve_run_checkpoint(runs_root, actor_family)
        controller = FrozenRuntimeController.from_checkpoints(
            config,
            jepa_ckpt,
            actor_ckpt,
            device=device,
        )
        report = baseline_report(config, controller, data_root=data_root, include_wireless=args.include_wireless)
        out_dir = runs_root / "eval"
        paths = write_evaluation_artifacts(report, out_dir)
        validation = report.get("baseline_validation") or {}
        print(json.dumps({"artifacts": paths, "baseline_validation": validation}, indent=2, default=str))
        if args.include_wireless and not validation.get("wireless_allowed"):
            print("WARNING: wireless evaluation skipped — baseline validation did not pass.")
        print(f"Wrote eval artifacts under {out_dir} in {time.time() - t0:.1f}s")

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
