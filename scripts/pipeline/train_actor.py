#!/usr/bin/env python
"""Train semantic actor with the paper 5-repetition protocol (best validation run)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import (
    actor_run_dirname,
    apply_cli_path_overrides,
    jepa_run_dirname,
    load_config,
    project_root,
)
from ts_jepa.device import describe_device, select_device
from ts_jepa.plan.actor import (
    assert_plan_semantic_actor_config,
    assert_plan_semantic_actor_training_config,
)
from ts_jepa.training.train_actor import train_semantic_actor, train_semantic_actor_repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--jepa-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--single-seed",
        type=int,
        default=None,
        help="If set, train only this seed instead of the full repetition protocol.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override config paths.data_root (e.g. data_dp_fixed).",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=None,
        help="Override config paths.runs_root.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    assert_plan_semantic_actor_config(config)
    # Full plan §15 table only when not using smoke epoch overrides.
    if args.epochs is None:
        assert_plan_semantic_actor_training_config(config)
    apply_cli_path_overrides(config, data_root=args.data_root, runs_root=args.runs_root)
    device = select_device(args.device)
    print(describe_device(device))
    print(
        {
            "data_root": str(project_root(config) / config["paths"]["data_root"]),
            "actor_runs": str(
                project_root(config) / config["paths"]["runs_root"] / actor_run_dirname(config)
            ),
            "default_jepa_family": jepa_run_dirname(config),
        }
    )
    ckpt = Path(args.jepa_checkpoint) if args.jepa_checkpoint else None
    if args.single_seed is not None:
        result = train_semantic_actor(
            config,
            jepa_checkpoint=ckpt,
            device=device,
            max_epochs=args.epochs,
            seed=args.single_seed,
        )
    else:
        result = train_semantic_actor_repetitions(
            config,
            jepa_checkpoint=ckpt,
            device=device,
            max_epochs=args.epochs,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
