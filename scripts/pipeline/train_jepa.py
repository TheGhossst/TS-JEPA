#!/usr/bin/env python
"""Train TS-JEPA with the paper 5-repetition protocol (best validation run)."""

from __future__ import annotations

import argparse
import json

from ts_jepa.config import apply_cli_path_overrides, jepa_run_dirname, load_config, project_root
from ts_jepa.data.temporal_plan import assert_plan_temporal_config
from ts_jepa.env.env_validation import assert_plan_environment_validated
from ts_jepa.losses.loss_plan import assert_plan_jepa_loss_config
from ts_jepa.models.predictor_command_resolution import assert_plan_predictor_command_resolution
from ts_jepa.models.predictor_plan import assert_plan_predictor_config
from ts_jepa.device import describe_device, select_device
from ts_jepa.training.jepa_procedure import assert_plan_jepa_procedure_config
from ts_jepa.training.jepa_training_plan import assert_plan_jepa_training_config
from ts_jepa.training.train_jepa import train_ts_jepa, train_ts_jepa_repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
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
    assert_plan_environment_validated(config)
    assert_plan_temporal_config(config)
    assert_plan_predictor_config(config)
    assert_plan_predictor_command_resolution(config)
    assert_plan_jepa_loss_config(config)
    assert_plan_jepa_training_config(config)
    assert_plan_jepa_procedure_config(config)
    apply_cli_path_overrides(config, data_root=args.data_root, runs_root=args.runs_root)
    device = select_device(args.device)
    print(describe_device(device))
    print(
        {
            "data_root": str(project_root(config) / config["paths"]["data_root"]),
            "jepa_runs": str(
                project_root(config) / config["paths"]["runs_root"] / jepa_run_dirname(config)
            ),
        }
    )
    if args.single_seed is not None:
        result = train_ts_jepa(config, device=device, max_epochs=args.epochs, seed=args.single_seed)
    else:
        result = train_ts_jepa_repetitions(config, device=device, max_epochs=args.epochs)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
