#!/usr/bin/env python
"""Generate plan §4 cart-pole RGB trajectories with DP teacher + sanity verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_jepa.config import load_config, project_root
from ts_jepa.data.dataset_sanity import sanity_check_trajectory_root
from ts_jepa.data.trajectory_generator import generate_all_trajectories
from ts_jepa.data.temporal_plan import assert_plan_temporal_config
from ts_jepa.env.env_validation import assert_plan_environment_validated
from ts_jepa.losses.loss_plan import assert_plan_jepa_loss_config
from ts_jepa.models.predictor_command_resolution import assert_plan_predictor_command_resolution
from ts_jepa.models.predictor_plan import assert_plan_predictor_config
from ts_jepa.preprocessing.plan import assert_plan_preprocessing_config
from ts_jepa.training.jepa_procedure import assert_plan_jepa_procedure_config
from ts_jepa.training.jepa_training_plan import assert_plan_jepa_training_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=(
            "Output dataset root (contains trajectories/). "
            "Default: config paths.data_root under the project. "
            "Use a separate directory (e.g. data_dp_fixed) to avoid overwriting existing data."
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    assert_plan_environment_validated(config)
    assert_plan_preprocessing_config(config)
    assert_plan_temporal_config(config)
    assert_plan_predictor_config(config)
    assert_plan_predictor_command_resolution(config)
    assert_plan_jepa_loss_config(config)
    assert_plan_jepa_training_config(config)
    assert_plan_jepa_procedure_config(config)
    data_root = Path(args.data_root) if args.data_root else (project_root(config) / config["paths"]["data_root"])
    root = generate_all_trajectories(config, data_root=data_root, run_sanity_check=False)
    report = sanity_check_trajectory_root(config, root, fit_normalizer=True)
    out = project_root(config) / "runs" / "eval" / "trajectory_sanity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Wrote trajectories under {root}")
    print(f"Wrote sanity report {out}")
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
