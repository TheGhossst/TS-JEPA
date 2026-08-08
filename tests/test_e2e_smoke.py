"""End-to-end smoke: tiny data + 1 epoch JEPA/actor + runtime/wireless."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from ts_jepa.config import load_config, project_root
from ts_jepa.data.datasets import fit_command_normalizer
from ts_jepa.data.trajectory_generator import generate_dataset_split, build_env_and_teacher
from ts_jepa.evaluation.evaluate import baseline_report
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.training.train_actor import train_semantic_actor
from ts_jepa.training.train_jepa import train_ts_jepa


def test_end_to_end_smoke(tmp_path: Path):
    config = load_config()
    config = copy.deepcopy(config)
    config["paths"]["data_root"] = str(tmp_path / "data")
    config["paths"]["runs_root"] = str(tmp_path / "runs")
    config["simulation"]["trajectory_steps"] = 40
    config["ts_jepa"]["dataset"]["train_trajectories"] = 4
    config["ts_jepa"]["dataset"]["test_trajectories"] = 2
    config["semantic_actor"]["dataset"]["train_trajectories"] = 3
    config["semantic_actor"]["dataset"]["test_trajectories"] = 1
    config["ts_jepa"]["prediction_horizon"]["Kp"] = 5
    config["ts_jepa"]["optimizer"]["batch_size"] = 2
    config["ts_jepa"]["early_stopping"]["validation_trajectory_count"] = 1
    config["ts_jepa"]["early_stopping"]["patience"] = 2
    config["semantic_actor"]["optimizer"]["batch_size"] = 4
    config["control_teacher"]["value_iteration_iters"] = 2
    config["control_teacher"]["force_bins"] = 5
    config["control_teacher"]["grid"] = {
        "x": [-0.4, 0.4, 5],
        "x_dot": [-1.0, 1.0, 5],
        "theta": [-0.2, 0.2, 5],
        "theta_dot": [-1.0, 1.0, 5],
    }
    config["evaluation"]["repetitions"] = 1

    root = Path(config["paths"]["data_root"])
    env, teacher = build_env_and_teacher(config)
    generate_dataset_split(config, "jepa_train", 4, 0, root / "trajectories" / "jepa" / "train", env, teacher)
    generate_dataset_split(config, "jepa_test", 2, 4, root / "trajectories" / "jepa" / "test", env, teacher)
    generate_dataset_split(config, "actor_train", 3, 0, root / "trajectories" / "actor" / "train", env, teacher)
    generate_dataset_split(config, "actor_test", 1, 3, root / "trajectories" / "actor" / "test", env, teacher)

    fit_command_normalizer(config, data_root=root)
    jepa_result = train_ts_jepa(config, device=torch.device("cpu"), max_epochs=1, data_root=root)
    assert Path(jepa_result["runs_dir"], "best.pt").exists()

    actor_result = train_semantic_actor(
        config,
        jepa_checkpoint=Path(jepa_result["runs_dir"]) / "best.pt",
        device=torch.device("cpu"),
        max_epochs=1,
        data_root=root,
    )
    assert Path(actor_result["runs_dir"], "best.pt").exists()

    controller = FrozenRuntimeController.from_checkpoints(
        config,
        Path(jepa_result["runs_dir"]) / "best.pt",
        Path(actor_result["runs_dir"]) / "best.pt",
        device=torch.device("cpu"),
    )
    report = baseline_report(config, controller)
    assert "control" in report
    assert "wireless" in report
    assert "channel_aware" in report["wireless"]
