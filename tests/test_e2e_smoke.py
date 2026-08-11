"""End-to-end smoke: tiny data + repetition protocol + NMAE + baseline validation."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from ts_jepa.config import load_config
from ts_jepa.data.datasets import fit_command_normalizer
from ts_jepa.data.trajectory_generator import build_env_and_teacher, generate_dataset_split
from ts_jepa.evaluation.evaluate import baseline_report
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.training.train_actor import train_semantic_actor_repetitions
from ts_jepa.training.train_jepa import train_ts_jepa_repetitions


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
    config["semantic_actor"]["early_stopping"]["patience"] = 2
    config["semantic_actor"]["early_stopping"]["val_fraction"] = 0.34
    config["control_teacher"]["value_iteration_iters"] = 2
    config["control_teacher"]["force_bins"] = 5
    config["control_teacher"]["grid"] = {
        "x": [-0.4, 0.4, 5],
        "x_dot": [-1.0, 1.0, 5],
        "theta": [-0.2, 0.2, 5],
        "theta_dot": [-1.0, 1.0, 5],
    }
    config["evaluation"]["repetitions"] = 2
    config["evaluation"]["seeds"] = [0, 1]

    root = Path(config["paths"]["data_root"])
    env, teacher = build_env_and_teacher(config)
    generate_dataset_split(config, "jepa_train", 4, 0, root / "trajectories" / "jepa" / "train", env, teacher)
    generate_dataset_split(config, "jepa_test", 2, 4, root / "trajectories" / "jepa" / "test", env, teacher)
    generate_dataset_split(config, "actor_train", 3, 0, root / "trajectories" / "actor" / "train", env, teacher)
    generate_dataset_split(config, "actor_test", 1, 3, root / "trajectories" / "actor" / "test", env, teacher)

    fit_command_normalizer(config, data_root=root)
    jepa_summary = train_ts_jepa_repetitions(
        config, device=torch.device("cpu"), max_epochs=1, data_root=root
    )
    assert Path(jepa_summary["best_checkpoint"]).exists()
    assert len(jepa_summary["seed_results"]) == 2
    assert (Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_0" / "best.pt").exists()
    assert (Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_1" / "metrics.json").exists()

    actor_summary = train_semantic_actor_repetitions(
        config,
        jepa_checkpoint=Path(jepa_summary["best_checkpoint"]),
        device=torch.device("cpu"),
        max_epochs=1,
        data_root=root,
    )
    assert Path(actor_summary["best_checkpoint"]).exists()
    assert len(actor_summary["seed_results"]) == 2

    controller = FrozenRuntimeController.from_checkpoints(
        config,
        Path(jepa_summary["best_checkpoint"]),
        Path(actor_summary["best_checkpoint"]),
        device=torch.device("cpu"),
    )
    report = baseline_report(config, controller, data_root=root)
    assert "control" in report
    assert "prediction_horizon_nmae" in report
    assert report["prediction_horizon_nmae"]["kp"] == 5
    assert "nmae_by_horizon" in report["prediction_horizon_nmae"]
    assert set(report["prediction_horizon_nmae"]["nmae_by_horizon"]) == {str(h) for h in range(1, 6)}
    assert report["prediction_horizon_nmae"]["split"] == "jepa_test_untouched"
    assert "test_losses" in report
    assert report["test_losses"]["jepa"]["best_test_loss"] is not None
    assert report["test_losses"]["semantic_actor"]["best_test_loss"] is not None
    assert report["test_losses"]["jepa"]["source"] == "repetition_summary"
    assert "baseline_validation" in report
    assert "embedding_tsne" in report
    resolution = report["predictor_command_resolution"]
    assert resolution["status"] == "OPEN"
    assert resolution["selected_source"] == "teacher_dp"
    assert resolution["paper_exact"] is False
    assert "wireless" not in report
