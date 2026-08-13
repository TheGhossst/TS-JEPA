"""Semantic Actor training-path audit tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ts_jepa.config import load_config
from ts_jepa.data.datasets import ActorEmbeddingDataset, fit_command_normalizer, load_command_normalizer
from ts_jepa.data.trajectory_generator import build_env_and_teacher, generate_dataset_split
from ts_jepa.evaluation.actor_training_audit import (
    _constant_mean_analysis,
    _split_disjointness,
    _verify_command_indexing,
    audit_actor_training,
)
from ts_jepa.models.actor import SemanticActor
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.training.train_actor import train_semantic_actor
from ts_jepa.training.train_jepa import train_ts_jepa_repetitions


def _tiny_config(tmp_path: Path) -> dict:
    config = copy.deepcopy(load_config())
    config["paths"]["data_root"] = str(tmp_path / "data")
    config["paths"]["runs_root"] = str(tmp_path / "runs")
    config["simulation"]["trajectory_steps"] = 20
    config["ts_jepa"]["dataset"]["train_trajectories"] = 3
    config["ts_jepa"]["dataset"]["test_trajectories"] = 1
    config["semantic_actor"]["dataset"]["train_trajectories"] = 2
    config["semantic_actor"]["dataset"]["test_trajectories"] = 1
    config["ts_jepa"]["prediction_horizon"]["Kp"] = 3
    config["ts_jepa"]["optimizer"]["batch_size"] = 2
    config["ts_jepa"]["early_stopping"]["validation_trajectory_count"] = 1
    config["ts_jepa"]["early_stopping"]["patience"] = 1
    config["semantic_actor"]["optimizer"]["batch_size"] = 2
    config["semantic_actor"]["optimizer"]["epochs"] = 1
    config["semantic_actor"]["early_stopping"]["patience"] = 1
    config["semantic_actor"]["early_stopping"]["val_fraction"] = 0.34
    config["control_teacher"]["value_iteration_iters"] = 2
    config["control_teacher"]["force_bins"] = 5
    config["control_teacher"]["grid"] = {
        "x": [-0.4, 0.4, 5],
        "x_dot": [-1.0, 1.0, 5],
        "theta": [-0.2, 0.2, 5],
        "theta_dot": [-1.0, 1.0, 5],
    }
    config["evaluation"]["repetitions"] = 1
    config["evaluation"]["seeds"] = [0]
    return config


def test_constant_mean_detection_flags_collapsed_predictions():
    targets_norm = np.array([1.0, -1.0, 0.5, -0.5, 0.2])
    preds_norm = np.full(5, 0.08)
    targets_phys = targets_norm * 10.0
    preds_phys = np.full(5, 0.8)
    out = _constant_mean_analysis(targets_norm, preds_norm, targets_phys, preds_phys)
    assert out["detected_near_constant_mean_solution"] is True
    assert out["pred_has_both_signs"] is False
    assert out["target_has_both_signs"] is True


def test_constant_mean_detection_allows_varied_predictions():
    targets_norm = np.array([1.0, -1.0, 0.5, -0.5])
    preds_norm = np.array([0.9, -0.8, 0.4, -0.3])
    targets_phys = targets_norm * 10.0
    preds_phys = preds_norm * 10.0
    out = _constant_mean_analysis(targets_norm, preds_norm, targets_phys, preds_phys)
    assert out["detected_near_constant_mean_solution"] is False
    assert out["pred_has_both_signs"] is True


def test_split_disjointness_and_command_indexing(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = Path(config["paths"]["data_root"])
    env, teacher = build_env_and_teacher(config)
    generate_dataset_split(
        config, "actor_train", 2, 0, root / "trajectories" / "actor" / "train", env, teacher
    )
    generate_dataset_split(
        config, "actor_test", 1, 2, root / "trajectories" / "actor" / "test", env, teacher
    )
    generate_dataset_split(config, "jepa_train", 2, 0, root / "trajectories" / "jepa" / "train", env, teacher)
    fit_command_normalizer(config, data_root=root)
    normalizer = load_command_normalizer(config, data_root=root)
    jepa = TSJEPA(config)
    train_dir = root / "trajectories" / "actor" / "train"
    test_dir = root / "trajectories" / "actor" / "test"
    split = _split_disjointness(train_dir, test_dir)
    assert split["splits_disjoint"] is True
    assert split["overlap_ids"] == []

    train_ds = ActorEmbeddingDataset(train_dir, config, normalizer, jepa.context_encoder, torch.device("cpu"))
    idx = _verify_command_indexing(train_ds, train_dir, max_check=40)
    assert idx["command_indexing_ok"] is True
    assert idx["command_index_mismatches"] == []


def test_checkpoint_integrity_and_jepa_consistency(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = Path(config["paths"]["data_root"])
    env, teacher = build_env_and_teacher(config)
    generate_dataset_split(config, "jepa_train", 3, 0, root / "trajectories" / "jepa" / "train", env, teacher)
    generate_dataset_split(config, "jepa_test", 1, 3, root / "trajectories" / "jepa" / "test", env, teacher)
    generate_dataset_split(config, "actor_train", 2, 0, root / "trajectories" / "actor" / "train", env, teacher)
    generate_dataset_split(config, "actor_test", 1, 2, root / "trajectories" / "actor" / "test", env, teacher)
    fit_command_normalizer(config, data_root=root)

    jepa_summary = train_ts_jepa_repetitions(
        config, device=torch.device("cpu"), max_epochs=1, data_root=root
    )
    actor_summary = train_semantic_actor(
        config,
        jepa_checkpoint=Path(jepa_summary["best_checkpoint"]),
        device=torch.device("cpu"),
        max_epochs=1,
        data_root=root,
        seed=0,
    )

    report = audit_actor_training(
        config,
        jepa_checkpoint=Path(jepa_summary["best_checkpoint"]),
        actor_checkpoint=Path(actor_summary["checkpoint"]),
        data_root=root,
        device=torch.device("cpu"),
        max_embedding_samples=50,
    )

    assert report["1_jepa_checkpoint_consistency"]["uses_same_frozen_encoder_as_actor_training"] is True
    assert report["10_checkpoint_integrity"]["ok"] is True
    assert report["10_checkpoint_integrity"]["differs_from_random_init"] is True
    assert report["9_split_and_command_indexing"]["train_command_indexing"]["command_indexing_ok"] is True
    assert report["8_preprocessing_runtime_parity"]["embedding_encode_context"]["parity_ok"] is True
    assert "6_training_metrics" in report
    assert np.isfinite(report["6_training_metrics"]["mse_normalized"]["test"])
    assert np.isfinite(report["6_training_metrics"]["nmae_physical"]["test"])
    assert "findings" in report
    for key in (
        "2_embedding_distributions",
        "3_target_command_distributions",
        "4_actor_prediction_distributions",
        "5_correlation",
        "7_constant_mean_solution",
    ):
        assert key in report


def test_audit_detects_jepa_checkpoint_path_mismatch(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = Path(config["paths"]["data_root"])
    env, teacher = build_env_and_teacher(config)
    generate_dataset_split(config, "jepa_train", 3, 0, root / "trajectories" / "jepa" / "train", env, teacher)
    generate_dataset_split(config, "jepa_test", 1, 3, root / "trajectories" / "jepa" / "test", env, teacher)
    generate_dataset_split(config, "actor_train", 2, 0, root / "trajectories" / "actor" / "train", env, teacher)
    generate_dataset_split(config, "actor_test", 1, 2, root / "trajectories" / "actor" / "test", env, teacher)
    fit_command_normalizer(config, data_root=root)

    jepa_summary = train_ts_jepa_repetitions(
        config, device=torch.device("cpu"), max_epochs=1, data_root=root
    )
    actor_summary = train_semantic_actor(
        config,
        jepa_checkpoint=Path(jepa_summary["best_checkpoint"]),
        device=torch.device("cpu"),
        max_epochs=1,
        data_root=root,
        seed=0,
    )

    # Point audit at a different JEPA checkpoint path than actor recorded.
    other_jepa = tmp_path / "other_jepa.pt"
    other_jepa.write_bytes(Path(jepa_summary["best_checkpoint"]).read_bytes())

    report = audit_actor_training(
        config,
        jepa_checkpoint=other_jepa,
        actor_checkpoint=Path(actor_summary["checkpoint"]),
        data_root=root,
        device=torch.device("cpu"),
        max_embedding_samples=20,
    )
    assert report["1_jepa_checkpoint_consistency"]["paths_match"] is False
    assert any("JEPA checkpoint" in f for f in report["findings"])


def test_audit_report_has_all_ten_sections(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = Path(config["paths"]["data_root"])
    env, teacher = build_env_and_teacher(config)
    generate_dataset_split(config, "jepa_train", 3, 0, root / "trajectories" / "jepa" / "train", env, teacher)
    generate_dataset_split(config, "actor_train", 2, 0, root / "trajectories" / "actor" / "train", env, teacher)
    generate_dataset_split(config, "actor_test", 1, 2, root / "trajectories" / "actor" / "test", env, teacher)
    fit_command_normalizer(config, data_root=root)
    jepa_summary = train_ts_jepa_repetitions(
        config, device=torch.device("cpu"), max_epochs=1, data_root=root
    )
    actor_summary = train_semantic_actor(
        config,
        jepa_checkpoint=Path(jepa_summary["best_checkpoint"]),
        device=torch.device("cpu"),
        max_epochs=1,
        data_root=root,
        seed=0,
    )
    report = audit_actor_training(
        config,
        jepa_checkpoint=Path(jepa_summary["best_checkpoint"]),
        actor_checkpoint=Path(actor_summary["checkpoint"]),
        data_root=root,
        device=torch.device("cpu"),
        max_embedding_samples=30,
    )
    expected = [f"{i}_" for i in range(1, 11)]
    keys = list(report.keys())
    for prefix in (
        "1_jepa_checkpoint_consistency",
        "2_embedding_distributions",
        "3_target_command_distributions",
        "4_actor_prediction_distributions",
        "5_correlation",
        "6_training_metrics",
        "7_constant_mean_solution",
        "8_preprocessing_runtime_parity",
        "9_split_and_command_indexing",
        "10_checkpoint_integrity",
    ):
        assert prefix in keys

    out = tmp_path / "audit.json"
    out.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["audit_version"] == 1
