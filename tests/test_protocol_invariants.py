"""Protocol invariant checks: counts, leakage, seeds, selection, reproducibility."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from ts_jepa.config import load_config
from ts_jepa.data.datasets import fit_command_normalizer
from ts_jepa.data.trajectory_generator import build_env_and_teacher, dataset_split_index_plan, generate_dataset_split
from ts_jepa.evaluation.evaluate import baseline_report
from ts_jepa.inference.infer import FrozenRuntimeController
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.training.train_actor import train_semantic_actor_repetitions
from ts_jepa.training.train_jepa import _set_seed, train_ts_jepa, train_ts_jepa_repetitions


def _tiny_config(tmp_path: Path) -> dict:
    config = copy.deepcopy(load_config())
    config["paths"]["data_root"] = str(tmp_path / "data")
    config["paths"]["runs_root"] = str(tmp_path / "runs")
    config["simulation"]["trajectory_steps"] = 30
    config["ts_jepa"]["dataset"]["train_trajectories"] = 6
    config["ts_jepa"]["dataset"]["test_trajectories"] = 3
    config["semantic_actor"]["dataset"]["train_trajectories"] = 4
    config["semantic_actor"]["dataset"]["test_trajectories"] = 2
    config["ts_jepa"]["prediction_horizon"]["Kp"] = 4
    config["ts_jepa"]["optimizer"]["batch_size"] = 2
    config["ts_jepa"]["early_stopping"]["validation_trajectory_count"] = 2
    config["ts_jepa"]["early_stopping"]["patience"] = 5
    config["semantic_actor"]["optimizer"]["batch_size"] = 4
    config["semantic_actor"]["early_stopping"]["patience"] = 5
    config["semantic_actor"]["early_stopping"]["val_fraction"] = 0.25
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
    return config


def _generate(config: dict) -> Path:
    root = Path(config["paths"]["data_root"])
    env, teacher = build_env_and_teacher(config)
    index_plan = dataset_split_index_plan(config)
    mapping = {
        "jepa_train": ("jepa", "train"),
        "jepa_test": ("jepa", "test"),
        "actor_train": ("actor", "train"),
        "actor_test": ("actor", "test"),
    }
    for split_name, (family, split) in mapping.items():
        count, start = index_plan[split_name]
        generate_dataset_split(
            config, split_name, count, start, root / "trajectories" / family / split, env, teacher
        )
    fit_command_normalizer(config, data_root=root)
    return root


def _traj_ids(split_dir: Path) -> set[int]:
    ids = set()
    for path in sorted(split_dir.glob("*.npz")):
        with np.load(path) as data:
            ids.add(int(np.asarray(data["trajectory_index"]).item()))
    return ids


def test_config_dataset_counts_match_paper_baseline():
    config = load_config()
    assert config["ts_jepa"]["dataset"]["train_trajectories"] == 200
    assert config["ts_jepa"]["dataset"]["test_trajectories"] == 40
    assert config["semantic_actor"]["dataset"]["train_trajectories"] == 100
    assert config["semantic_actor"]["dataset"]["test_trajectories"] == 20
    assert config["evaluation"]["repetitions"] == 5
    assert config["evaluation"]["seeds"] == [0, 1, 2, 3, 4]


def test_generated_counts_and_no_train_test_leakage(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = _generate(config)

    jepa_train = root / "trajectories" / "jepa" / "train"
    jepa_test = root / "trajectories" / "jepa" / "test"
    actor_train = root / "trajectories" / "actor" / "train"
    actor_test = root / "trajectories" / "actor" / "test"

    assert len(list(jepa_train.glob("*.npz"))) == 6
    assert len(list(jepa_test.glob("*.npz"))) == 3
    assert len(list(actor_train.glob("*.npz"))) == 4
    assert len(list(actor_test.glob("*.npz"))) == 2

    jepa_train_ids = _traj_ids(jepa_train)
    jepa_test_ids = _traj_ids(jepa_test)
    actor_train_ids = _traj_ids(actor_train)
    actor_test_ids = _traj_ids(actor_test)

    assert jepa_train_ids.isdisjoint(jepa_test_ids)
    assert actor_train_ids.isdisjoint(actor_test_ids)
    assert jepa_train_ids.isdisjoint(actor_train_ids)
    assert jepa_train_ids.isdisjoint(actor_test_ids)
    assert jepa_test_ids.isdisjoint(actor_train_ids)
    assert jepa_test_ids.isdisjoint(actor_test_ids)
    # Generation uses disjoint index ranges across D_s then D_a.
    assert max(jepa_train_ids) < min(jepa_test_ids)
    assert max(jepa_test_ids) < min(actor_train_ids)
    assert max(actor_train_ids) < min(actor_test_ids)


def test_independent_seed_initialization():
    config = load_config()
    _set_seed(0)
    m0 = TSJEPA(config)
    w0 = m0.context_encoder.projection.weight.detach().clone()
    _set_seed(1)
    m1 = TSJEPA(config)
    w1 = m1.context_encoder.projection.weight.detach().clone()
    assert not torch.allclose(w0, w1)


def test_five_seed_loop_independent_init_and_val_only_selection(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"])

    # Capture initial weights implied by each seed (before any shared training state).
    inits = {}
    for seed in config["evaluation"]["seeds"]:
        _set_seed(int(seed))
        model = TSJEPA(config)
        inits[seed] = model.context_encoder.projection.weight.detach().clone()
    assert not torch.allclose(inits[0], inits[1])

    summary = train_ts_jepa_repetitions(config, device=torch.device("cpu"), max_epochs=1, data_root=root)
    assert summary["selection_criterion"] == "best_validation_cosine_alignment_loss"
    assert summary["best_seed"] in config["evaluation"]["seeds"]
    assert len(summary["seed_results"]) == 2

    # Each seed saved its own checkpoint; selected best matches min validation.
    vals = {r["seed"]: r["best_val"] for r in summary["seed_results"]}
    assert summary["best_val"] == min(vals.values())
    assert summary["best_seed"] == min(vals, key=vals.get)

    for seed in config["evaluation"]["seeds"]:
        best = torch.load(runs / "ts_jepa" / f"seed_{seed}" / "best.pt", map_location="cpu", weights_only=False)
        metrics = json.loads((runs / "ts_jepa" / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))
        # Checkpoint chosen during training must not embed test metrics.
        assert "test_loss" not in best
        assert best["selection_split"] == "train_holdout_validation"
        assert "val_loss" in best
        assert "config" in best
        assert best["seed"] == seed
        # Test metrics exist only in post-selection artifacts.
        assert "test_loss" in metrics
        assert metrics["best_val"] == best["val_loss"]

    selected = torch.load(summary["best_checkpoint"], map_location="cpu", weights_only=False)
    assert selected["selection_criterion"] == "best_validation_cosine_alignment_loss"
    assert selected["seed"] == summary["best_seed"]
    # Cross-seed selection uses validation, not test.
    assert all("best_val" in r for r in selected["seed_results"])


def test_same_seed_reproducible_checkpoint(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"])

    r1 = train_ts_jepa(
        config,
        device=torch.device("cpu"),
        max_epochs=1,
        data_root=root,
        seed=7,
        run_dir=runs / "ts_jepa" / "repro_a",
    )
    r2 = train_ts_jepa(
        config,
        device=torch.device("cpu"),
        max_epochs=1,
        data_root=root,
        seed=7,
        run_dir=runs / "ts_jepa" / "repro_b",
    )
    assert abs(r1["best_val"] - r2["best_val"]) < 1e-6
    ckpt1 = torch.load(r1["checkpoint"], map_location="cpu", weights_only=False)
    ckpt2 = torch.load(r2["checkpoint"], map_location="cpu", weights_only=False)
    assert ckpt1["config"]["ts_jepa"]["optimizer"]["learning_rate"] == ckpt2["config"]["ts_jepa"]["optimizer"]["learning_rate"]
    assert ckpt1["seed"] == ckpt2["seed"] == 7
    for key in ckpt1["model"]:
        assert torch.allclose(ckpt1["model"][key], ckpt2["model"][key]), key


def test_end_to_end_smoke_with_invariants(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = _generate(config)
    jepa_summary = train_ts_jepa_repetitions(config, device=torch.device("cpu"), max_epochs=1, data_root=root)
    actor_summary = train_semantic_actor_repetitions(
        config,
        jepa_checkpoint=Path(jepa_summary["best_checkpoint"]),
        device=torch.device("cpu"),
        max_epochs=1,
        data_root=root,
    )
    controller = FrozenRuntimeController.from_checkpoints(
        config,
        Path(jepa_summary["best_checkpoint"]),
        Path(actor_summary["best_checkpoint"]),
        device=torch.device("cpu"),
    )
    report = baseline_report(config, controller, data_root=root)
    assert report["prediction_horizon_nmae"]["split"] == "jepa_test_untouched"
    assert "nmae" in report["prediction_horizon_nmae"]
    assert Path(jepa_summary["best_checkpoint"]).exists()
    assert Path(actor_summary["best_checkpoint"]).exists()
