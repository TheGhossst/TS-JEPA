"""Resume-from-checkpoint and early-stopping baseline tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from ts_jepa.config import load_config
from ts_jepa.data.datasets import fit_command_normalizer
from ts_jepa.data.trajectory_generator import build_env_and_teacher, generate_dataset_split
from ts_jepa.runtime import load_checkpoint, save_checkpoint
from ts_jepa.training.jepa_optimizer import jepa_scheduled_lr
from ts_jepa.training.train_jepa import (
    train_ts_jepa,
    train_ts_jepa_repetitions,
    validate_checkpoint_config_compatibility,
)


def _tiny_config(tmp_path: Path) -> dict:
    config = copy.deepcopy(load_config())
    config["paths"]["data_root"] = str(tmp_path / "data")
    config["paths"]["runs_root"] = str(tmp_path / "runs")
    config["simulation"]["trajectory_steps"] = 30
    config["ts_jepa"]["dataset"]["train_trajectories"] = 6
    config["ts_jepa"]["dataset"]["test_trajectories"] = 3
    config["ts_jepa"]["prediction_horizon"]["Kp"] = 4
    config["ts_jepa"]["optimizer"]["batch_size"] = 2
    config["ts_jepa"]["early_stopping"]["enabled"] = False
    config["ts_jepa"]["early_stopping"]["validation_trajectory_count"] = 2
    config["ts_jepa"]["early_stopping"]["patience"] = 1
    config["control_teacher"]["value_iteration_iters"] = 2
    config["control_teacher"]["force_bins"] = 5
    config["control_teacher"]["grid"] = {
        "x": [-0.4, 0.4, 5],
        "x_dot": [-1.0, 1.0, 5],
        "theta": [-0.2, 0.2, 5],
        "theta_dot": [-1.0, 1.0, 5],
    }
    return config


def _generate(config: dict) -> Path:
    root = Path(config["paths"]["data_root"])
    env, teacher = build_env_and_teacher(config)
    generate_dataset_split(
        config, "jepa_train", 6, 0, root / "trajectories" / "jepa" / "train", env, teacher
    )
    generate_dataset_split(
        config, "jepa_test", 3, 6, root / "trajectories" / "jepa" / "test", env, teacher
    )
    fit_command_normalizer(config, data_root=root)
    return root


def test_baseline_early_stopping_enabled():
    config = load_config()
    assert config["ts_jepa"]["early_stopping"]["enabled"] is True


def test_dp_fixed_early_stopping_enabled():
    config = load_config("configs/ts_jepa_dp_fixed.yaml")
    assert config["ts_jepa"]["early_stopping"]["enabled"] is True


def test_resumable_checkpoint_contains_required_fields(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_0"
    train_ts_jepa(config, device=torch.device("cpu"), max_epochs=1, data_root=root, seed=0, run_dir=runs)

    ckpt = load_checkpoint(runs / "last.pt")
    for key in (
        "model",
        "optimizer",
        "epoch",
        "best_val",
        "best_epoch",
        "stale",
        "history",
        "config",
        "normalizer",
        "seed",
        "predictor_command_resolution",
    ):
        assert key in ckpt, key
    assert ckpt["epoch"] == 1
    assert isinstance(ckpt["optimizer"]["param_groups"], list)


def test_resume_starts_from_checkpoint_epoch_plus_one(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_0"

    train_ts_jepa(config, device=torch.device("cpu"), max_epochs=1, data_root=root, seed=0, run_dir=runs)
    first_ckpt = load_checkpoint(runs / "last.pt")
    assert first_ckpt["epoch"] == 1

    train_ts_jepa(
        config,
        device=torch.device("cpu"),
        max_epochs=2,
        data_root=root,
        seed=0,
        run_dir=runs,
        resume_from=runs / "last.pt",
    )
    final_ckpt = load_checkpoint(runs / "last.pt")
    assert len(final_ckpt["history"]) == 2
    assert final_ckpt["epoch"] == 2
    assert [entry["epoch"] for entry in final_ckpt["history"]] == [1, 2]


def test_resume_restores_optimizer_state(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_0"

    train_ts_jepa(config, device=torch.device("cpu"), max_epochs=1, data_root=root, seed=0, run_dir=runs)
    ckpt = load_checkpoint(runs / "last.pt")
    saved_optimizer = copy.deepcopy(ckpt["optimizer"])

    train_ts_jepa(
        config,
        device=torch.device("cpu"),
        max_epochs=2,
        data_root=root,
        seed=0,
        run_dir=runs,
        resume_from=runs / "last.pt",
    )
    resumed_ckpt = load_checkpoint(runs / "last.pt")
    assert len(resumed_ckpt["optimizer"]["state"]) == len(saved_optimizer["state"])
    assert resumed_ckpt["optimizer"]["param_groups"][0]["lr"] == pytest.approx(
        jepa_scheduled_lr(2, config)
    )


def test_resume_applies_scheduled_lr_not_stale_checkpoint_lr(tmp_path: Path):
    """IC warmup: resume uses jepa_scheduled_lr(epoch), not a hacked optimizer LR."""
    config = _tiny_config(tmp_path)
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_0"

    train_ts_jepa(config, device=torch.device("cpu"), max_epochs=1, data_root=root, seed=0, run_dir=runs)
    ckpt = load_checkpoint(runs / "last.pt")
    ckpt["optimizer"]["param_groups"][0]["lr"] = 0.198
    save_checkpoint(runs / "last.pt", ckpt)

    train_ts_jepa(
        config,
        device=torch.device("cpu"),
        max_epochs=2,
        data_root=root,
        seed=0,
        run_dir=runs,
        resume_from=runs / "last.pt",
    )
    resumed = load_checkpoint(runs / "last.pt")
    scheduled = jepa_scheduled_lr(2, config)
    assert resumed["optimizer"]["param_groups"][0]["lr"] == pytest.approx(scheduled)
    assert resumed["optimizer"]["param_groups"][0]["lr"] != pytest.approx(0.198)
    assert resumed["optimizer"]["param_groups"][0]["lr"] != pytest.approx(
        float(config["ts_jepa"]["optimizer"]["learning_rate"])
    )


def test_resume_preserves_best_val_and_best_epoch(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_0"

    train_ts_jepa(
        config, device=torch.device("cpu"), max_epochs=1, data_root=root, seed=0, run_dir=runs
    )
    ckpt = load_checkpoint(runs / "last.pt")
    ckpt["best_val"] = -1e9
    ckpt["best_epoch"] = 99
    save_checkpoint(runs / "last.pt", ckpt)

    train_ts_jepa(
        config,
        device=torch.device("cpu"),
        max_epochs=2,
        data_root=root,
        seed=0,
        run_dir=runs,
        resume_from=runs / "last.pt",
    )
    final_ckpt = load_checkpoint(runs / "last.pt")
    assert final_ckpt["best_val"] == -1e9
    assert final_ckpt["best_epoch"] == 99


def test_resume_extends_history_instead_of_replacing(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_0"

    train_ts_jepa(config, device=torch.device("cpu"), max_epochs=1, data_root=root, seed=0, run_dir=runs)
    first_history = load_checkpoint(runs / "last.pt")["history"]

    train_ts_jepa(
        config,
        device=torch.device("cpu"),
        max_epochs=2,
        data_root=root,
        seed=0,
        run_dir=runs,
        resume_from=runs / "last.pt",
    )
    final_history = load_checkpoint(runs / "last.pt")["history"]
    assert final_history[: len(first_history)] == first_history
    assert len(final_history) == len(first_history) + 1


def test_resume_incompatible_config_fails_clearly(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_0"
    train_ts_jepa(config, device=torch.device("cpu"), max_epochs=1, data_root=root, seed=0, run_dir=runs)

    bad_config = copy.deepcopy(config)
    bad_config["ts_jepa"]["prediction_horizon"]["Kp"] = 99
    with pytest.raises(ValueError, match="incompatible"):
        train_ts_jepa(
            bad_config,
            device=torch.device("cpu"),
            max_epochs=2,
            data_root=root,
            seed=0,
            run_dir=runs,
            resume_from=runs / "last.pt",
        )


def test_validate_checkpoint_config_compatibility_detects_mismatch():
    config = load_config()
    bad = copy.deepcopy(config)
    bad["ts_jepa"]["encoder"]["embedding_dim"] = 64
    with pytest.raises(ValueError, match="embedding_dim"):
        validate_checkpoint_config_compatibility(config, bad)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda c: c["ts_jepa"]["optimizer"].update({"momentum": 0.9}), "optimizer momentum"),
        (lambda c: c["ts_jepa"]["lr_decay"].update({"factor": 0.98}), "lr_decay.factor"),
        (lambda c: c["ts_jepa"]["lr_decay"].update({"interval_epochs": 10}), "lr_decay.interval_epochs"),
        (lambda c: c["ts_jepa"]["optimizer"].update({"batch_size": 128}), "batch_size"),
        (lambda c: c["ts_jepa"]["optimizer"].update({"microbatch_size": 8}), "microbatch_size"),
        (
            lambda c: c["ts_jepa"]["predictor_command_resolution"].update({"selected_source": "semantic_actor"}),
            "predictor_command_resolution.selected_source",
        ),
        (
            lambda c: c["ts_jepa"]["predictor_command_resolution"].update({"paper_exact": True}),
            "predictor_command_resolution.paper_exact",
        ),
    ],
    ids=[
        "momentum",
        "lr_decay_factor",
        "lr_decay_interval",
        "batch_size",
        "microbatch_size",
        "command_selected_source",
        "command_paper_exact",
    ],
)
def test_validate_checkpoint_config_compatibility_detects_training_param_mismatch(mutator, match):
    config = load_config()
    bad = copy.deepcopy(config)
    mutator(bad)
    with pytest.raises(ValueError, match=match):
        validate_checkpoint_config_compatibility(config, bad)


def test_training_without_resume_behaves_normally(tmp_path: Path):
    config = _tiny_config(tmp_path)
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_0"

    result = train_ts_jepa(
        config, device=torch.device("cpu"), max_epochs=1, data_root=root, seed=0, run_dir=runs
    )
    ckpt = load_checkpoint(runs / "last.pt")
    assert ckpt["epoch"] == 1
    assert result["best_epoch"] >= 1
    assert (runs / "best.pt").exists()
    best = load_checkpoint(runs / "best.pt")
    assert "val_loss" in best
    assert "optimizer" not in best


def test_disabled_early_stopping_runs_full_max_epochs(tmp_path: Path):
    config = _tiny_config(tmp_path)
    config["ts_jepa"]["early_stopping"]["enabled"] = False
    config["ts_jepa"]["early_stopping"]["patience"] = 1
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_0"

    train_ts_jepa(config, device=torch.device("cpu"), max_epochs=3, data_root=root, seed=0, run_dir=runs)
    ckpt = load_checkpoint(runs / "last.pt")
    assert len(ckpt["history"]) == 3
    assert ckpt["epoch"] == 3


def test_early_stopping_still_available_when_enabled(tmp_path: Path):
    config = _tiny_config(tmp_path)
    config["ts_jepa"]["early_stopping"]["enabled"] = True
    config["ts_jepa"]["early_stopping"]["patience"] = 1
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_0"

    train_ts_jepa(config, device=torch.device("cpu"), max_epochs=10, data_root=root, seed=0, run_dir=runs)
    ckpt = load_checkpoint(runs / "last.pt")
    assert len(ckpt["history"]) < 10


def test_seed_training_complete_requires_epoch_budget(tmp_path: Path):
    from ts_jepa.training.train_jepa import seed_training_complete

    run_dir = tmp_path / "seed_0"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text('{"seed": 0, "test_loss": -0.98}', encoding="utf-8")
    torch.save({"test_loss": -0.98, "epoch": 2, "history": [{"epoch": 1}, {"epoch": 2}]}, run_dir / "last.pt")
    assert seed_training_complete(run_dir, expected_epochs=2) is True
    assert seed_training_complete(run_dir, expected_epochs=150) is False


def test_repetitions_resume_continues_incomplete_seed(tmp_path: Path):
    config = _tiny_config(tmp_path)
    config["evaluation"]["repetitions"] = 2
    config["evaluation"]["seeds"] = [0, 1]
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"]) / "ts_jepa"

    train_ts_jepa(
        config, device=torch.device("cpu"), max_epochs=1, data_root=root, seed=0, run_dir=runs / "seed_0"
    )
    first = load_checkpoint(runs / "seed_0" / "last.pt")
    assert first["epoch"] == 1

    train_ts_jepa_repetitions(
        config,
        device=torch.device("cpu"),
        max_epochs=2,
        data_root=root,
        resume=True,
    )
    resumed = load_checkpoint(runs / "seed_0" / "last.pt")
    assert resumed["epoch"] == 2
    assert len(resumed["history"]) == 2
    assert (runs / "seed_1" / "last.pt").is_file()


def test_repetitions_without_resume_restarts_incomplete_seed(tmp_path: Path):
    config = _tiny_config(tmp_path)
    config["evaluation"]["repetitions"] = 1
    config["evaluation"]["seeds"] = [0]
    root = _generate(config)
    runs = Path(config["paths"]["runs_root"]) / "ts_jepa" / "seed_0"

    train_ts_jepa(config, device=torch.device("cpu"), max_epochs=1, data_root=root, seed=0, run_dir=runs)
    first_history_len = len(load_checkpoint(runs / "last.pt")["history"])

    train_ts_jepa_repetitions(
        config,
        device=torch.device("cpu"),
        max_epochs=1,
        data_root=root,
        resume=False,
    )
    restarted = load_checkpoint(runs / "last.pt")
    assert restarted["epoch"] == 1
    assert len(restarted["history"]) == first_history_len
