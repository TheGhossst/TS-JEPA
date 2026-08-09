from __future__ import annotations

from pathlib import Path

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, load_config, project_root


def test_baseline_paths_unchanged():
    config = load_config()
    assert config["paths"]["data_root"] == "data"
    assert jepa_run_dirname(config) == "ts_jepa"
    assert actor_run_dirname(config) == "semantic_actor"
    assert config["ts_jepa"]["prediction_horizon"]["Kp"] == 15
    assert config["ts_jepa"]["optimizer"]["batch_size"] == 256
    assert config["ts_jepa"]["optimizer"]["microbatch_size"] == 16
    assert config["simulation"]["init_noise"] == 0.35


def test_dp_fixed_overlay_paths_and_hparams():
    config = load_config("configs/ts_jepa_dp_fixed.yaml")
    assert config["paths"]["data_root"] == "data_dp_fixed"
    assert jepa_run_dirname(config) == "ts_jepa_dp_fixed"
    assert actor_run_dirname(config) == "semantic_actor_dp_fixed"
    # Paper / baseline hparams preserved via _base merge.
    assert config["ts_jepa"]["prediction_horizon"]["Kp"] == 15
    assert config["ts_jepa"]["optimizer"]["batch_size"] == 256
    assert config["ts_jepa"]["optimizer"]["microbatch_size"] == 16
    assert config["ts_jepa"]["optimizer"]["learning_rate"] == 0.2
    assert config["ts_jepa"]["optimizer"]["weight_decay"] == 0.0004
    assert config["ts_jepa"]["target_encoder"]["ema_decay"] == 0.99
    assert config["simulation"]["init_noise"] == 0.35
    # Separated run dirs under project.
    root = project_root(config)
    assert (root / config["paths"]["runs_root"] / jepa_run_dirname(config)) != (
        root / config["paths"]["runs_root"] / "ts_jepa"
    ) or jepa_run_dirname(config) != "ts_jepa"
    assert jepa_run_dirname(config) != "ts_jepa"
    assert actor_run_dirname(config) != "semantic_actor"
