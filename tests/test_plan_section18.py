"""Plan §18: paper-silent items must not be asserted as paper-exact."""

from __future__ import annotations

import copy

import pytest

from ts_jepa.config import load_config
from ts_jepa.evaluation.metrics import communication_reduction_report
from ts_jepa.models.predictor_command_resolution import assert_plan_predictor_command_resolution
from ts_jepa.plan.actor import PLAN_SEMANTIC_ACTOR, assert_plan_semantic_actor_config
from ts_jepa.plan.encoder import PLAN_ENCODER, assert_plan_encoder_config
from ts_jepa.plan.environment import PLAN_ENVIRONMENT, assert_plan_environment_config
from ts_jepa.plan.predictor import PLAN_PREDICTOR, assert_plan_predictor_config
from ts_jepa.plan.training import PLAN_JEPA_TRAINING, assert_plan_jepa_training_config
from ts_jepa.plan.wireless import PLAN_WIRELESS, assert_plan_wireless_config
from ts_jepa.wireless.scheduler import ChannelAwareScheduler


def test_plan_dicts_omit_section18_implementation_choices():
    assert "render_height" not in PLAN_ENVIRONMENT
    assert "render_width" not in PLAN_ENVIRONMENT
    assert "cart_mass" not in PLAN_ENVIRONMENT
    assert "blocks_per_stage" not in PLAN_ENCODER
    assert "stem" not in PLAN_ENCODER
    assert "spatial_pool_hw" not in PLAN_ENCODER
    assert "input_dim" not in PLAN_PREDICTOR
    assert PLAN_PREDICTOR.get("input_tensor_construction") is None
    assert "activation" not in PLAN_PREDICTOR
    assert "output_activation" not in PLAN_SEMANTIC_ACTOR
    assert "patience" not in PLAN_JEPA_TRAINING
    assert "validation_trajectory_count" not in PLAN_JEPA_TRAINING
    assert "momentum" not in PLAN_JEPA_TRAINING
    assert PLAN_WIRELESS["lyapunov_B_omitted"] is True
    assert "lyapunov_B" not in PLAN_WIRELESS
    for key in ("max_devices_scheduled_J", "p_max_watt", "drift_plus_penalty_V", "aoi_threshold_beta_th", "num_devices"):
        assert key not in PLAN_WIRELESS
        assert key in PLAN_WIRELESS["implementation_choice_keys"]


def test_native_render_physics_and_stacking_are_not_plan_requirements():
    config = copy.deepcopy(load_config())
    config["simulation"]["render_height"] = 32
    config["simulation"]["render_width"] = 64
    config["simulation"]["physics"]["cart_mass"] = 9.0
    config["simulation"]["physics"]["pole_mass"] = 2.0
    config["simulation"]["process_noise_std"] = 0.3
    config["control_teacher"]["control_effort_weight"] = 1.0
    config["input"]["multi_frame_tensor_construction"] = "not_channel_concat"
    assert_plan_environment_config(config)


def test_jepa_patience_and_val_count_are_not_plan_requirements():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["early_stopping"]["patience"] = 3
    config["ts_jepa"]["early_stopping"]["validation_trajectory_count"] = 1
    config["semantic_actor"]["early_stopping"]["patience"] = 7
    config["semantic_actor"]["early_stopping"]["val_fraction"] = 0.5
    config["ts_jepa"]["optimizer"]["momentum"] = 0.9
    assert_plan_jepa_training_config(config)


def test_scheduler_scalars_are_not_plan_requirements():
    config = copy.deepcopy(load_config())
    config["wireless"]["num_devices"] = 8
    config["wireless"]["max_devices_scheduled_J"] = 3
    config["wireless"]["drift_plus_penalty_V"] = 99.0
    config["wireless"]["aoi_threshold_beta_th"] = 1.0
    config["wireless"]["p_max_watt"] = 5.0
    assert_plan_wireless_config(config)


def test_lyapunov_B_is_omitted_not_recovered():
    config = load_config()
    sched = ChannelAwareScheduler(config, policy="channel_aware")
    cost = sched.drift_plus_penalty_cost(aoi=3.0, queue=2.0, p_req=0.01)
    expected_without_b = 1.0 - (3.0 + 1.0) ** 2 - 2.0 * 2.0 * 3.0 + sched.V * 0.01
    assert cost == pytest.approx(expected_without_b)
    assert not hasattr(sched, "lyapunov_B")


def test_predictor_concat_and_command_source_are_not_paper_exact():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["predictor"]["input_tensor_construction"] = "gated_fusion"
    config["ts_jepa"]["predictor"]["activation"] = "GELU"
    assert_plan_predictor_config(config)
    resolution = config["ts_jepa"]["predictor_command_resolution"]
    assert resolution["paper_exact"] is False
    assert resolution["status"] == "OPEN"
    assert_plan_predictor_command_resolution(config)


def test_encoder_ic_fields_are_not_plan_requirements():
    config = copy.deepcopy(load_config())
    config["ts_jepa"]["encoder"]["blocks_per_stage"] = 4
    config["ts_jepa"]["encoder"]["spatial_pool_hw"] = [2, 2]
    assert_plan_encoder_config(config)


def test_actor_output_activation_is_not_in_paper_spec_dict():
    assert_plan_semantic_actor_config(load_config())
    assert "output_activation" not in PLAN_SEMANTIC_ACTOR
    config = copy.deepcopy(load_config())
    config["semantic_actor"]["architecture"]["output_activation"] = "sigmoid"
    assert_plan_semantic_actor_config(config)


def test_embedding_bitwidth_is_recovered_not_paper_exact():
    report = communication_reduction_report(height=64, width=128, embedding_dim=256)
    assert report["bitwidth_paper_exact"] is False
    assert "recovered" in report["embedding_bitwidth_source"]
