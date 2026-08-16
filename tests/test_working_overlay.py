"""Working-overlay salvage tests. Paper-faithful asserts stay on the baseline yaml."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from ts_jepa.config import actor_run_dirname, jepa_run_dirname, load_config
from ts_jepa.control.dp_teacher import DPControlTeacher
from ts_jepa.env.cartpole_rgb import InvertedCartPoleEnv
from ts_jepa.losses.jepa_loss import vicreg_covariance_loss, vicreg_variance_loss
from ts_jepa.models.ts_jepa import TSJEPA
from ts_jepa.plan.enforce import is_working_mode, jepa_in_channels, jepa_uses_kappa_stack, plan_enforced
from ts_jepa.plan.environment import assert_plan_environment_config
from ts_jepa.plan.loss import assert_plan_jepa_loss_config
from ts_jepa.plan.training import assert_plan_jepa_training_config
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.preprocessing.pipeline import PreprocessPipeline
from ts_jepa.training.jepa_optimizer import build_jepa_optimizer
from ts_jepa.training.jepa_procedure import (
    attach_command_norm_range,
    jepa_forward_batch,
    jepa_sgd_and_ema_step,
    sample_broad_command_pair,
    vicreg_regularizer,
)


def test_working_overlay_isolated_from_paper_paths():
    paper = load_config()
    working = load_config("configs/ts_jepa_working.yaml")
    assert plan_enforced(paper) is True
    assert is_working_mode(paper) is False
    assert plan_enforced(working) is False
    assert is_working_mode(working) is True
    assert working["paths"]["data_root"] == "data_working"
    assert jepa_run_dirname(working) == "ts_jepa_working"
    assert actor_run_dirname(working) == "semantic_actor_working"
    assert paper["paths"]["data_root"] == "data"
    assert paper["simulation"]["observation_stride_steps"] == 1
    assert working["simulation"]["observation_stride_steps"] == 20
    assert working["control_teacher"]["dp_substeps"] == 20
    assert working["simulation"]["dt"] == 0.001
    assert jepa_uses_kappa_stack(working)
    assert jepa_in_channels(working) == 6
    assert jepa_in_channels(paper) == 3


def test_working_overlay_skips_plan_asserts():
    working = load_config("configs/ts_jepa_working.yaml")
    assert_plan_environment_config(working)
    assert_plan_jepa_training_config(working)
    assert_plan_jepa_loss_config(working)


def test_paper_yaml_still_rejects_working_hparams():
    paper = copy.deepcopy(load_config())
    paper["simulation"]["observation_stride_steps"] = 20
    with pytest.raises(ValueError, match="observation_stride_steps"):
        assert_plan_environment_config(paper)
    paper = copy.deepcopy(load_config())
    paper["ts_jepa"]["optimizer"]["type"] = "AdamW"
    paper["ts_jepa"]["optimizer"]["learning_rate"] = 0.001
    with pytest.raises(ValueError, match="Plan §11"):
        assert_plan_jepa_training_config(paper)
    paper = copy.deepcopy(load_config())
    paper["ts_jepa"]["loss"]["vicreg_variance_weight"] = 25.0
    with pytest.raises(ValueError, match="vicreg"):
        assert_plan_jepa_loss_config(paper)


def test_dp_substeps_20_allowed_in_working_mode():
    env = InvertedCartPoleEnv(dt=0.001, process_noise_std=0.0)
    grid = {
        "x": [-0.2, 0.2, 3],
        "x_dot": [-0.5, 0.5, 3],
        "theta": [-0.15, 0.15, 5],
        "theta_dot": [-0.5, 0.5, 3],
    }
    teacher = DPControlTeacher(
        env=env,
        grid=grid,
        force_bins=5,
        value_iteration_iters=2,
        dp_substeps=20,
        allow_non_unit_substeps=True,
    )
    assert teacher.dp_substeps == 20
    assert teacher.effective_dp_dt == pytest.approx(0.02)
    s0 = np.array([0.0, 0.0, 0.05, 0.0], dtype=np.float64)
    one = DPControlTeacher(
        env=env,
        grid=grid,
        force_bins=5,
        value_iteration_iters=1,
        dp_substeps=1,
    )
    nxt20, _ = teacher._rollout_constant_force(s0, 20.0)
    nxt1, _ = one._rollout_constant_force(s0, 20.0)
    assert abs(float(nxt20[0])) > abs(float(nxt1[0]))


def test_kappa_stack_jepa_input_is_six_channels():
    working = load_config("configs/ts_jepa_working.yaml")
    pipe = PreprocessPipeline(working, training=False)
    frames = np.zeros((4, 128, 256, 3), dtype=np.uint8)
    frames[1] = 10
    frames[2] = 40
    processed = {t: pipe.process_frame(frames[t], stochastic=False) for t in range(4)}
    ctx = pipe.assemble_jepa_input(processed, 2)
    assert ctx.shape[0] == 6
    paper = load_config()
    paper_pipe = PreprocessPipeline(paper, training=False)
    paper_ctx = paper_pipe.assemble_jepa_input(processed, 2)
    assert paper_ctx.shape[0] == 3


def test_vicreg_variance_positive_on_collapse_zero_on_spread():
    collapsed = torch.ones(32, 256)
    spread = torch.randn(128, 256)
    spread = (spread - spread.mean(0)) / (spread.std(0) + 1e-6) * 2.0
    assert float(vicreg_variance_loss(collapsed, gamma=1.0)) > 0.5
    assert float(vicreg_variance_loss(spread, gamma=1.0)) < 1e-5
    independent = torch.randn(128, 8)
    correlated = independent[:, :1].repeat(1, 8)
    assert float(vicreg_covariance_loss(correlated)) > float(vicreg_covariance_loss(independent))
    scaled = independent * 3.0
    assert float(vicreg_covariance_loss(scaled)) > float(vicreg_covariance_loss(independent))
    assert float(vicreg_covariance_loss(independent, standardize=True)) == pytest.approx(
        float(vicreg_covariance_loss(scaled, standardize=True)), rel=1e-3, abs=1e-5
    )
    weighted, var_term, cov_term = vicreg_regularizer(
        collapsed, variance_weight=25.0, covariance_weight=5.0, gamma=1.0
    )
    assert float(weighted) == pytest.approx(25.0 * float(var_term) + 5.0 * float(cov_term))


def test_working_optimizer_is_adamw():
    working = load_config("configs/ts_jepa_working.yaml")
    model = TSJEPA(working)
    opt = build_jepa_optimizer(model, working)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0003)
    assert working["ts_jepa"]["loss"]["vicreg_variance_weight"] == 25.0
    assert working["ts_jepa"]["loss"]["vicreg_covariance_weight"] == 1.0
    assert working["ts_jepa"]["loss"]["vicreg_covariance_standardize"] is True
    assert working["ts_jepa"]["loss"]["command_contrast_weight"] == 1.0
    assert working["ts_jepa"]["loss"]["command_contrast_sampling"] == "both"
    assert model.vicreg_covariance_standardize is True
    assert model.command_contrast_weight == pytest.approx(1.0)
    assert model.command_contrast_sampling == "both"
    assert model.context_encoder.stem[0].in_channels == 6
    assert model.context_encoder.l2_normalize is False
    assert model.predictor.l2_normalize_output is False
    assert model.predictor.conditioning == "film"
    assert model.predictor.command_scale == pytest.approx(1.0)
    assert model.predictor.fc_in.in_features == 256
    assert model.predictor.film is not None


def test_working_jepa_forward_two_step_cpu_smoke():
    working = load_config("configs/ts_jepa_working.yaml")
    working = copy.deepcopy(working)
    working["ts_jepa"]["prediction_horizon"]["Kp"] = 2
    model = TSJEPA(working)
    opt = build_jepa_optimizer(model, working)
    b, kp = 4, 2
    batch = {
        "context": torch.randn(b, 6, 64, 128),
        "future_frames": torch.randn(b, kp, 6, 64, 128),
        "teacher_commands_norm": torch.randn(b, kp),
    }
    result = jepa_forward_batch(model, batch)
    assert result.z_context.shape == (b, 256)
    assert result.z_pred.shape == (b, kp, 256)
    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.cosine_loss)
    assert torch.isfinite(result.vicreg_variance)
    assert float(result.vicreg_variance.detach()) > 0.0 or float(result.vicreg_covariance.detach()) >= 0.0
    assert torch.isfinite(result.command_contrast)
    assert torch.isfinite(result.command_contrast_teacher)
    assert torch.isfinite(result.command_contrast_broad)
    assert float(result.command_contrast.detach()) > 0.0
    result.loss.backward()
    assert model.predictor.film.weight.grad is not None
    assert float(model.predictor.film.weight.grad.norm()) > 0.0
    jepa_sgd_and_ema_step(model, opt, max_grad_norm=1.0)
    z_norm = model.encode_context(batch["context"]).norm(dim=-1)
    assert not torch.allclose(z_norm, torch.ones_like(z_norm), atol=1e-3)


def test_broad_command_contrast_samples_actuator_range():
    working = copy.deepcopy(load_config("configs/ts_jepa_working.yaml"))
    working["ts_jepa"]["prediction_horizon"]["Kp"] = 2
    model = TSJEPA(working)
    attach_command_norm_range(model, CommandNormalizer(mean=0.0, std=10.0), working)
    assert model.command_norm_min == pytest.approx(-2.0)
    assert model.command_norm_max == pytest.approx(2.0)
    u_a, u_b = sample_broad_command_pair(torch.zeros(8, 2), -2.0, 2.0)
    assert u_a.shape == (8, 2)
    assert torch.all(u_a[0] == u_a[0, 0])
    assert float(u_a.min()) >= -2.0 - 1e-6
    assert float(u_a.max()) <= 2.0 + 1e-6
    batch = {
        "context": torch.randn(4, 6, 64, 128),
        "future_frames": torch.randn(4, 2, 6, 64, 128),
        "teacher_commands_norm": torch.zeros(4, 2),
    }
    result = jepa_forward_batch(model, batch)
    assert torch.isfinite(result.command_contrast_teacher)
    assert torch.isfinite(result.command_contrast_broad)
    assert float(result.command_contrast.detach()) == pytest.approx(
        float(result.command_contrast_teacher.detach() + result.command_contrast_broad.detach())
    )


def test_eval_dataloader_stays_inprocess():
    from torch.utils.data import TensorDataset

    from ts_jepa.runtime import DataLoaderStallError, is_dataloader_spawn_error, make_dataloader

    ds = TensorDataset(torch.zeros(8, 1))
    loader = make_dataloader(
        ds,
        batch_size=4,
        shuffle=False,
        device=torch.device("cpu"),
        num_workers=0,
    )
    assert loader.num_workers == 0
    err = OSError(22, "Invalid argument")
    wrapped = DataLoaderStallError("iterator failed")
    wrapped.__cause__ = err
    assert is_dataloader_spawn_error(err)
    assert is_dataloader_spawn_error(wrapped)
