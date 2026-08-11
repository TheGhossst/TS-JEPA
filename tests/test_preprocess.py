from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ts_jepa.config import load_config
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.preprocessing.plan import (
    EVAL_PIPELINE_STAGES,
    PLAN_PREPROCESSING,
    TRAINING_PIPELINE_STAGES,
    assert_plan_preprocessing_config,
)
from ts_jepa.preprocessing.pipeline import PreprocessPipeline


def test_plan_preprocessing_config_matches_baseline_yaml():
    config = load_config()
    assert_plan_preprocessing_config(config)
    assert TRAINING_PIPELINE_STAGES == ("augmentation", "normalization", "formatting")
    assert EVAL_PIPELINE_STAGES == ("resize", "normalization")
    assert config["input"]["resize"] == PLAN_PREPROCESSING["resize"]


def test_preprocess_shapes_train_and_test():
    config = load_config()
    frame = np.random.randint(0, 255, size=(64, 128, 3), dtype=np.uint8)
    train = PreprocessPipeline(config, training=True)
    test = PreprocessPipeline(config, training=False)
    t1 = train.process_frame(frame)
    t2 = test.process_frame(frame)
    assert t1.shape == (3, 64, 128)
    assert t2.shape == (3, 64, 128)

    frames = np.stack([frame, frame], axis=0)
    ctx = test.make_context_tensor(frames, time_index=1, kappa=2)
    assert ctx.shape == (6, 64, 128)
    assert torch.isfinite(ctx).all()


def test_eval_path_is_resize_then_normalize_only():
    """Plan §5.2: no augmentations or blur during evaluation."""
    config = load_config()
    frame = np.random.randint(0, 255, size=(64, 128, 3), dtype=np.uint8)
    pipe = PreprocessPipeline(config, training=False)

    img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    resized = F.interpolate(img.unsqueeze(0), size=(64, 128), mode="bilinear", align_corners=False).squeeze(0)
    mean = torch.tensor(PLAN_PREPROCESSING["normalization"]["mean"]).view(3, 1, 1)
    std = torch.tensor(PLAN_PREPROCESSING["normalization"]["std"]).view(3, 1, 1)
    expected = (resized - mean) / std

    actual = pipe.process_frame(frame)
    assert torch.allclose(actual, expected, atol=1e-5)

    # Deterministic across repeated eval calls.
    assert torch.allclose(actual, pipe.process_frame(frame), atol=0.0)


def test_training_path_applies_augmentation_and_blur():
    """Plan §5.1: training differs from eval and is stochastic under augmentation."""
    config = load_config()
    frame = np.random.randint(0, 255, size=(64, 128, 3), dtype=np.uint8)
    train = PreprocessPipeline(config, training=True)
    eval_pipe = PreprocessPipeline(config, training=False)

    eval_out = eval_pipe.process_frame(frame)
    rng = np.random.default_rng(0)
    train_out = train.process_frame(frame, stochastic=True, rng=rng)
    assert not torch.allclose(train_out, eval_out, atol=1e-3)

    # Same seed → same training output.
    train_repeat = train.process_frame(frame, stochastic=True, rng=np.random.default_rng(0))
    assert torch.allclose(train_out, train_repeat, atol=1e-6)


def test_training_skips_augmentation_when_stochastic_false():
    config = load_config()
    frame = np.random.randint(0, 255, size=(64, 128, 3), dtype=np.uint8)
    train = PreprocessPipeline(config, training=True)
    eval_pipe = PreprocessPipeline(config, training=False)
    # stochastic=False on a training pipeline follows eval path (resize → normalize).
    assert torch.allclose(
        train.process_frame(frame, stochastic=False),
        eval_pipe.process_frame(frame),
        atol=1e-5,
    )


def test_command_normalization_plan_formula():
    """Plan §5.3: z-score normalize and inverse denormalize."""
    normalizer = CommandNormalizer(mean=2.0, std=4.0)
    u = np.array([-2.0, 2.0, 6.0], dtype=np.float32)
    u_norm = normalizer.normalize(u)
    assert np.allclose(u_norm, (u - 2.0) / 4.0)
    u_back = normalizer.denormalize(u_norm)
    assert np.allclose(u_back, u)


def test_plan_config_rejects_wrong_jitter():
    config = copy.deepcopy(load_config())
    config["input"]["color_jitter"]["brightness"] = 0.2
    with pytest.raises(ValueError, match="Plan §5 preprocessing"):
        assert_plan_preprocessing_config(config)
