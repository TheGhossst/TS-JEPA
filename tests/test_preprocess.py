from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

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
    assert TRAINING_PIPELINE_STAGES == ("augmentation", "normalization", "resize")
    assert EVAL_PIPELINE_STAGES == ("normalization", "resize")
    assert config["input"]["resize"] == PLAN_PREPROCESSING["resize"]
    assert list(config["preprocessing"]["training"]["order"]) == list(TRAINING_PIPELINE_STAGES)
    assert list(config["preprocessing"]["evaluation"]["order"]) == list(EVAL_PIPELINE_STAGES)


def test_preprocess_shapes_train_and_test():
    config = load_config()
    frame = np.random.randint(0, 255, size=(128, 256, 3), dtype=np.uint8)
    train = PreprocessPipeline(config, training=True)
    test = PreprocessPipeline(config, training=False)
    t1 = train.process_frame(frame)
    t2 = test.process_frame(frame)
    assert t1.shape == (3, 64, 128)
    assert t2.shape == (3, 64, 128)

    frames = np.stack([frame, frame], axis=0)
    ctx = test.make_context_tensor(frames, time_index=1, kappa=2)
    assert ctx.shape == (6, 64, 128)
    jepa_frame = test.make_jepa_frame(frames, time_index=1)
    assert jepa_frame.shape == (3, 64, 128)
    assert torch.isfinite(ctx).all()


def test_eval_path_is_normalize_then_gaussian_resize():
    """Eval uses the same normalize→resize order as training (no stochastic aug)."""
    config = load_config()
    frame = np.random.randint(0, 255, size=(128, 256, 3), dtype=np.uint8)
    pipe = PreprocessPipeline(config, training=False)

    img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    expected = pipe._gaussian_resize(pipe._normalize(img), rng=None)

    actual = pipe.process_frame(frame)
    assert torch.allclose(actual, expected, atol=1e-5)
    assert torch.allclose(actual, pipe.process_frame(frame), atol=0.0)


def test_training_path_is_augment_normalize_then_gaussian_resize():
    """Plan §5 numbered list: jitter/drop → normalize → resize."""
    config = load_config()
    frame = np.random.randint(0, 255, size=(128, 256, 3), dtype=np.uint8)
    pipe = PreprocessPipeline(config, training=True)
    rng_expected = np.random.default_rng(7)
    rng_actual = np.random.default_rng(7)

    img = pipe._decode_frame(frame)
    img = pipe._augment(img, rng_expected)
    img = pipe._normalize(img)
    img = pipe._gaussian_resize(img, rng_expected)

    actual = pipe.process_frame(frame, stochastic=True, rng=rng_actual)
    assert torch.allclose(actual, img, atol=1e-5)


def test_training_path_applies_augmentation_and_blur():
    """Plan §5: training differs from eval and is stochastic under augmentation."""
    config = load_config()
    frame = np.random.randint(0, 255, size=(64, 128, 3), dtype=np.uint8)
    train = PreprocessPipeline(config, training=True)
    eval_pipe = PreprocessPipeline(config, training=False)

    eval_out = eval_pipe.process_frame(frame)
    rng = np.random.default_rng(0)
    train_out = train.process_frame(frame, stochastic=True, rng=rng)
    assert not torch.allclose(train_out, eval_out, atol=1e-3)

    train_repeat = train.process_frame(frame, stochastic=True, rng=np.random.default_rng(0))
    assert torch.allclose(train_out, train_repeat, atol=1e-6)


def test_training_skips_augmentation_when_stochastic_false():
    config = load_config()
    frame = np.random.randint(0, 255, size=(64, 128, 3), dtype=np.uint8)
    train = PreprocessPipeline(config, training=True)
    eval_pipe = PreprocessPipeline(config, training=False)
    assert torch.allclose(
        train.process_frame(frame, stochastic=False),
        eval_pipe.process_frame(frame),
        atol=1e-5,
    )


def test_command_normalization_plan_formula():
    """Plan §5: z-score normalize and inverse denormalize."""
    normalizer = CommandNormalizer(mean=2.0, std=4.0)
    u = np.array([-2.0, 2.0, 6.0], dtype=np.float32)
    u_norm = normalizer.normalize(u)
    assert np.allclose(u_norm, (u - 2.0) / 4.0)
    u_back = normalizer.denormalize(u_norm)
    assert np.allclose(u_back, u)


def test_gaussian_resize_is_kernel_resample_not_bilinear():
    config = load_config()
    frame = np.random.randint(0, 255, size=(128, 256, 3), dtype=np.uint8)
    pipe = PreprocessPipeline(config, training=False)
    img = pipe._decode_frame(frame)
    gauss = pipe._gaussian_resize(img, rng=None)
    bilinear = torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(64, 128), mode="bilinear", align_corners=False
    ).squeeze(0)
    assert gauss.shape == bilinear.shape
    assert not torch.allclose(gauss, bilinear, atol=1e-4)


def test_gaussian_resize_always_downsamples_native_frames():
    """Plan §5 resize must run on IC native 128×256, not skip as a no-op."""
    config = load_config()
    frame = np.random.randint(0, 255, size=(128, 256, 3), dtype=np.uint8)
    pipe = PreprocessPipeline(config, training=False)
    img = pipe._decode_frame(frame)
    out = pipe._gaussian_resize(img, rng=None)
    assert img.shape[-2:] == (128, 256)
    assert out.shape == (3, 64, 128)
    crop = img[:, :64, :128]
    assert not torch.allclose(out, crop, atol=1e-3)


def test_native_frame_resizes_to_encoder_hw():
    config = load_config()
    frame = np.random.randint(0, 255, size=(128, 256, 3), dtype=np.uint8)
    train = PreprocessPipeline(config, training=True)
    eval_pipe = PreprocessPipeline(config, training=False)
    assert train.process_frame(frame, stochastic=False).shape == (3, 64, 128)
    assert eval_pipe.process_frame(frame).shape == (3, 64, 128)


def test_plan_config_rejects_wrong_jitter():
    config = copy.deepcopy(load_config())
    config["input"]["color_jitter"]["brightness"] = 0.2
    with pytest.raises(ValueError, match="Plan §5 preprocessing"):
        assert_plan_preprocessing_config(config)
