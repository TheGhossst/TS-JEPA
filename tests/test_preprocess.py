from __future__ import annotations

import numpy as np
import torch

from ts_jepa.config import load_config
from ts_jepa.preprocessing.pipeline import PreprocessPipeline


def test_preprocess_shapes_train_and_test():
    config = load_config()
    frame = np.random.randint(0, 255, size=(96, 192, 3), dtype=np.uint8)
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
