"""Tests for observation observability diagnostics."""

from __future__ import annotations

import numpy as np

from ts_jepa.diagnostics.observability import (
    hash_frame,
    hash_raw_context,
    raw_context_frame_indices,
)
from ts_jepa.env.sampling import observation_stride_steps


def test_raw_context_frame_indices_padding():
    assert raw_context_frame_indices(0, 2, 10) == [0, 0]
    assert raw_context_frame_indices(1, 2, 10) == [0, 1]
    assert raw_context_frame_indices(5, 4, 10) == [2, 3, 4, 5]


def test_kappa_context_hash_changes_when_any_frame_changes():
    frames = np.zeros((4, 8, 16, 3), dtype=np.uint8)
    frames[1, 0, 0, 0] = 255
    h0 = hash_raw_context(frames, 1, kappa=2)
    h1 = hash_raw_context(frames, 2, kappa=2)
    assert h0 != h1


def test_identical_single_frames_produce_identical_kappa2_context_when_consecutive():
    frame = np.full((8, 16, 3), 128, dtype=np.uint8)
    frames = np.stack([frame, frame, frame], axis=0)
    assert hash_frame(frames[0]) == hash_frame(frames[1])
    # At t=1, context is [frame0, frame1] — both identical
    assert hash_raw_context(frames, 1, kappa=2) == hash_raw_context(frames, 2, kappa=2)


def test_observation_stride_includes_2ms():
    dt = 0.001
    assert observation_stride_steps(2, dt) == 2
