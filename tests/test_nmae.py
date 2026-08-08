from __future__ import annotations

import numpy as np

from ts_jepa.evaluation.metrics import nmae


def test_nmae_zero_when_equal():
    x = np.array([1.0, -2.0, 3.0])
    assert nmae(x, x) == 0.0


def test_nmae_positive_when_different():
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.0, 0.0, 3.0])
    assert nmae(pred, target) > 0.0
