"""DAgger helpers for the frozen-z actor (no JEPA updates)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ts_jepa.config import load_config
from ts_jepa.evaluation.raw_state_actor import ActorFeatureDataset, train_feature_actor
from ts_jepa.evaluation.working_gates import nmae_label_source_from_actor_payload
from ts_jepa.inference.temporal_actor import pair_feature
from ts_jepa.training.dagger_actor import (
    concat_arrays,
    dagger_beta,
    dagger_settings,
    mix_execute_action,
    subsample_offline,
)


def test_pair_feature_concat_and_delta():
    z0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    z1 = np.array([1.5, 2.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(pair_feature(z1, None, "z"), z1)
    pair = pair_feature(z1, z0, "z_pair")
    assert pair.shape == (6,)
    np.testing.assert_allclose(pair[:3], z1)
    np.testing.assert_allclose(pair[3:], z0)
    delta = pair_feature(z1, z0, "z_delta")
    np.testing.assert_allclose(delta[3:], z1 - z0)


def test_dagger_beta_decays_and_clips():
    assert dagger_beta(0, 0.6, 0.7) == pytest.approx(0.6)
    assert dagger_beta(1, 0.6, 0.7) == pytest.approx(0.42)
    assert dagger_beta(20, 0.6, 0.5) < 1e-6
    with pytest.raises(ValueError):
        dagger_beta(-1, 0.6, 0.7)


def test_mix_execute_action_respects_beta():
    rng = np.random.default_rng(0)
    n = 4000
    n_teacher = sum(
        mix_execute_action(9.0, -3.0, 0.75, rng, mode="bernoulli")[1] for _ in range(n)
    )
    frac = n_teacher / n
    assert 0.72 < frac < 0.78
    u, used = mix_execute_action(9.0, -3.0, 1.0, np.random.default_rng(1), mode="bernoulli")
    assert used is True
    assert u == pytest.approx(9.0)
    u, used = mix_execute_action(9.0, -3.0, 0.0, np.random.default_rng(1), mode="bernoulli")
    assert used is False
    assert u == pytest.approx(-3.0)
    u, _ = mix_execute_action(10.0, 0.0, 0.8, np.random.default_rng(0), mode="convex")
    assert u == pytest.approx(8.0)


def test_subsample_and_concat_keep_alignment():
    z = np.arange(20, dtype=np.float32).reshape(10, 2)
    u = np.arange(10, dtype=np.float32)
    un = u / 2.0
    zs, us, uns = subsample_offline(z, u, un, max_samples=4, seed=3)
    assert zs.shape == (4, 2)
    assert us.shape == (4,)
    np.testing.assert_allclose(uns, us / 2.0)
    z2, u2, un2 = concat_arrays([(zs, us, uns), (z[:2], u[:2], un[:2])])
    assert z2.shape[0] == 6
    assert u2.shape[0] == 6
    assert un2.shape[0] == 6


def test_working_overlay_has_dagger_block():
    config = load_config("configs/ts_jepa_working.yaml")
    settings = dagger_settings(config)
    assert settings["rounds"] >= 1
    assert settings["run_dirname"] == "semantic_actor_working_dagger"


def test_train_feature_actor_accepts_init_state_dict():
    config = load_config("configs/ts_jepa_working.yaml")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 256)).astype(np.float32)
    y = rng.normal(size=(32,)).astype(np.float32)
    yn = y / 10.0
    ds = ActorFeatureDataset(x, y, yn)
    first = train_feature_actor(
        config,
        ds,
        ds,
        device=torch.device("cpu"),
        max_epochs=1,
        seed=0,
    )
    second = train_feature_actor(
        config,
        ds,
        ds,
        device=torch.device("cpu"),
        max_epochs=1,
        seed=1,
        init_state_dict=first["actor"].state_dict(),
        epoch_desc="dagger test epoch",
    )
    assert "best_val" in second
    assert second["feature_dim"] == 256


def test_lqr_dagger_checkpoint_uses_lqr_nmae_labels_not_dp():
    assert nmae_label_source_from_actor_payload({"actor": {}}) == "dp_teacher"
    assert nmae_label_source_from_actor_payload({"method": "dagger", "expert": "dp"}) == "dp_teacher"
    assert nmae_label_source_from_actor_payload({"method": "dagger", "expert": "lqr"}) == "lqr"
    assert (
        nmae_label_source_from_actor_payload(
            {"method": "dagger", "config": {"semantic_actor": {"dagger": {"expert": "lqr"}}}}
        )
        == "lqr"
    )
