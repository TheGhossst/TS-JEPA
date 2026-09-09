"""Actor results-recipe helpers: weighted loss and trajectory val split."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ts_jepa.config import load_config
from ts_jepa.plan.enforce import plan_enforced
from ts_jepa.training.actor_helpers import (
    actor_regression_loss,
    actor_train_recipe_id,
    split_train_val_actor,
)


class _LenDataset(Dataset):
    def __init__(self, n: int, file_lengths: list[int] | None = None) -> None:
        self.n = n
        self.file_lengths = file_lengths or []

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> int:
        return index


def test_contiguous_split_still_uses_tail():
    ds = _LenDataset(10)
    train, val = split_train_val_actor(ds, 0.2, mode="contiguous")
    assert list(val.indices) == [8, 9]
    assert list(train.indices) == list(range(8))


def test_shuffled_trajectory_split_holds_out_whole_files():
    lengths = [4, 4, 4, 4]
    ds = _LenDataset(16, file_lengths=lengths)
    train, val = split_train_val_actor(ds, 0.25, seed=0, mode="shuffled_trajectories")
    train_idx = set(train.indices)
    val_idx = set(val.indices)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx | val_idx == set(range(16))
    # Each file occupies a contiguous 4-sample block.
    blocks = [set(range(i, i + 4)) for i in range(0, 16, 4)]
    val_blocks = [b for b in blocks if b <= val_idx]
    assert len(val_blocks) == 1
    assert val_idx == val_blocks[0]


def test_weighted_huber_upweights_large_forces():
    pred = torch.zeros(2)
    small = torch.tensor([0.0, 1.0])
    large = torch.tensor([0.0, 20.0])
    kwargs = dict(kind="huber", huber_beta=4.0, force_max=20.0, large_force_weight=10.0)
    loss_small = actor_regression_loss(pred, small, **kwargs)
    loss_large = actor_regression_loss(pred, large, **kwargs)
    assert float(loss_large) > float(loss_small)


def test_std_match_term_penalizes_collapsed_predictions():
    target = torch.tensor([-20.0, 20.0, -8.0, 8.0])
    collapsed = torch.zeros(4)
    matched = target.clone()
    kwargs = dict(kind="mse", large_force_weight=0.0, std_match_weight=1.0)
    loss_collapsed = actor_regression_loss(collapsed, target, **kwargs)
    loss_matched = actor_regression_loss(matched, target, **kwargs)
    assert float(loss_collapsed) > float(loss_matched)


def test_dp_fixed_recipe_id_differs_from_baseline():
    paper = load_config()
    results = load_config("configs/ts_jepa_dp_fixed.yaml")
    assert plan_enforced(results) is False
    assert actor_train_recipe_id(paper) != actor_train_recipe_id(results)


def test_dp_fixed_control_hold_is_20ms_not_dataset_stride():
    from ts_jepa.evaluation.evaluate import _observation_stride, control_loop_stride

    paper = load_config()
    results = load_config("configs/ts_jepa_dp_fixed.yaml")
    assert _observation_stride(paper) == 1
    assert control_loop_stride(paper) == 1
    assert _observation_stride(results) == 1
    assert control_loop_stride(results) == 20


def test_receive_every_kp_mask_matches_prediction_horizon():
    from ts_jepa.evaluation.evaluate import receive_every_kp_mask

    mask = receive_every_kp_mask(30, 15)
    assert mask[0] is True
    assert mask[1] is False
    assert mask[15] is True
    assert mask.count(True) == 2


def test_stability_mean_passes_when_one_seed_is_zero(monkeypatch):
    """Seed 101 scoring 0 must not fail the gate if the 5-seed mean is > 0."""
    from ts_jepa.evaluation.evaluate import (
        closed_loop_eval_seeds,
        evaluate_closed_loop_stability,
        receive_every_kp_mask,
    )

    config = load_config("configs/ts_jepa_dp_fixed.yaml")
    seeds = closed_loop_eval_seeds(config)
    assert seeds == [100, 101, 102, 103, 104]
    calls: list[dict] = []

    def fake_closed_loop(cfg, controller, steps=None, seed=0, packet_receive_mask=None):
        del cfg, controller, steps
        calls.append(
            {
                "seed": int(seed),
                "mask": None if packet_receive_mask is None else list(packet_receive_mask),
            }
        )
        score = 0.0 if int(seed) == 101 else 0.1
        if packet_receive_mask is not None:
            score = 0.0 if int(seed) == 101 else 0.05
        return {
            "mean_control_score": score,
            "forces": [1.0],
            "scores": [score],
        }

    monkeypatch.setattr(
        "ts_jepa.evaluation.evaluate.evaluate_closed_loop", fake_closed_loop
    )
    out = evaluate_closed_loop_stability(config, controller=None)  # type: ignore[arg-type]
    assert out["loss_pattern"] == "receive_every_kp"
    assert out["full_receive"]["per_seed"][1] == 0.0
    assert out["full_receive"]["mean_control_score"] > 0.0
    assert out["intermittent_loss"]["mean_control_score"] > 0.0
    assert out["passed"] is True
    lossy = [c for c in calls if c["mask"] is not None]
    assert [c["seed"] for c in lossy] == seeds
    steps = int(config["simulation"]["trajectory_steps"])
    kp = int(config["ts_jepa"]["prediction_horizon"]["Kp"])
    expected = receive_every_kp_mask(steps, kp)
    assert all(c["mask"] == expected for c in lossy)
