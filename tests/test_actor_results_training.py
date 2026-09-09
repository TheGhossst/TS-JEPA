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
