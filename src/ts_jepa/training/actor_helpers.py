"""Shared Semantic Actor training helpers (split + MSE eval)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset

from ts_jepa.models.actor import SemanticActor
from ts_jepa.runtime import CUDAPrefetcher, DataLoaderStallError, reraise_cuda_context


def split_train_val_actor(dataset: Dataset, val_fraction: float) -> tuple[Subset, Subset]:
    """
    Hold out a fraction of the actor TRAIN embeddings for early stopping.

    Count/fraction is IMPLEMENTATION CHOICE. Untouched actor test trajectories
    are never used for model selection.
    """
    n = len(dataset)
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, max(1, n - 1)) if n > 1 else 1
    indices = list(range(n))
    val_idx = indices[-n_val:]
    train_idx = indices[:-n_val] if n > n_val else indices
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# Backward-compatible alias used by diagnostics.
_split_train_val_actor = split_train_val_actor


@torch.no_grad()
def evaluate_mse(actor: SemanticActor, loader, device: torch.device, criterion: nn.Module) -> float:
    actor.eval()
    total = 0.0
    n_batches = 0
    try:
        for batch in CUDAPrefetcher(loader, device):
            emb = batch["embedding"]
            target = batch["command"]
            pred = actor(emb)
            total += float(criterion(pred, target).item())
            n_batches += 1
    except DataLoaderStallError:
        raise
    except Exception as exc:
        reraise_cuda_context(exc, where=f"evaluate_mse after {n_batches} batches", device=device)
    return total / max(1, n_batches)


def mean_command_baseline_mse(targets: np.ndarray, mean_value: float) -> float:
    """MSE of predicting a constant (train-mean) command in the actor loss domain (Newtons)."""
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return float("nan")
    return float(np.mean((y - float(mean_value)) ** 2))
