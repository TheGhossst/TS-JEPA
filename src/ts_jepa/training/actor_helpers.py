"""Shared Semantic Actor training helpers (split + MSE eval)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from ts_jepa.models.actor import SemanticActor
from ts_jepa.runtime import CUDAPrefetcher, DataLoaderStallError, reraise_cuda_context


def split_train_val_actor(
    dataset: Dataset,
    val_fraction: float,
    *,
    seed: int = 0,
    mode: str = "contiguous",
) -> tuple[Subset, Subset]:
    """
    Hold out a fraction of the actor TRAIN embeddings for early stopping.

    Count/fraction is IMPLEMENTATION CHOICE. Untouched actor test trajectories
    are never used for model selection.

    ``contiguous`` (default): last N samples — leaks a trajectory-tail shift into val.
    ``shuffled``: random samples.
    ``shuffled_trajectories``: hold out whole trajectory files (preferred).
    """
    n = len(dataset)
    n_val = max(1, int(round(n * float(val_fraction))))
    n_val = min(n_val, max(1, n - 1)) if n > 1 else 1
    split_mode = str(mode or "contiguous").strip().lower()
    if split_mode in {"contiguous", "tail", ""}:
        indices = list(range(n))
        val_idx = indices[-n_val:]
        train_idx = indices[:-n_val] if n > n_val else indices
        return Subset(dataset, train_idx), Subset(dataset, val_idx)

    if split_mode in {"shuffled_trajectories", "trajectories", "by_trajectory"}:
        lengths = getattr(dataset, "file_lengths", None)
        if isinstance(lengths, list) and lengths and int(sum(int(x) for x in lengths)) == n:
            train_idx, val_idx = _trajectory_index_split(lengths, val_fraction, seed=seed)
            if train_idx and val_idx:
                return Subset(dataset, train_idx), Subset(dataset, val_idx)

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:] if n > n_val else perm
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _trajectory_index_split(
    lengths: list[int],
    val_fraction: float,
    *,
    seed: int,
) -> tuple[list[int], list[int]]:
    n_files = len(lengths)
    if n_files <= 1:
        return [], []
    n_val_files = max(1, int(round(n_files * float(val_fraction))))
    n_val_files = min(n_val_files, n_files - 1)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n_files)
    val_files = set(int(i) for i in order[:n_val_files])
    offsets = np.cumsum([0, *list(lengths[:-1])]).astype(int)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for file_i, length in enumerate(lengths):
        start = int(offsets[file_i])
        sl = range(start, start + int(length))
        (val_idx if file_i in val_files else train_idx).extend(sl)
    return train_idx, val_idx


# Backward-compatible alias used by diagnostics.
_split_train_val_actor = split_train_val_actor


def actor_train_recipe_id(config: Mapping[str, Any]) -> str:
    """Fingerprint of actor-train settings so stale checkpoints are retrained."""
    sa = config.get("semantic_actor") or {}
    arch = sa.get("architecture") or {}
    early = sa.get("early_stopping") or {}
    dagger = sa.get("dagger") or {}
    opt = sa.get("optimizer") or {}
    return "|".join(
        [
            str(sa.get("loss", "MSE")),
            str(sa.get("large_force_weight", 0.0)),
            str(sa.get("std_match_weight", 0.0)),
            str(early.get("split", "contiguous")),
            str(early.get("min_epochs", 0)),
            str(arch.get("layernorm", False)),
            str(arch.get("output_init_scale", 1.0)),
            str(arch.get("dropout", 0.2)),
            str(opt.get("learning_rate", "")),
            str(bool(dagger.get("enabled", False))),
            "v2",
        ]
    )


def actor_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    kind: str = "mse",
    huber_beta: float = 1.0,
    force_max: float = 20.0,
    large_force_weight: float = 0.0,
    std_match_weight: float = 0.0,
) -> torch.Tensor:
    """Physical-Newton regression. Large |u| samples can be upweighted to fight mean-shrinkage."""
    pred_f = pred.reshape(-1)
    target_f = target.reshape(-1)
    name = str(kind or "mse").strip().lower()
    if name in {"huber", "smooth_l1", "smoothl1"}:
        per = F.smooth_l1_loss(pred_f, target_f, beta=float(huber_beta), reduction="none")
    else:
        per = (pred_f - target_f) ** 2
    weight = float(large_force_weight)
    if weight > 0.0:
        scale = max(float(force_max), 1e-6)
        per = per * (1.0 + weight * (target_f.abs() / scale))
    loss = per.mean()
    match_w = float(std_match_weight)
    if match_w > 0.0 and int(pred_f.numel()) > 1:
        loss = loss + match_w * (pred_f.std() - target_f.std()).abs()
    return loss


def actor_loss_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    sa = config.get("semantic_actor") or {}
    sim = config.get("simulation") or {}
    force_max = abs(float(sim.get("control_max_N", 20.0)))
    return {
        "kind": str(sa.get("loss", "MSE")),
        "huber_beta": float(sa.get("huber_beta", sa.get("huber_delta", 4.0))),
        "force_max": force_max,
        "large_force_weight": float(sa.get("large_force_weight", 0.0)),
        "std_match_weight": float(sa.get("std_match_weight", 0.0)),
    }


@torch.no_grad()
def evaluate_mse(actor: SemanticActor, loader, device: torch.device, criterion: nn.Module) -> float:
    actor.eval()
    total = 0.0
    n_batches = 0
    prefetcher = CUDAPrefetcher(loader, device)
    try:
        for batch in prefetcher:
            emb = batch["embedding"]
            target = batch["command"]
            pred = actor(emb)
            total += float(criterion(pred, target).item())
            n_batches += 1
    except DataLoaderStallError:
        raise
    except Exception as exc:
        reraise_cuda_context(exc, where=f"evaluate_mse after {n_batches} batches", device=device)
    finally:
        prefetcher.close()
    return total / max(1, n_batches)


@torch.no_grad()
def evaluate_actor_objective(
    actor: SemanticActor,
    loader,
    device: torch.device,
    *,
    loss_kwargs: Mapping[str, Any],
) -> float:
    actor.eval()
    total = 0.0
    n_batches = 0
    prefetcher = CUDAPrefetcher(loader, device)
    try:
        for batch in prefetcher:
            emb = batch["embedding"]
            target = batch["command"]
            pred = actor(emb)
            total += float(actor_regression_loss(pred, target, **dict(loss_kwargs)).item())
            n_batches += 1
    except DataLoaderStallError:
        raise
    except Exception as exc:
        reraise_cuda_context(exc, where=f"evaluate_actor_objective after {n_batches} batches", device=device)
    finally:
        prefetcher.close()
    return total / max(1, n_batches)


def mean_command_baseline_mse(targets: np.ndarray, mean_value: float) -> float:
    """MSE of predicting a constant (train-mean) command in the actor loss domain (Newtons)."""
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return float("nan")
    return float(np.mean((y - float(mean_value)) ** 2))
