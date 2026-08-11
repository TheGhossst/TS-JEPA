"""
Plan §11 TS-JEPA cosine alignment loss.

Paper objective (per horizon step j):

    cos_sim(z̃_j, z̄_j) = (z̃_j · z̄_j) / (||z̃_j||₂ ||z̄_j||₂)

    L_JEPA = -(1/K_p) Σ_{j=1}^{K_p} cos_sim(z̃_j, z̄_j)

Target embeddings z̄_j come from the EMA target encoder with stop-gradient
(see `TSJEPA.encode_targets` and plan §7).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_bkp(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")
    if pred.ndim == 2:
        return pred.unsqueeze(1), target.unsqueeze(1)
    if pred.ndim != 3:
        raise ValueError(f"expected [B, Kp, D] or [B, D] tensors, got ndim={pred.ndim}")
    return pred, target


def jepa_cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Per-step cosine similarity cos_sim(z̃_j, z̄_j).

    Returns:
        Tensor of shape [B, Kp] (or [B, 1] when inputs are [B, D]).
    """
    pred_bkp, target_bkp = _as_bkp(pred, target)
    target_bkp = target_bkp.detach()
    return F.cosine_similarity(pred_bkp, target_bkp, dim=-1, eps=1e-8)


def jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Plan §11 JEPA loss L_JEPA = -mean_{b,j}(cos_sim(z̃_{b,j}, z̄_{b,j})).

    Equivalent to: mean over batch of -(1/Kp) Σ_j cos_sim for fixed Kp.
    Minimizing this loss maximizes cosine alignment.
    """
    cos = jepa_cosine_similarity(pred, target)
    return -cos.mean()


def cosine_alignment_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Minimizing form equivalent to `jepa_loss` (differs by constant +1).

    1 - mean(cos_sim) has the same gradients as -mean(cos_sim) and is sometimes
    easier to log (0 = perfect alignment). Config may record either implementation.
    """
    return 1.0 + jepa_loss(pred, target)
