from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_alignment_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Minimizing form that maximizes cosine alignment.

    Paper objective: cosine similarity alignment.
    Implementation: 1 - cosine_similarity (not the literal printed argmin expression).
    """
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1]).detach()
    cos = F.cosine_similarity(pred_flat, target_flat, dim=-1)
    return (1.0 - cos).mean()
