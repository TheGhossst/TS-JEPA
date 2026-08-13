"""Backward-compatible re-export of plan §10 JEPA loss."""

from __future__ import annotations

from ts_jepa.losses.jepa_loss import (
    cosine_alignment_loss,
    jepa_cosine_similarity,
    jepa_loss,
)

__all__ = ["cosine_alignment_loss", "jepa_cosine_similarity", "jepa_loss"]
