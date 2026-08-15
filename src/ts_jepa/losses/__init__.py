"""Loss functions for TS-JEPA training."""

from ts_jepa.losses.jepa_loss import (
    cosine_alignment_loss,
    jepa_cosine_similarity,
    jepa_loss,
    vicreg_covariance_loss,
    vicreg_variance_loss,
)
from ts_jepa.losses.loss_plan import PLAN_JEPA_LOSS, assert_plan_jepa_loss_config

__all__ = [
    "PLAN_JEPA_LOSS",
    "assert_plan_jepa_loss_config",
    "cosine_alignment_loss",
    "jepa_cosine_similarity",
    "jepa_loss",
    "vicreg_covariance_loss",
    "vicreg_variance_loss",
]
