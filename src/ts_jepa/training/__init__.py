"""Training entrypoints."""

from ts_jepa.training.jepa_optimizer import (
    apply_jepa_lr_decay,
    build_jepa_optimizer,
    jepa_learning_rate_at_epoch,
    jepa_trainable_parameters,
    should_apply_jepa_lr_decay,
)
from ts_jepa.training.jepa_procedure import (
    PLAN_JEPA_PROCEDURE,
    assert_plan_jepa_procedure_config,
    jepa_forward_batch,
    jepa_sgd_and_ema_step,
)
from ts_jepa.training.jepa_training_plan import PLAN_JEPA_TRAINING, assert_plan_jepa_training_config
from ts_jepa.training.train_actor import train_semantic_actor, train_semantic_actor_repetitions
from ts_jepa.training.train_jepa import train_ts_jepa, train_ts_jepa_repetitions

__all__ = [
    "PLAN_JEPA_PROCEDURE",
    "PLAN_JEPA_TRAINING",
    "assert_plan_jepa_procedure_config",
    "assert_plan_jepa_training_config",
    "apply_jepa_lr_decay",
    "build_jepa_optimizer",
    "jepa_forward_batch",
    "jepa_learning_rate_at_epoch",
    "jepa_sgd_and_ema_step",
    "jepa_trainable_parameters",
    "should_apply_jepa_lr_decay",
    "train_ts_jepa",
    "train_ts_jepa_repetitions",
    "train_semantic_actor",
    "train_semantic_actor_repetitions",
]
