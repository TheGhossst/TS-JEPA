"""Paper plan specifications and config assertions (docs/plan.md)."""

from ts_jepa.plan.encoder import PLAN_ENCODER, PLAN_TARGET_ENCODER, assert_plan_encoder_config
from ts_jepa.plan.environment import PLAN_ENVIRONMENT, assert_plan_environment_config
from ts_jepa.plan.loss import PLAN_JEPA_LOSS, assert_plan_jepa_loss_config
from ts_jepa.plan.predictor import PLAN_PREDICTOR, assert_plan_predictor_config
from ts_jepa.plan.preprocessing import (
    EVAL_PIPELINE_STAGES,
    PLAN_PREPROCESSING,
    TRAINING_PIPELINE_STAGES,
    assert_plan_preprocessing_config,
)
from ts_jepa.plan.procedure import PLAN_JEPA_PROCEDURE, assert_plan_jepa_procedure_config
from ts_jepa.plan.temporal import PLAN_TEMPORAL, assert_plan_temporal_config
from ts_jepa.plan.training import PLAN_JEPA_TRAINING, assert_plan_jepa_training_config

__all__ = [
    "EVAL_PIPELINE_STAGES",
    "PLAN_ENCODER",
    "PLAN_ENVIRONMENT",
    "PLAN_JEPA_LOSS",
    "PLAN_JEPA_PROCEDURE",
    "PLAN_JEPA_TRAINING",
    "PLAN_PREDICTOR",
    "PLAN_PREPROCESSING",
    "PLAN_TARGET_ENCODER",
    "PLAN_TEMPORAL",
    "TRAINING_PIPELINE_STAGES",
    "assert_plan_encoder_config",
    "assert_plan_environment_config",
    "assert_plan_jepa_loss_config",
    "assert_plan_jepa_procedure_config",
    "assert_plan_jepa_training_config",
    "assert_plan_predictor_config",
    "assert_plan_preprocessing_config",
    "assert_plan_temporal_config",
]
