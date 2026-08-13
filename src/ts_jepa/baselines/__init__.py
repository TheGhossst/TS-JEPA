"""Plan §16–§17 control baselines and reported experiments."""

from ts_jepa.baselines.controllers import (
    AutoencoderDPController,
    NonlinearDPController,
    SupervisedController,
    ZeroActionController,
)
from ts_jepa.baselines.experiments import (
    apply_embedding_dim,
    evaluate_agent_closed_loop,
    evaluate_fig6_prediction_modes,
    evaluate_with_scheduler_agent,
    max_devices_for_score_band,
)
from ts_jepa.baselines.load import build_dp_controller, load_autoencoder_controller, load_supervised_controller
from ts_jepa.baselines.models import GenerativeAutoencoder, SupervisedRGBToCommand
from ts_jepa.baselines.train import train_autoencoder_baseline, train_supervised_baseline

__all__ = [
    "AutoencoderDPController",
    "GenerativeAutoencoder",
    "NonlinearDPController",
    "SupervisedController",
    "SupervisedRGBToCommand",
    "ZeroActionController",
    "apply_embedding_dim",
    "build_dp_controller",
    "evaluate_agent_closed_loop",
    "evaluate_fig6_prediction_modes",
    "evaluate_with_scheduler_agent",
    "load_autoencoder_controller",
    "load_supervised_controller",
    "max_devices_for_score_band",
    "train_autoencoder_baseline",
    "train_supervised_baseline",
]
