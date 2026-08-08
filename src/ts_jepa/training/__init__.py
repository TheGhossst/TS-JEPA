"""Training entrypoints."""

from ts_jepa.training.train_actor import train_semantic_actor, train_semantic_actor_repetitions
from ts_jepa.training.train_jepa import train_ts_jepa, train_ts_jepa_repetitions

__all__ = [
    "train_ts_jepa",
    "train_ts_jepa_repetitions",
    "train_semantic_actor",
    "train_semantic_actor_repetitions",
]
