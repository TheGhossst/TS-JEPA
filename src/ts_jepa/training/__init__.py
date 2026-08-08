"""Training entrypoints."""

from ts_jepa.training.train_actor import train_semantic_actor
from ts_jepa.training.train_jepa import train_ts_jepa

__all__ = ["train_ts_jepa", "train_semantic_actor"]
