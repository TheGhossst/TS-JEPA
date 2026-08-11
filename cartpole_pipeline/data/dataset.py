"""PyTorch dataset for CartPole trajectory data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CartPoleFrameDataset(Dataset):
    """
    Dataset over single (frame, action) pairs from saved trajectory files.

    Expects a compressed .npz file with arrays:
        frames:  (num_episodes, episode_length, H, W, C) uint8
        actions: (num_episodes, episode_length, 1) float32

  Each __getitem__ returns one timestep's frame and action.
    """

    def __init__(self, data_path: str | Path, transform=None) -> None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        with np.load(data_path) as archive:
            frames = archive["frames"]
            actions = archive["actions"]

        if frames.ndim != 5:
            raise ValueError(f"Expected frames with shape (E, T, H, W, C), got {frames.shape}")
        if actions.ndim != 3 or actions.shape[-1] != 1:
            raise ValueError(f"Expected actions with shape (E, T, 1), got {actions.shape}")

        self.frames = frames
        self.actions = actions.astype(np.float32)
        self.transform = transform

        self.num_episodes, self.episode_length = self.frames.shape[:2]
        self._length = self.num_episodes * self.episode_length

    def __len__(self) -> int:
        return self._length

    def _flat_to_episode_step(self, index: int) -> tuple[int, int]:
        episode = index // self.episode_length
        step = index % self.episode_length
        return episode, step

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode, step = self._flat_to_episode_step(index)

        frame = self.frames[episode, step]  # (H, W, C) uint8
        action = self.actions[episode, step, 0]  # scalar

        if self.transform is not None:
            frame = self.transform(frame)
        else:
            # Convert to float tensor in CHW format, scaled to [0, 1].
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        action_tensor = torch.tensor(action, dtype=torch.float32)
        return frame, action_tensor
