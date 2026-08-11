"""Sequence dataset for TS-JEPA representation learning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cartpole_pipeline.transforms import TSJEPATransform

ACTION_SCALE = 20.0


class TSJEPADataset(Dataset):
    """
    Dataset of (current frame, future frame sequence, intervening actions) tuples.

    Expects a compressed .npz file with arrays:
        frames:  (num_episodes, episode_length, H, W, C) uint8
        actions: (num_episodes, episode_length, 1) float32

    For each sample with start time ``t`` and horizon ``Kp``:
        - current frame at time ``t``
        - future frames from ``t + 1`` to ``t + Kp`` (shape ``Kp, 3, H, W``)
        - normalized actions from ``t`` to ``t + Kp - 1`` (shape ``Kp, 1``)
    """

    def __init__(
        self,
        data_path: str | Path,
        kp: int = 5,
        training: bool = True,
    ) -> None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        if kp < 1:
            raise ValueError(f"Prediction horizon Kp must be >= 1, got {kp}")

        with np.load(data_path) as archive:
            frames = archive["frames"]
            actions = archive["actions"]

        if frames.ndim != 5:
            raise ValueError(f"Expected frames with shape (E, T, H, W, C), got {frames.shape}")
        if actions.ndim != 3 or actions.shape[-1] != 1:
            raise ValueError(f"Expected actions with shape (E, T, 1), got {actions.shape}")

        self.frames = frames
        self.actions = actions.astype(np.float32)
        self.kp = kp
        self.training = training

        self.transform = TSJEPATransform()
        if training:
            self.transform.train()
        else:
            self.transform.eval()

        self.num_episodes, self.episode_length = self.frames.shape[:2]
        if self.episode_length <= kp:
            raise ValueError(
                f"Episode length ({self.episode_length}) must be greater than Kp ({kp})"
            )

        self.samples_per_episode = self.episode_length - kp
        self._length = self.num_episodes * self.samples_per_episode

    def __len__(self) -> int:
        return self._length

    def _index_to_episode_and_time(self, index: int) -> tuple[int, int]:
        episode = index // self.samples_per_episode
        start_time = index % self.samples_per_episode
        return episode, start_time

    @staticmethod
    def _frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
        """Convert HWC or THWC uint8 frames to CHW or TCHW float tensors in [0, 1]."""
        if frames.ndim == 3:
            return torch.from_numpy(frames).permute(2, 0, 1).float() / 255.0
        return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

    def _apply_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply TSJEPATransform to a single frame or a sequence of frames."""
        if frames.ndim == 3:
            return self.transform(frames.unsqueeze(0)).squeeze(0)
        return self.transform(frames)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        episode, start_time = self._index_to_episode_and_time(index)
        future_end = start_time + self.kp + 1

        current_frame = self._frames_to_tensor(self.frames[episode, start_time])
        future_frames = self._frames_to_tensor(self.frames[episode, start_time + 1:future_end])
        future_actions = torch.from_numpy(
            self.actions[episode, start_time:start_time + self.kp]
        ).float() / ACTION_SCALE

        current_frame = self._apply_transform(current_frame)
        future_frames = self._apply_transform(future_frames)

        return current_frame, future_frames, future_actions
