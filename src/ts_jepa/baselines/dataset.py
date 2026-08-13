"""Frame / command / state samples for §16 supervised and autoencoder training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.preprocessing.pipeline import PreprocessPipeline


class FrameCommandStateDataset(Dataset):
    """κ-frame context, teacher command, and 4D plant state from JEPA trajectories."""

    def __init__(
        self,
        trajectory_dir: Path,
        config: dict[str, Any],
        normalizer: CommandNormalizer,
        *,
        kappa: int,
        training: bool,
        max_trajectories: int | None = None,
        file_list: list[Path] | None = None,
    ) -> None:
        self.config = config
        self.normalizer = normalizer
        self.kappa = int(kappa)
        cfg = dict(config)
        cfg = {**config, "input": {**config["input"], "kappa": self.kappa}}
        self.pipeline = PreprocessPipeline(cfg, training=training)
        files = file_list if file_list is not None else sorted(Path(trajectory_dir).glob("*.npz"))
        if max_trajectories is not None:
            files = files[: int(max_trajectories)]
        self.frames: list[np.ndarray] = []
        self.commands: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.index_map: list[tuple[int, int]] = []
        for file_idx, path in enumerate(files):
            with np.load(path) as data:
                frames = np.asarray(data["frames"])
                commands = np.asarray(data["commands"], dtype=np.float32)
                states = np.asarray(data["states"], dtype=np.float32)
            self.frames.append(frames)
            self.commands.append(commands)
            self.states.append(states)
            for t in range(int(commands.shape[0])):
                self.index_map.append((file_idx, t))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, t = self.index_map[index]
        frames = self.frames[file_idx]
        context = self.pipeline.make_context_tensor(frames, t, self.kappa)
        command = float(self.commands[file_idx][t])
        command_arr = np.asarray([command], dtype=np.float32)
        state = torch.from_numpy(np.asarray(self.states[file_idx][t], dtype=np.float32))
        return {
            "context": context,
            "command_norm": torch.from_numpy(self.normalizer.normalize(command_arr)),
            "command_phys": torch.from_numpy(command_arr),
            "state": state,
        }
