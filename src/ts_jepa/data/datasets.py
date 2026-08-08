from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ts_jepa.config import project_root
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.preprocessing.pipeline import PreprocessPipeline


class TrajectoryDataset(Dataset):
    """Dataset over trajectory time steps for TS-JEPA training."""

    def __init__(
        self,
        trajectory_dir: Path,
        config: dict[str, Any],
        normalizer: CommandNormalizer,
        training: bool = True,
        kp: int | None = None,
    ) -> None:
        self.trajectory_dir = Path(trajectory_dir)
        self.config = config
        self.normalizer = normalizer
        self.training = training
        self.kp = kp or config["ts_jepa"]["prediction_horizon"]["Kp"]
        self.kappa = config["input"]["kappa"]
        self.trajectory_steps = config["simulation"]["trajectory_steps"]
        self.pipeline = PreprocessPipeline(config, training=training)
        self.files = sorted(self.trajectory_dir.glob("*.npz"))
        self.index_map: list[tuple[int, int]] = []
        for file_idx, file_path in enumerate(self.files):
            with np.load(file_path) as data:
                length = int(data["commands"].shape[0])
            max_start = length - self.kp - 1
            for time_index in range(max(0, max_start + 1)):
                self.index_map.append((file_idx, time_index))

    def __len__(self) -> int:
        return len(self.index_map)

    def _load_trajectory(self, file_idx: int) -> tuple[np.ndarray, np.ndarray]:
        file_path = self.files[file_idx]
        with np.load(file_path) as data:
            frames = data["frames"]
            commands = data["commands"]
        return frames, commands

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, time_index = self.index_map[index]
        frames, commands = self._load_trajectory(file_idx)

        context = self.pipeline.make_context_tensor(frames, time_index, self.kappa)
        future_contexts = []
        for offset in range(1, self.kp + 1):
            future_contexts.append(
                self.pipeline.make_context_tensor(frames, time_index + offset, self.kappa)
            )
        future_stack = torch.stack(future_contexts, dim=0)
        command_slice = commands[time_index : time_index + self.kp].astype(np.float32)
        command_norm = self.normalizer.normalize(command_slice)

        return {
            "context": context,
            "future_frames": future_stack,  # [Kp, C_kappa, H, W]
            "commands": torch.from_numpy(command_slice.copy()),
            "commands_norm": torch.from_numpy(command_norm.copy()),
            "time_index": torch.tensor(time_index, dtype=torch.long),
        }


class ActorEmbeddingDataset(Dataset):
    """Pairs of encoder embeddings and normalized control commands."""

    def __init__(
        self,
        trajectory_dir: Path,
        config: dict[str, Any],
        normalizer: CommandNormalizer,
        encoder: torch.nn.Module,
        device: torch.device,
        training: bool = True,
    ) -> None:
        self.trajectory_dir = Path(trajectory_dir)
        self.config = config
        self.normalizer = normalizer
        self.pipeline = PreprocessPipeline(config, training=False)
        self.kappa = config["input"]["kappa"]
        self.files = sorted(self.trajectory_dir.glob("*.npz"))
        self.samples: list[tuple[torch.Tensor, float]] = []

        encoder.eval()
        with torch.no_grad():
            for file_path in self.files:
                with np.load(file_path) as data:
                    frames = data["frames"]
                    commands = data["commands"]
                for time_index in range(len(commands)):
                    context = self.pipeline.make_context_tensor(frames, time_index, self.kappa)
                    embedding = encoder(context.unsqueeze(0).to(device)).squeeze(0).cpu()
                    command_norm = float(
                        self.normalizer.normalize(np.array([commands[time_index]], dtype=np.float32))[0]
                    )
                    self.samples.append((embedding, command_norm))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        embedding, command_norm = self.samples[index]
        return {
            "embedding": embedding,
            "command_norm": torch.tensor([command_norm], dtype=torch.float32),
        }


def fit_command_normalizer(config: dict[str, Any], data_root: Path | None = None) -> CommandNormalizer:
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    train_dir = root / "trajectories" / "jepa" / "train"
    commands = []
    for file_path in sorted(train_dir.glob("*.npz")):
        with np.load(file_path) as data:
            commands.append(data["commands"])
    all_commands = np.concatenate(commands, axis=0)
    normalizer = CommandNormalizer.from_array(all_commands)
    stats_dir = root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    with (stats_dir / "command_norm.json").open("w", encoding="utf-8") as handle:
        json.dump(normalizer.to_dict(), handle, indent=2)
    return normalizer


def load_command_normalizer(config: dict[str, Any], data_root: Path | None = None) -> CommandNormalizer:
    root = data_root or (project_root(config) / config["paths"]["data_root"])
    stats_path = root / "stats" / "command_norm.json"
    if not stats_path.exists():
        return fit_command_normalizer(config, data_root=root)
    with stats_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return CommandNormalizer.from_dict(payload)
