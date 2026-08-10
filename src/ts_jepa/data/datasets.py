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
        self.kappa = int(config["input"]["kappa"])
        self.trajectory_steps = config["simulation"]["trajectory_steps"]
        self.pipeline = PreprocessPipeline(config, training=training)
        self.files = sorted(self.trajectory_dir.glob("*.npz"))
        self.frames: list[np.ndarray] = []
        self.commands: list[np.ndarray] = []
        self.index_map: list[tuple[int, int]] = []
        for file_idx, file_path in enumerate(self.files):
            with np.load(file_path) as data:
                frames = np.asarray(data["frames"])
                commands = np.asarray(data["commands"], dtype=np.float32)
            self.frames.append(frames)
            self.commands.append(commands)
            length = int(commands.shape[0])
            max_start = length - self.kp - 1
            for time_index in range(max(0, max_start + 1)):
                self.index_map.append((file_idx, time_index))

        # Deterministic eval path: cache resized+normalized RGB frames once.
        self._eval_cache: list[list[torch.Tensor] | None] = [None] * len(self.files)
        if not training:
            for file_idx, frames in enumerate(self.frames):
                self._eval_cache[file_idx] = [
                    self.pipeline.process_frame(frames[t], stochastic=False) for t in range(frames.shape[0])
                ]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_idx, time_index = self.index_map[index]
        frames = self.frames[file_idx]
        commands = self.commands[file_idx]

        if self._eval_cache[file_idx] is not None:
            processed = {t: self._eval_cache[file_idx][t] for t in range(frames.shape[0])}
        else:
            start = max(0, time_index - self.kappa + 1)
            end = time_index + self.kp
            processed = self.pipeline.process_frames_cached(frames, start, end, stochastic=self.training)

        context = self.pipeline.assemble_context(processed, time_index, kappa=self.kappa)
        future_stack = torch.stack(
            [
                self.pipeline.assemble_context(processed, time_index + offset, kappa=self.kappa)
                for offset in range(1, self.kp + 1)
            ],
            dim=0,
        )
        # Trajectory/teacher control sequence from the DP teacher dataset.
        # This is NOT Semantic Actor-predicted command ũ during JEPA training.
        teacher_commands = commands[time_index : time_index + self.kp].astype(np.float32)
        teacher_commands_norm = self.normalizer.normalize(teacher_commands)

        return {
            "context": context,
            "future_frames": future_stack,  # [Kp, C_kappa, H, W]
            "teacher_commands": torch.from_numpy(teacher_commands.copy()),
            "teacher_commands_norm": torch.from_numpy(teacher_commands_norm.copy()),
            "time_index": torch.tensor(time_index, dtype=torch.long),
        }


def assemble_state_context(states: np.ndarray, time_index: int, kappa: int) -> np.ndarray:
    """
    κ-length raw-state window ending at time_index (same framing as assemble_context).

    states: [T, S] physical state (x, x_dot, theta, theta_dot).
    Returns flattened float32 vector of shape [kappa * S], left-padded by repeating
    the first available state when time_index < kappa - 1.
    """
    k = int(kappa)
    start = max(0, int(time_index) - k + 1)
    selected = [states[t] for t in range(start, int(time_index) + 1)]
    while len(selected) < k:
        selected.insert(0, selected[0])
    return np.concatenate(selected, axis=0).astype(np.float32)


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
        self.kappa = int(config["input"]["kappa"])
        self.files = sorted(self.trajectory_dir.glob("*.npz"))
        self.samples: list[tuple[torch.Tensor, float]] = []
        self.feature_dim: int | None = None

        encoder.eval()
        with torch.no_grad():
            for file_path in self.files:
                with np.load(file_path) as data:
                    frames = np.asarray(data["frames"])
                    commands = np.asarray(data["commands"], dtype=np.float32)
                cached = [self.pipeline.process_frame(frames[t], stochastic=False) for t in range(len(commands))]
                batch_contexts = []
                batch_cmds = []
                for time_index in range(len(commands)):
                    processed = {t: cached[t] for t in range(len(cached))}
                    context = self.pipeline.assemble_context(processed, time_index, kappa=self.kappa)
                    batch_contexts.append(context)
                    batch_cmds.append(float(self.normalizer.normalize(np.array([commands[time_index]], dtype=np.float32))[0]))
                # Encode in mini-batches for speed.
                bs = 64
                for start in range(0, len(batch_contexts), bs):
                    chunk = torch.stack(batch_contexts[start : start + bs], dim=0).to(device)
                    emb = encoder(chunk).cpu()
                    if self.feature_dim is None:
                        self.feature_dim = int(emb.shape[-1])
                    for i in range(emb.shape[0]):
                        self.samples.append((emb[i], batch_cmds[start + i]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        embedding, command_norm = self.samples[index]
        return {
            "embedding": embedding,
            "command_norm": torch.tensor([command_norm], dtype=torch.float32),
        }


class ActorStateDataset(Dataset):
    """
    Pairs of κ-window raw physical state features and normalized control commands.

    Same trajectory / time_index / command pairing as ActorEmbeddingDataset, but
    bypasses the JEPA encoder: actor input is flatten([s_{t-κ+1}, ..., s_t]).
    Dict keys match ActorEmbeddingDataset so the training loop is unchanged.
    """

    def __init__(
        self,
        trajectory_dir: Path,
        config: dict[str, Any],
        normalizer: CommandNormalizer,
        training: bool = True,
    ) -> None:
        self.trajectory_dir = Path(trajectory_dir)
        self.config = config
        self.normalizer = normalizer
        self.training = training
        self.kappa = int(config["input"]["kappa"])
        self.files = sorted(self.trajectory_dir.glob("*.npz"))
        self.samples: list[tuple[torch.Tensor, float]] = []
        self.feature_dim: int | None = None
        self.state_dim: int | None = None

        for file_path in self.files:
            with np.load(file_path) as data:
                states = np.asarray(data["states"], dtype=np.float32)
                commands = np.asarray(data["commands"], dtype=np.float32)
            if states.ndim != 2:
                raise ValueError(f"Expected states [T, S] in {file_path}, got shape {states.shape}")
            if self.state_dim is None:
                self.state_dim = int(states.shape[1])
                self.feature_dim = int(self.kappa * self.state_dim)
            elif int(states.shape[1]) != self.state_dim:
                raise ValueError(
                    f"Inconsistent state dim in {file_path}: {states.shape[1]} vs {self.state_dim}"
                )
            for time_index in range(len(commands)):
                features = assemble_state_context(states, time_index, self.kappa)
                cmd_norm = float(
                    self.normalizer.normalize(np.array([commands[time_index]], dtype=np.float32))[0]
                )
                self.samples.append((torch.from_numpy(features.copy()), cmd_norm))

        if self.feature_dim is None:
            raise ValueError(f"No trajectory files found under {self.trajectory_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        features, command_norm = self.samples[index]
        return {
            "embedding": features,
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
