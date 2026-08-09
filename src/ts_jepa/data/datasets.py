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
