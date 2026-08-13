from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ts_jepa.config import project_root
from ts_jepa.data.temporal import (
    command_indices,
    max_valid_time_index,
    predicted_command_indices,
    target_end_indices,
)
from ts_jepa.preprocessing.command_stats import CommandNormalizer
from ts_jepa.preprocessing.pipeline import PreprocessPipeline


def assert_native_frame_hw(frames: np.ndarray, config: dict[str, Any], *, source: str = "") -> None:
    """Stored RGB must be the IC native camera size, not the 64×128 encoder input."""
    exp_h = int(config["simulation"]["render_height"])
    exp_w = int(config["simulation"]["render_width"])
    enc_h, enc_w = [int(x) for x in config["input"]["resize"]]
    where = f" ({source})" if source else ""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames{where} must be [T,H,W,3], got {getattr(frames, 'shape', None)}")
    got_h, got_w = int(frames.shape[1]), int(frames.shape[2])
    if (got_h, got_w) != (exp_h, exp_w):
        raise ValueError(
            f"Stored frames{where} have spatial size {got_h}×{got_w}; config native render is "
            f"{exp_h}×{exp_w} (IC). Encoder input is {enc_h}×{enc_w} after Gaussian resize. "
            "Regenerate trajectories with scripts/pipeline/generate_trajectories.py."
        )
    if (got_h, got_w) == (enc_h, enc_w) and (exp_h, exp_w) != (enc_h, enc_w):
        raise ValueError(
            f"Stored frames{where} are already encoder size {enc_h}×{enc_w}; native render "
            f"must be {exp_h}×{exp_w} so the plan §5 resize actually downsamples."
        )


class TrajectoryDataset(Dataset):
    """
    Dataset over trajectory time steps for TS-JEPA training.

    Algorithm 1 sample at time index k:
      context:  single RGB frame x_k           → [3, 64, 128]
      controls: u_k .. u_{k+Kp-1}             → teacher_commands[Kp]
      targets:  frames x_{k+1} .. x_{k+Kp}    → future_frames[Kp, 3, H, W]
    """

    def __init__(
        self,
        trajectory_dir: Path,
        config: dict[str, Any],
        normalizer: CommandNormalizer,
        training: bool = True,
        kp: int | None = None,
        files: list[Path] | None = None,
    ) -> None:
        self.trajectory_dir = Path(trajectory_dir)
        self.config = config
        self.normalizer = normalizer
        self.training = training
        self.kp = kp or config["ts_jepa"]["prediction_horizon"]["Kp"]
        self.trajectory_steps = config["simulation"]["trajectory_steps"]
        self.pipeline = PreprocessPipeline(config, training=training)
        self.files = files if files is not None else sorted(self.trajectory_dir.glob("*.npz"))
        self.frames: list[np.ndarray] = []
        self.commands: list[np.ndarray] = []
        self.index_map: list[tuple[int, int]] = []
        for file_idx, file_path in enumerate(self.files):
            with np.load(file_path) as data:
                frames = np.asarray(data["frames"])
                commands = np.asarray(data["commands"], dtype=np.float32)
            assert_native_frame_hw(frames, config, source=str(file_path))
            self.frames.append(frames)
            self.commands.append(commands)
            length = int(commands.shape[0])
            max_start = max_valid_time_index(length, self.kp)
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
            start = time_index
            end = time_index + self.kp
            processed = self.pipeline.process_frames_cached(frames, start, end, stochastic=self.training)

        context = self.pipeline.assemble_jepa_frame(processed, time_index)
        future_stack = torch.stack(
            [
                self.pipeline.assemble_jepa_frame(processed, end_t)
                for end_t in target_end_indices(time_index, self.kp)
            ],
            dim=0,
        )
        # Plan §6: predictor conditioning [u_k, u_{k+1}, ..., u_{k+Kp-1}]
        cmd_idx = command_indices(time_index, self.kp)
        teacher_commands = commands[cmd_idx[0] : cmd_idx[0] + self.kp].astype(np.float32)
        teacher_commands_norm = self.normalizer.normalize(teacher_commands)
        # Horizon NMAE: actor(z̃_{k+j}) vs u_{k+j} for j=1..Kp
        tgt_cmd_idx = predicted_command_indices(time_index, self.kp)
        target_commands = commands[tgt_cmd_idx[0] : tgt_cmd_idx[0] + self.kp].astype(np.float32)
        target_commands_norm = self.normalizer.normalize(target_commands)

        return {
            "context": context,
            "future_frames": future_stack,  # [Kp, 3, H, W]
            "teacher_commands": torch.from_numpy(teacher_commands.copy()),
            "teacher_commands_norm": torch.from_numpy(teacher_commands_norm.copy()),
            "target_commands": torch.from_numpy(target_commands.copy()),
            "target_commands_norm": torch.from_numpy(target_commands_norm.copy()),
            "time_index": torch.tensor(time_index, dtype=torch.long),
        }


class ActorEmbeddingDataset(Dataset):
    """
    Plan §12 D_a: precomputed context-encoder embeddings paired with commands.

    Embeddings are z_{i,k} = Ψθ(x_{i,k}) from a frozen encoder (Algorithm 1,
    one RGB frame). Commands are stored in physical Newtons for Eq. 15 MSE.
    """

    def __init__(
        self,
        trajectory_dir: Path,
        config: dict[str, Any],
        normalizer: CommandNormalizer,
        encoder: torch.nn.Module,
        device: torch.device,
        training: bool = True,
    ) -> None:
        del training  # embeddings are fixed; no stochastic RGB augmentation
        self.trajectory_dir = Path(trajectory_dir)
        self.config = config
        self.normalizer = normalizer
        self.pipeline = PreprocessPipeline(config, training=False)
        self.files = sorted(self.trajectory_dir.glob("*.npz"))
        self.samples: list[tuple[torch.Tensor, float, float]] = []

        encoder.eval()
        with torch.no_grad():
            for file_path in self.files:
                with np.load(file_path) as data:
                    frames = np.asarray(data["frames"])
                    commands = np.asarray(data["commands"], dtype=np.float32)
                assert_native_frame_hw(frames, config, source=str(file_path))
                cached = [self.pipeline.process_frame(frames[t], stochastic=False) for t in range(len(commands))]
                batch_contexts = []
                batch_cmds_phys = []
                batch_cmds_norm = []
                for time_index in range(len(commands)):
                    batch_contexts.append(cached[time_index])
                    phys = float(commands[time_index])
                    batch_cmds_phys.append(phys)
                    batch_cmds_norm.append(
                        float(self.normalizer.normalize(np.array([phys], dtype=np.float32))[0])
                    )
                bs = 64
                for start in range(0, len(batch_contexts), bs):
                    chunk = torch.stack(batch_contexts[start : start + bs], dim=0).to(device)
                    emb = encoder(chunk).detach().cpu()
                    for i in range(emb.shape[0]):
                        self.samples.append(
                            (emb[i].clone(), batch_cmds_phys[start + i], batch_cmds_norm[start + i])
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        embedding, command_phys, command_norm = self.samples[index]
        return {
            "embedding": embedding,
            "command": torch.tensor([command_phys], dtype=torch.float32),
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
