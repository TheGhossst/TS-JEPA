"""Actor that conditions on (z_k, z_{k-1}) without changing JEPA."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from ts_jepa.inference.infer import FrozenRuntimeController, RuntimeCommandStats
from ts_jepa.models.actor import SemanticActor

FeatureKind = Literal["z", "z_pair", "z_delta"]


def pair_feature(z: np.ndarray, prev_z: np.ndarray | None, kind: FeatureKind) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    if kind == "z":
        return z
    prev = z if prev_z is None else np.asarray(prev_z, dtype=np.float32).reshape(-1)
    if kind == "z_pair":
        return np.concatenate([z, prev], axis=0)
    if kind == "z_delta":
        return np.concatenate([z, z - prev], axis=0)
    raise ValueError(f"unknown feature kind {kind!r}")


def feature_dim(z_dim: int, kind: FeatureKind) -> int:
    return int(z_dim) if kind == "z" else 2 * int(z_dim)


class PairZRuntimeController:
    """RGB → frozen Ψθ → [z_k, z_{k-1}] → Cε."""

    miss_behavior = "hold_last_feature"

    def __init__(
        self,
        encoder: FrozenRuntimeController,
        actor: nn.Module,
        *,
        feature: FeatureKind = "z_pair",
        full_information: bool = True,
    ) -> None:
        self.encoder = encoder
        self.actor = actor.to(encoder.device).eval()
        for p in self.actor.parameters():
            p.requires_grad_(False)
        self.feature = feature
        self.full_information = bool(full_information)
        self.stats: RuntimeCommandStats = encoder.stats
        self.prev_z: np.ndarray | None = None
        self.last_feat: np.ndarray | None = None

    def reset_episode(self) -> None:
        self.encoder.reset_episode()
        self.prev_z = None
        self.last_feat = None

    def observe_frame(self, frame: np.ndarray) -> None:
        self.encoder.observe_frame(frame)

    def _encode_z(self, frame: np.ndarray) -> np.ndarray:
        self.encoder.observe_frame(frame)
        context = self.encoder._context_from_buffer()
        z = self.encoder.jepa.encode_context(context)
        return z.detach().cpu().reshape(-1).numpy().astype(np.float32)

    @torch.no_grad()
    def step(
        self,
        frame,
        packet_received: bool,
        plant_state: np.ndarray | None = None,
    ) -> float:
        del plant_state
        if frame is None:
            raise ValueError("PairZRuntimeController requires an RGB frame")
        use_encode = self.full_information or packet_received or self.last_feat is None
        if use_encode:
            z = self._encode_z(frame)
            feat = pair_feature(z, self.prev_z, self.feature)
            self.prev_z = z
            self.last_feat = feat
        else:
            self.encoder.observe_frame(frame)
            feat = self.last_feat
            assert feat is not None
        x = torch.from_numpy(feat).unsqueeze(0).to(self.encoder.device)
        u_actor = self.actor(x)
        force, _ = self.stats.actor_output_to_force_and_norm(u_actor)
        return force


def make_pair_actor(config: dict[str, Any], kind: FeatureKind) -> SemanticActor:
    z_dim = int(config["ts_jepa"]["encoder"]["embedding_dim"])
    arch = config["semantic_actor"]["architecture"]
    return SemanticActor(
        embedding_dim=feature_dim(z_dim, kind),
        hidden_dims=tuple(arch["hidden_dims"]),
        dropout=float(arch.get("dropout", 0.2)),
    )
