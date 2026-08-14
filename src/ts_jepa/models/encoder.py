from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block (3×3 conv, BN, ReLU). Block count per stage is IC."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)
        return self.relu(out)


class ContextEncoder(nn.Module):
    """
    Context / online encoder Ψθ (plan §7).

    Paper-specified: deep conv ResNet widths 64, 128, 256, each with BN and ReLU.

    Stem, MaxPool, residual-block count, and spatial-pool head are IMPLEMENTATION
    CHOICES (plan §7: do not label them paper architecture).
    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: list[int] | tuple[int, ...] = (64, 128, 256),
        embedding_dim: int = 256,
        blocks_per_stage: int = 2,
        spatial_pool_hw: tuple[int, int] | list[int] = (4, 8),
        *,
        strict_baseline_dim: bool = True,
    ) -> None:
        super().__init__()
        if tuple(widths) != (64, 128, 256):
            raise ValueError(f"plan §7 widths must be [64, 128, 256], got {list(widths)}")
        if int(embedding_dim) != 256 and strict_baseline_dim:
            raise ValueError(
                f"paper-implied baseline embedding_dim is 256 (plan §6/§9), got {embedding_dim}. "
                "Pass strict_baseline_dim=False for the plan §17 Fig. 7 grid."
            )

        self.in_channels = int(in_channels)
        self.widths = tuple(int(w) for w in widths)
        self.embedding_dim = int(embedding_dim)
        self.blocks_per_stage = int(blocks_per_stage)
        self.spatial_pool_hw = (int(spatial_pool_hw[0]), int(spatial_pool_hw[1]))
        if self.spatial_pool_hw[0] < 1 or self.spatial_pool_hw[1] < 1:
            raise ValueError(f"spatial_pool_hw must be positive, got {self.spatial_pool_hw}")

        # IC stem (not paper-specified): Conv 7×7 s2 → BN → ReLU → MaxPool 3×3 s2
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.widths[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage64 = self._make_stage(self.widths[0], self.widths[0], stride=1)
        self.stage128 = self._make_stage(self.widths[0], self.widths[1], stride=2)
        self.stage256 = self._make_stage(self.widths[1], self.widths[2], stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d(tuple(self.spatial_pool_hw))
        pooled_dim = self.widths[-1] * self.spatial_pool_hw[0] * self.spatial_pool_hw[1]
        self.projection = nn.Linear(pooled_dim, self.embedding_dim)
        # IC: Eq. (13) is cosine (direction-only). Unbounded embeddings + SGD 0.2
        # make 15-step AR prediction oscillate; unit-sphere z is not paper architecture.

    def _make_stage(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(self.blocks_per_stage - 1):
            blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage64(x)
        x = self.stage128(x)
        x = self.stage256(x)
        x = self.global_pool(x).flatten(1)
        return F.normalize(self.projection(x), dim=-1, eps=1e-8)

    def architecture_summary(self) -> dict[str, object]:
        return {
            "in_channels": self.in_channels,
            "widths": list(self.widths),
            "embedding_dim": self.embedding_dim,
            "blocks_per_stage": self.blocks_per_stage,
            "stem": "Conv7x7s2-BN-ReLU-MaxPool3x3s2",
            "stages": ["64", "128@s2", "256@s2"],
            "head": "SpatialPool-Flatten-Linear-L2Norm",
            "spatial_pool_hw": list(self.spatial_pool_hw),
            "l2_normalize": True,
        }


# Alias: target encoder Ψθ̄ uses the same class as the context encoder (plan §8).
TargetEncoder = ContextEncoder
