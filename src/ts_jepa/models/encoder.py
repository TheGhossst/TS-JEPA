from __future__ import annotations

import torch
import torch.nn as nn


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
    Context / online encoder Ψθ (plan §6.1).

    Architecture:
      Input [B, C_in, 64, 128]  (C_in = 3 or 6 for κ = 1 / 2)
        → Conv2D C_in→64, k=7, s=2 → BN → ReLU → MaxPool
        → residual stage 64
        → residual stage 128 (stride 2)
        → residual stage 256 (stride 2)
        → global average pool
        → linear → 256-D embedding
    """

    def __init__(
        self,
        in_channels: int = 6,
        widths: list[int] | tuple[int, ...] = (64, 128, 256),
        embedding_dim: int = 256,
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        if tuple(widths) != (64, 128, 256):
            raise ValueError(f"plan §6.1 widths must be [64, 128, 256], got {list(widths)}")
        if int(embedding_dim) != 256:
            raise ValueError(f"plan §6.1 embedding_dim must be 256, got {embedding_dim}")

        self.in_channels = int(in_channels)
        self.widths = tuple(int(w) for w in widths)
        self.embedding_dim = int(embedding_dim)
        self.blocks_per_stage = int(blocks_per_stage)

        # Plan §6.1 stem: Conv2D → BN → ReLU → MaxPool
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.widths[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage64 = self._make_stage(self.widths[0], self.widths[0], stride=1)
        self.stage128 = self._make_stage(self.widths[0], self.widths[1], stride=2)
        self.stage256 = self._make_stage(self.widths[1], self.widths[2], stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(self.widths[-1], self.embedding_dim)

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
        return self.projection(x)

    def architecture_summary(self) -> dict[str, object]:
        return {
            "in_channels": self.in_channels,
            "widths": list(self.widths),
            "embedding_dim": self.embedding_dim,
            "blocks_per_stage": self.blocks_per_stage,
            "stem": "Conv7x7s2-BN-ReLU-MaxPool3x3s2",
            "stages": ["64", "128@s2", "256@s2"],
            "head": "GAP-Linear",
        }


# Resolve forward reference for plan §7 alias.
TargetEncoder = ContextEncoder
