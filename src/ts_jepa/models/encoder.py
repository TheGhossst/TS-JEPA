from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Basic residual block (kernel/stride/padding details are IMPLEMENTATION CHOICE)."""

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
    Context / online encoder Ψθ.

    Paper-specified: residual CNN widths 64→128→256, BatchNorm, ReLU.
    Low-level block/kernel/stride/pooling choices are IMPLEMENTATION CHOICE.
    """

    def __init__(
        self,
        in_channels: int = 6,
        widths: list[int] | tuple[int, ...] = (64, 128, 256),
        embedding_dim: int = 256,
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages = []
        in_c = widths[0]
        for stage_idx, out_c in enumerate(widths):
            stride = 1 if stage_idx == 0 else 2
            blocks = [ResidualBlock(in_c, out_c, stride=stride)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(ResidualBlock(out_c, out_c, stride=1))
            stages.append(nn.Sequential(*blocks))
            in_c = out_c
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[-1], self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
