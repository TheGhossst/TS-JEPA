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

    Spatial: coarse grid pooling (4×8) preserves cart position without full-map flatten.

    Temporal (κ≥2): channel-concat input [frame_{t-1}, frame_t] is expanded to
    [frame_t, frame_{t-1}, frame_t - frame_{t-1}] (9 channels for κ=2) before the stem.
    """

    SPATIAL_POOL_HW = (4, 8)

    def __init__(
        self,
        in_channels: int = 6,
        widths: list[int] | tuple[int, ...] = (64, 128, 256),
        embedding_dim: int = 256,
        blocks_per_stage: int = 2,
        input_hw: tuple[int, int] = (64, 128),
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        widths_tuple = tuple(widths)
        bottleneck_dim = int(widths_tuple[-1])

        self.raw_in_channels = int(in_channels)
        # κ=2 channel-concat (6 ch): add explicit temporal difference channels.
        self.use_frame_diff = self.raw_in_channels >= 6 and self.raw_in_channels % 3 == 0
        self.stem_in_channels = 9 if self.use_frame_diff else self.raw_in_channels

        self.stem = nn.Sequential(
            nn.Conv2d(self.stem_in_channels, widths_tuple[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths_tuple[0]),
            nn.ReLU(inplace=True),
        )

        stage_blocks: list[nn.Sequential] = []
        in_c = widths_tuple[0]
        for stage_idx, out_c in enumerate(widths_tuple):
            stride = 1 if stage_idx == 0 else 2
            blocks: list[ResidualBlock] = [ResidualBlock(in_c, out_c, stride=stride)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(ResidualBlock(out_c, out_c, stride=1))
            stage_blocks.append(nn.Sequential(*blocks))
            in_c = out_c
        self.stages = nn.ModuleList(stage_blocks)

        self.spatial_pool = nn.AdaptiveAvgPool2d(self.SPATIAL_POOL_HW)
        self.flatten = nn.Flatten()

        h, w = int(input_hw[0]), int(input_hw[1])
        with torch.no_grad():
            dummy = torch.zeros(1, self.raw_in_channels, h, w)
            feat_dim = self._pool_and_flatten(self._forward_conv(self._prepare_stem_input(dummy))).shape[1]

        self.fc_bottleneck = nn.Linear(feat_dim, bottleneck_dim)
        self.fc_relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(bottleneck_dim, self.embedding_dim)

    def _prepare_stem_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_frame_diff:
            return x
        # Channel-concat order from assemble_context: [..., frame_{t-1}, frame_t].
        frame_t_minus_1 = x[:, -6:-3]
        frame_t = x[:, -3:]
        diff = frame_t - frame_t_minus_1
        return torch.cat([frame_t, frame_t_minus_1, diff], dim=1)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def _pool_and_flatten(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.spatial_pool(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_stem_input(x)
        x = self._forward_conv(x)
        x = self._pool_and_flatten(x)
        x = self.fc_relu(self.fc_bottleneck(x))
        return self.fc(x)
