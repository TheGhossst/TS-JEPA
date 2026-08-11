"""Data augmentation transforms for TS-JEPA training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class TSJEPATransform(nn.Module):
    """
    Augmentations for TS-JEPA image inputs.

    Expects a batch of images with shape (B, C, H, W) in float32 on [0, 1].
    Color jittering and color dropping are applied only in training mode.
    Normalization is always applied.
    """

    def __init__(
        self,
        brightness: float = 0.05,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05,
        grayscale_prob: float = 0.05,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.grayscale_prob = grayscale_prob

        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor)

    def _color_jitter(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        device = images.device

        brightness_factors = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
            1.0 - self.brightness,
            1.0 + self.brightness,
        )
        images = images * brightness_factors

        channel_means = images.mean(dim=(-2, -1), keepdim=True)
        contrast_factors = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
            1.0 - self.contrast,
            1.0 + self.contrast,
        )
        images = (images - channel_means) * contrast_factors + channel_means

        grayscale = (
            0.299 * images[:, 0:1]
            + 0.587 * images[:, 1:2]
            + 0.114 * images[:, 2:3]
        )
        saturation_factors = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
            1.0 - self.saturation,
            1.0 + self.saturation,
        )
        images = grayscale + saturation_factors * (images - grayscale)

        hue_factors = torch.empty(batch_size, device=device).uniform_(-self.hue, self.hue)
        images = torch.stack(
            [TF.adjust_hue(images[i], float(hue_factors[i])) for i in range(batch_size)],
            dim=0,
        )

        return images.clamp(0.0, 1.0)

    def _color_drop(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        drop_mask = torch.rand(batch_size, device=images.device) < self.grayscale_prob
        if not drop_mask.any():
            return images

        grayscale = (
            0.299 * images[:, 0:1]
            + 0.587 * images[:, 1:2]
            + 0.114 * images[:, 2:3]
        ).expand_as(images)
        return torch.where(drop_mask.view(batch_size, 1, 1, 1), grayscale, images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected input shape (B, 3, H, W), got {tuple(images.shape)}")

        x = images
        if self.training:
            x = self._color_jitter(x)
            x = self._color_drop(x)

        return (x - self.mean) / self.std
