from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ts_jepa.preprocessing.plan import EVAL_PIPELINE_STAGES, TRAINING_PIPELINE_STAGES


class PreprocessPipeline:
    """
    Paper-faithful RGB preprocessing and κ-frame context construction.

    Plan §5.1 training:
      raw RGB → augmentation → normalization → formatting (Gaussian blur + resize)

    Plan §5.2 evaluation:
      raw RGB → resize → normalization (no stochastic augmentations, no blur)
    """

    training_stages = TRAINING_PIPELINE_STAGES
    eval_stages = EVAL_PIPELINE_STAGES

    def __init__(self, config: dict[str, Any], training: bool = True) -> None:
        self.config = config
        self.training = training
        inp = config["input"]
        self.resize_hw = tuple(inp["resize"])  # (H, W) = (64, 128)
        self.jitter = inp["color_jitter"]
        self.color_drop_p = float(inp["color_drop_probability"])
        self.mean = torch.tensor(inp["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(inp["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
        self.gaussian_kernel = tuple(inp["gaussian_kernel"])
        self.sigma_range = tuple(inp["gaussian_sigma_range"])
        self.kappa = int(inp["kappa"])
        self.construction = inp.get("multi_frame_tensor_construction", "channel_concat")
        self._blur_kernels: dict[float, torch.Tensor] = {}

    def _rng(self, rng: np.random.Generator | None = None) -> np.random.Generator:
        if rng is not None:
            return rng
        return np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))

    def _decode_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Raw uint8 HWC RGB → float CHW in [0, 1]."""
        return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0

    def _gaussian_kernel_2d(self, sigma: float) -> torch.Tensor:
        key = round(float(sigma), 5)
        cached = self._blur_kernels.get(key)
        if cached is not None:
            return cached
        k_h, k_w = self.gaussian_kernel
        ys = torch.arange(k_h, dtype=torch.float32) - (k_h - 1) / 2.0
        xs = torch.arange(k_w, dtype=torch.float32) - (k_w - 1) / 2.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2 + 1e-12))
        kernel = kernel / kernel.sum()
        weight = kernel.view(1, 1, k_h, k_w).repeat(3, 1, 1, 1)
        self._blur_kernels[key] = weight
        return weight

    def _augment(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Plan §5.1 augmentation: color jitter + probabilistic grayscale/color drop."""
        img = self._color_jitter(img, rng)
        return self._color_drop(img, rng)

    def _color_jitter(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        b = 1.0 + float(rng.uniform(-self.jitter["brightness"], self.jitter["brightness"]))
        c = 1.0 + float(rng.uniform(-self.jitter["contrast"], self.jitter["contrast"]))
        s = 1.0 + float(rng.uniform(-self.jitter["saturation"], self.jitter["saturation"]))
        img = (img * b).clamp(0.0, 1.0)
        mean = img.mean(dim=(1, 2), keepdim=True)
        img = ((img - mean) * c + mean).clamp(0.0, 1.0)
        gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0)
        img = (img * s + gray * (1.0 - s)).clamp(0.0, 1.0)
        hue_delta = float(rng.uniform(-self.jitter["hue"], self.jitter["hue"]))
        if abs(hue_delta) > 1e-8:
            y = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            i = 0.596 * img[0] - 0.275 * img[1] - 0.321 * img[2]
            q = 0.212 * img[0] - 0.523 * img[1] + 0.311 * img[2]
            angle = hue_delta * 2.0 * np.pi
            cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
            i2 = i * cos_a - q * sin_a
            q2 = i * sin_a + q * cos_a
            r = y + 0.956 * i2 + 0.621 * q2
            g = y - 0.272 * i2 - 0.647 * q2
            bch = y - 1.106 * i2 + 1.703 * q2
            img = torch.stack([r, g, bch], dim=0).clamp(0.0, 1.0)
        return img

    def _color_drop(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if rng.random() >= self.color_drop_p:
            return img
        luma = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        return torch.stack([luma, luma, luma], dim=0)

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """Plan §5.1 / §5.2 ImageNet normalization."""
        return (img - self.mean) / self.std

    def _resize(self, img: torch.Tensor) -> torch.Tensor:
        h, w = self.resize_hw
        if img.shape[-2:] == (h, w):
            return img
        x = F.interpolate(img.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
        return x.squeeze(0)

    def _gaussian_blur(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Plan §5.1 formatting: 5×5 Gaussian blur with σ ~ Uniform[0.1, 0.2]."""
        sigma = float(rng.uniform(self.sigma_range[0], self.sigma_range[1]))
        weight = self._gaussian_kernel_2d(sigma)
        pad_h = self.gaussian_kernel[0] // 2
        pad_w = self.gaussian_kernel[1] // 2
        x = F.pad(img.unsqueeze(0), (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        x = F.conv2d(x, weight, groups=3)
        return x.squeeze(0)

    def _format_training(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Plan §5.1 formatting: Gaussian blur then resize to 64×128."""
        img = self._gaussian_blur(img, rng)
        return self._resize(img)

    def process_frame(
        self,
        frame: np.ndarray,
        stochastic: bool | None = None,
        rng: np.random.Generator | None = None,
    ) -> torch.Tensor:
        use_aug = self.training if stochastic is None else stochastic
        img = self._decode_frame(frame)
        if use_aug:
            gen = self._rng(rng)
            img = self._augment(img, gen)
            img = self._normalize(img)
            img = self._format_training(img, gen)
        else:
            img = self._resize(img)
            img = self._normalize(img)
        return img

    def process_frames_cached(
        self,
        frames: np.ndarray,
        start: int,
        end: int,
        stochastic: bool | None = None,
    ) -> dict[int, torch.Tensor]:
        """Process inclusive frame indices once and return a cache."""
        return {t: self.process_frame(frames[t], stochastic=stochastic) for t in range(start, end + 1)}

    def assemble_context(self, processed: dict[int, torch.Tensor], time_index: int, kappa: int | None = None) -> torch.Tensor:
        k = self.kappa if kappa is None else int(kappa)
        start = max(0, time_index - k + 1)
        selected = [processed[t] for t in range(start, time_index + 1)]
        while len(selected) < k:
            selected.insert(0, selected[0])
        if self.construction != "channel_concat":
            raise ValueError(f"Unsupported multi_frame_tensor_construction: {self.construction}")
        return torch.cat(selected, dim=0)

    def make_context_tensor(self, frames: np.ndarray, time_index: int, kappa: int | None = None) -> torch.Tensor:
        """κ consecutive frames channel-concatenated (κ=2 → [6, H, W])."""
        k = self.kappa if kappa is None else int(kappa)
        start = max(0, time_index - k + 1)
        processed = self.process_frames_cached(frames, start, time_index, stochastic=self.training)
        return self.assemble_context(processed, time_index, kappa=k)
