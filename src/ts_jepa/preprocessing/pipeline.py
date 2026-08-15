from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ts_jepa.preprocessing.plan import EVAL_PIPELINE_STAGES, TRAINING_PIPELINE_STAGES
from ts_jepa.plan.enforce import jepa_uses_kappa_stack


class PreprocessPipeline:
    """
    Paper-faithful RGB preprocessing and κ-frame context construction.

    Training (plan §5 numbered list):
      raw RGB → color jitter (random order) → color drop → ImageNet normalize
      → resize to 64×128 with a 5×5 Gaussian kernel (σ ~ U[0.1, 0.2])

    Evaluation (same two geometric/color stages, no stochastic aug):
      raw RGB → ImageNet normalize → 5×5 Gaussian resize to 64×128
      (σ = range midpoint). Same order as training so ImageNet stats are
      applied in the same domain.
    """

    training_stages = TRAINING_PIPELINE_STAGES
    eval_stages = EVAL_PIPELINE_STAGES

    def __init__(self, config: dict[str, Any], training: bool = True) -> None:
        self.config = config
        self.training = training
        inp = config["input"]
        self.resize_hw = tuple(inp["resize"])  # (H, W) = (64, 128) encoder input
        self.jitter = inp["color_jitter"]
        self.color_drop_p = float(inp["color_drop_probability"])
        self.mean = torch.tensor(inp["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(inp["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
        self.gaussian_kernel = tuple(inp["gaussian_kernel"])
        self.sigma_range = tuple(inp["gaussian_sigma_range"])
        self.kappa = int(inp["kappa"])
        self.construction = inp.get("multi_frame_tensor_construction", "channel_concat")

    def _rng(self, rng: np.random.Generator | None = None) -> np.random.Generator:
        if rng is not None:
            return rng
        return np.random.default_rng(int(np.random.randint(0, 2**31 - 1)))

    def _decode_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Raw uint8 HWC RGB → float CHW in [0, 1]."""
        return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0

    def _gaussian_1d_weights(self, centers: torch.Tensor, size: int, sigma: float, n_taps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-output-pixel 1D Gaussian taps and clamped integer sample indices."""
        offs = torch.arange(n_taps, device=centers.device, dtype=centers.dtype) - (n_taps - 1) / 2.0
        loc = centers[:, None] + offs[None, :]
        dist = loc - centers[:, None]
        w = torch.exp(-(dist**2) / (2.0 * sigma**2 + 1e-12))
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        idx = loc.round().long().clamp(0, size - 1)
        return w, idx

    def _augment(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        """Paper: color jitter in random order, then probabilistic grayscale/color drop."""
        img = self._color_jitter(img, rng)
        return self._color_drop(img, rng)

    def _color_jitter(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        ops = ["brightness", "contrast", "saturation", "hue"]
        rng.shuffle(ops)
        for op in ops:
            if op == "brightness":
                b = 1.0 + float(rng.uniform(-self.jitter["brightness"], self.jitter["brightness"]))
                img = (img * b).clamp(0.0, 1.0)
            elif op == "contrast":
                c = 1.0 + float(rng.uniform(-self.jitter["contrast"], self.jitter["contrast"]))
                mean = img.mean(dim=(1, 2), keepdim=True)
                img = ((img - mean) * c + mean).clamp(0.0, 1.0)
            elif op == "saturation":
                s = 1.0 + float(rng.uniform(-self.jitter["saturation"], self.jitter["saturation"]))
                gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0)
                img = (img * s + gray * (1.0 - s)).clamp(0.0, 1.0)
            else:
                hue_delta = float(rng.uniform(-self.jitter["hue"], self.jitter["hue"]))
                if abs(hue_delta) > 1e-8:
                    img = self._apply_hue(img, hue_delta)
        return img

    def _apply_hue(self, img: torch.Tensor, hue_delta: float) -> torch.Tensor:
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
        return torch.stack([r, g, bch], dim=0).clamp(0.0, 1.0)

    def _color_drop(self, img: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if rng.random() >= self.color_drop_p:
            return img
        luma = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        return torch.stack([luma, luma, luma], dim=0)

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        """Plan §5.1 / §5.2 ImageNet normalization."""
        return (img - self.mean) / self.std

    def _gaussian_resize(self, img: torch.Tensor, rng: np.random.Generator | None = None) -> torch.Tensor:
        """
        Paper Section IV.A: resize to 64×128 using a 5×5 Gaussian kernel
        (σ ~ U[0.1, 0.2] in training; midpoint σ when rng is None).

        This is Gaussian-kernel resampling in input coordinates, not blur-then-bilinear.
        """
        if rng is None:
            sigma = 0.5 * (float(self.sigma_range[0]) + float(self.sigma_range[1]))
        else:
            sigma = float(rng.uniform(self.sigma_range[0], self.sigma_range[1]))
        out_h, out_w = self.resize_hw
        _c, in_h, in_w = img.shape
        k_h, k_w = self.gaussian_kernel
        device, dtype = img.device, img.dtype
        cy = (torch.arange(out_h, device=device, dtype=dtype) + 0.5) * (in_h / float(out_h)) - 0.5
        cx = (torch.arange(out_w, device=device, dtype=dtype) + 0.5) * (in_w / float(out_w)) - 0.5
        wy, iy = self._gaussian_1d_weights(cy, in_h, sigma, k_h)
        wx, ix = self._gaussian_1d_weights(cx, in_w, sigma, k_w)
        gathered = img[:, iy[:, :, None, None], ix[None, None, :, :]]
        return torch.einsum("ciajb,ia,jb->cij", gathered, wy, wx)

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
            img = self._gaussian_resize(img, gen)
        else:
            img = self._normalize(img)
            img = self._gaussian_resize(img, rng=None)
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

    def assemble_jepa_frame(self, processed: dict[int, torch.Tensor], time_index: int) -> torch.Tensor:
        """Algorithm 1: one RGB frame → [3, 64, 128] for Ψ(x_{i,k})."""
        return processed[int(time_index)]

    def assemble_context(self, processed: dict[int, torch.Tensor], time_index: int, kappa: int | None = None) -> torch.Tensor:
        """κ-frame channel concat for supervised / AE baselines (Section IV.D.3)."""
        k = self.kappa if kappa is None else int(kappa)
        start = max(0, time_index - k + 1)
        selected = [processed[t] for t in range(start, time_index + 1)]
        while len(selected) < k:
            selected.insert(0, selected[0])
        if self.construction != "channel_concat":
            raise ValueError(f"Unsupported multi_frame_tensor_construction: {self.construction}")
        return torch.cat(selected, dim=0)

    def assemble_jepa_input(self, processed: dict[int, torch.Tensor], time_index: int) -> torch.Tensor:
        """Encoder tensor at time k: one RGB frame (paper) or κ-stack (working)."""
        if jepa_uses_kappa_stack(self.config):
            return self.assemble_context(processed, time_index)
        return self.assemble_jepa_frame(processed, time_index)

    def make_jepa_input(self, frames: np.ndarray, time_index: int) -> torch.Tensor:
        """Encode-ready Ψ input at time k (single frame or κ-stack)."""
        if jepa_uses_kappa_stack(self.config):
            return self.make_context_tensor(frames, time_index)
        return self.make_jepa_frame(frames, time_index)

    def make_jepa_frame(self, frames: np.ndarray, time_index: int) -> torch.Tensor:
        """Encode-ready single frame x_k (Algorithm 1)."""
        t = int(time_index)
        return self.process_frame(frames[t], stochastic=self.training)

    def make_context_tensor(self, frames: np.ndarray, time_index: int, kappa: int | None = None) -> torch.Tensor:
        """κ consecutive frames packed for supervised/AE (IC: channel_concat)."""
        k = self.kappa if kappa is None else int(kappa)
        start = max(0, time_index - k + 1)
        processed = self.process_frames_cached(frames, start, time_index, stochastic=self.training)
        return self.assemble_context(processed, time_index, kappa=k)
