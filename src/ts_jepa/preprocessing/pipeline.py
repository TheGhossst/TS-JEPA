from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PreprocessPipeline:
    """Paper-faithful RGB preprocessing and κ-frame context construction."""

    def __init__(self, config: dict[str, Any], training: bool = True) -> None:
        self.config = config
        self.training = training
        inp = config["input"]
        self.resize_hw = tuple(inp["resize"])  # (H, W) = (64, 128)
        self.jitter = inp["color_jitter"]
        self.color_drop_p = float(inp["color_drop_probability"])
        self.mean = np.asarray(inp["normalization"]["mean"], dtype=np.float32)
        self.std = np.asarray(inp["normalization"]["std"], dtype=np.float32)
        self.gaussian_kernel = tuple(inp["gaussian_kernel"])
        self.sigma_range = tuple(inp["gaussian_sigma_range"])
        self.kappa = int(inp["kappa"])
        self.construction = inp.get("multi_frame_tensor_construction", "channel_concat")

    def _pil_from_rgb(self, frame: np.ndarray) -> Image.Image:
        return Image.fromarray(frame.astype(np.uint8), mode="RGB")

    def _color_jitter(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        b = 1.0 + float(rng.uniform(-self.jitter["brightness"], self.jitter["brightness"]))
        c = 1.0 + float(rng.uniform(-self.jitter["contrast"], self.jitter["contrast"]))
        s = 1.0 + float(rng.uniform(-self.jitter["saturation"], self.jitter["saturation"]))
        image = ImageEnhance.Brightness(image).enhance(b)
        image = ImageEnhance.Contrast(image).enhance(c)
        image = ImageEnhance.Color(image).enhance(s)
        # Approximate hue jitter via HSV channel shift.
        hue_delta = float(rng.uniform(-self.jitter["hue"], self.jitter["hue"]))
        if abs(hue_delta) > 1e-8:
            hsv = np.asarray(image.convert("HSV"), dtype=np.int16)
            hsv[..., 0] = (hsv[..., 0].astype(np.float32) + hue_delta * 255.0) % 256
            image = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
        return image

    def _color_drop(self, image: Image.Image, rng: np.random.Generator) -> Image.Image:
        if rng.random() >= self.color_drop_p:
            return image
        arr = np.asarray(image, dtype=np.float32)
        # Rec. 601 luma
        luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        gray = np.stack([luma, luma, luma], axis=-1).astype(np.uint8)
        return Image.fromarray(gray, mode="RGB")

    def _gaussian_resize(self, image: Image.Image, rng: np.random.Generator | None) -> Image.Image:
        if rng is None:
            sigma = float(np.mean(self.sigma_range))
        else:
            sigma = float(rng.uniform(self.sigma_range[0], self.sigma_range[1]))
        # Approximate 5x5 Gaussian kernel blur before resize (PAPER-SPECIFIED).
        radius = max(self.gaussian_kernel) / 2.0
        blurred = image.filter(ImageFilter.GaussianBlur(radius=max(0.1, sigma * radius)))
        h, w = self.resize_hw
        return blurred.resize((w, h), resample=Image.BILINEAR)

    def process_frame(self, frame: np.ndarray, stochastic: bool | None = None) -> torch.Tensor:
        use_aug = self.training if stochastic is None else stochastic
        rng = np.random.default_rng()
        image = self._pil_from_rgb(frame)
        if use_aug:
            image = self._color_jitter(image, rng)
            image = self._color_drop(image, rng)
            image = self._gaussian_resize(image, rng)
        else:
            image = self._gaussian_resize(image, None)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor

    def make_context_tensor(self, frames: np.ndarray, time_index: int, kappa: int | None = None) -> torch.Tensor:
        """
        κ consecutive frames. Channel-concat to [6, H, W] for κ=2 is IMPLEMENTATION CHOICE.
        """
        k = self.kappa if kappa is None else int(kappa)
        start = max(0, time_index - k + 1)
        selected = []
        for t in range(start, time_index + 1):
            selected.append(self.process_frame(frames[t], stochastic=self.training))
        while len(selected) < k:
            selected.insert(0, selected[0].clone())
        if self.construction != "channel_concat":
            raise ValueError(f"Unsupported multi_frame_tensor_construction: {self.construction}")
        return torch.cat(selected, dim=0)
