from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ts_jepa.env.cartpole_ode import CartPoleParams


class CartPoleRenderer:
    """RGB renderer for inverted cart-pole (raw resolution is IMPLEMENTATION CHOICE)."""

    def __init__(
        self,
        height: int = 64,
        width: int = 128,
        params: CartPoleParams | None = None,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.params = params or CartPoleParams()

    def render(self, state: np.ndarray) -> np.ndarray:
        x, _, theta, _ = state
        image = Image.new("RGB", (self.width, self.height), color=(245, 245, 245))
        draw = ImageDraw.Draw(image)

        # World-to-pixel mapping.
        world_half = self.params.track_limit
        px_per_m = (self.width * 0.8) / (2.0 * world_half)
        origin_x = self.width // 2
        rail_y = int(self.height * 0.72)

        cart_x = int(origin_x + x * px_per_m)
        cart_w = max(12, int(0.35 * px_per_m))
        cart_h = max(8, int(0.2 * px_per_m))

        # Track
        draw.line([(int(self.width * 0.05), rail_y), (int(self.width * 0.95), rail_y)], fill=(80, 80, 80), width=2)

        # Cart
        left = cart_x - cart_w // 2
        top = rail_y - cart_h
        draw.rectangle([left, top, left + cart_w, rail_y], fill=(40, 90, 180), outline=(20, 40, 90))

        # Pole (from cart center upward when theta=0)
        pole_len_px = self.params.pole_length * 2.0 * px_per_m
        pivot = (cart_x, top)
        tip_x = pivot[0] + pole_len_px * np.sin(theta)
        tip_y = pivot[1] - pole_len_px * np.cos(theta)
        draw.line([pivot, (tip_x, tip_y)], fill=(200, 60, 40), width=3)
        r = 4
        draw.ellipse([tip_x - r, tip_y - r, tip_x + r, tip_y + r], fill=(220, 80, 50))

        return np.asarray(image, dtype=np.uint8)
