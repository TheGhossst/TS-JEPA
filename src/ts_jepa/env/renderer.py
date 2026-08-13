from __future__ import annotations

import numpy as np

from ts_jepa.env.cartpole_ode import CartPoleParams

# Paper-silent renderer (IMPLEMENTATION CHOICE): float coordinates + coverage
# anti-aliasing so 1 ms / sub-pixel cart motion changes the RGB array.


def _over(img: np.ndarray, y0: int, y1: int, x0: int, x1: int, coverage: np.ndarray, color: np.ndarray) -> None:
    if coverage.size == 0:
        return
    region = img[y0 : y1 + 1, x0 : x1 + 1]
    alpha = coverage[..., None]
    region *= 1.0 - alpha
    region += alpha * color.reshape(1, 1, 3)


def _point_segment_distance(
    px: np.ndarray,
    py: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> np.ndarray:
    vx = x1 - x0
    vy = y1 - y0
    length_sq = vx * vx + vy * vy
    if length_sq < 1e-18:
        return np.hypot(px - x0, py - y0)
    t = np.clip(((px - x0) * vx + (py - y0) * vy) / length_sq, 0.0, 1.0)
    return np.hypot(px - (x0 + t * vx), py - (y0 + t * vy))


# Paper Fig. 5(c): same physical state, different cart colors. Palettes are IC.
CART_APPEARANCE_PALETTE: tuple[dict[str, tuple[float, float, float]], ...] = (
    {"cart": (40.0, 90.0, 180.0), "cart_edge": (20.0, 40.0, 90.0), "pole": (200.0, 60.0, 40.0), "tip": (220.0, 80.0, 50.0)},
    {"cart": (180.0, 50.0, 40.0), "cart_edge": (90.0, 20.0, 20.0), "pole": (40.0, 90.0, 180.0), "tip": (50.0, 110.0, 200.0)},
    {"cart": (40.0, 140.0, 70.0), "cart_edge": (15.0, 70.0, 30.0), "pole": (180.0, 120.0, 30.0), "tip": (200.0, 140.0, 40.0)},
    {"cart": (140.0, 50.0, 160.0), "cart_edge": (70.0, 20.0, 80.0), "pole": (40.0, 160.0, 150.0), "tip": (50.0, 180.0, 170.0)},
    {"cart": (200.0, 140.0, 30.0), "cart_edge": (110.0, 70.0, 10.0), "pole": (50.0, 50.0, 160.0), "tip": (70.0, 70.0, 190.0)},
    {"cart": (30.0, 120.0, 150.0), "cart_edge": (10.0, 60.0, 80.0), "pole": (190.0, 70.0, 90.0), "tip": (210.0, 90.0, 110.0)},
    {"cart": (90.0, 90.0, 90.0), "cart_edge": (40.0, 40.0, 40.0), "pole": (200.0, 80.0, 40.0), "tip": (220.0, 100.0, 50.0)},
    {"cart": (20.0, 60.0, 120.0), "cart_edge": (10.0, 30.0, 70.0), "pole": (160.0, 40.0, 40.0), "tip": (190.0, 60.0, 60.0)},
)


def sample_appearance(rng: np.random.Generator) -> dict[str, np.ndarray]:
    palette = CART_APPEARANCE_PALETTE[int(rng.integers(0, len(CART_APPEARANCE_PALETTE)))]
    return {key: np.asarray(value, dtype=np.float64) for key, value in palette.items()}


class CartPoleRenderer:
    """
    RGB renderer for inverted cart-pole (native resolution is IMPLEMENTATION CHOICE).

    Uses floating-point world-to-pixel mapping and coverage anti-aliasing so that
    sub-pixel state changes (1 ms at τ_o) produce a different uint8 RGB frame.
    Encoder input is 64×128 after the paper's Gaussian resize, not this native size.
    """

    _BG = np.array([245.0, 245.0, 245.0])
    _RAIL = np.array([80.0, 80.0, 80.0])

    def __init__(
        self,
        height: int = 128,
        width: int = 256,
        params: CartPoleParams | None = None,
        appearance: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.params = params or CartPoleParams()
        self.set_appearance(appearance)

    def set_appearance(self, appearance: dict[str, np.ndarray] | None = None) -> None:
        # Default is palette[0] (blue cart / red pole) so centroid checks stay deterministic.
        palette = appearance or {
            key: np.asarray(value, dtype=np.float64) for key, value in CART_APPEARANCE_PALETTE[0].items()
        }
        self._CART = np.asarray(palette["cart"], dtype=np.float64)
        self._CART_EDGE = np.asarray(palette["cart_edge"], dtype=np.float64)
        self._POLE = np.asarray(palette["pole"], dtype=np.float64)
        self._TIP = np.asarray(palette["tip"], dtype=np.float64)

    def _clip_bbox(self, x_min: float, y_min: float, x_max: float, y_max: float) -> tuple[int, int, int, int] | None:
        x0 = max(0, int(np.floor(x_min)))
        y0 = max(0, int(np.floor(y_min)))
        x1 = min(self.width - 1, int(np.ceil(x_max)))
        y1 = min(self.height - 1, int(np.ceil(y_max)))
        if x0 > x1 or y0 > y1:
            return None
        return y0, y1, x0, x1

    def _stroke_line(
        self,
        img: np.ndarray,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: np.ndarray,
        width: float,
    ) -> None:
        pad = width * 0.5 + 1.5
        bbox = self._clip_bbox(min(x0, x1) - pad, min(y0, y1) - pad, max(x0, x1) + pad, max(y0, y1) + pad)
        if bbox is None:
            return
        top, bottom, left, right = bbox
        ys, xs = np.mgrid[top : bottom + 1, left : right + 1]
        dist = _point_segment_distance(xs + 0.5, ys + 0.5, x0, y0, x1, y1)
        coverage = np.clip(0.5 + width * 0.5 - dist, 0.0, 1.0)
        _over(img, top, bottom, left, right, coverage, color)

    def _fill_rect(
        self,
        img: np.ndarray,
        left: float,
        top: float,
        right: float,
        bottom: float,
        color: np.ndarray,
    ) -> None:
        bbox = self._clip_bbox(left - 1.0, top - 1.0, right + 1.0, bottom + 1.0)
        if bbox is None:
            return
        y0, y1, x0, x1 = bbox
        ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        cov_x = np.clip(np.minimum(right, xs + 1.0) - np.maximum(left, xs.astype(np.float64)), 0.0, 1.0)
        cov_y = np.clip(np.minimum(bottom, ys + 1.0) - np.maximum(top, ys.astype(np.float64)), 0.0, 1.0)
        _over(img, y0, y1, x0, x1, cov_x * cov_y, color)

    def _fill_circle(self, img: np.ndarray, cx: float, cy: float, radius: float, color: np.ndarray) -> None:
        pad = radius + 1.5
        bbox = self._clip_bbox(cx - pad, cy - pad, cx + pad, cy + pad)
        if bbox is None:
            return
        y0, y1, x0, x1 = bbox
        ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        dist = np.hypot((xs + 0.5) - cx, (ys + 0.5) - cy)
        coverage = np.clip(0.5 + radius - dist, 0.0, 1.0)
        _over(img, y0, y1, x0, x1, coverage, color)

    def render(self, state: np.ndarray) -> np.ndarray:
        x, _, theta, _ = np.asarray(state, dtype=np.float64)[:4]
        img = np.empty((self.height, self.width, 3), dtype=np.float64)
        img[...] = self._BG

        world_half = float(self.params.track_limit)
        px_per_m = (self.width * 0.8) / (2.0 * world_half)
        origin_x = self.width / 2.0
        rail_y = self.height * 0.72

        cart_x = origin_x + float(x) * px_per_m
        cart_w = max(12.0, 0.35 * px_per_m)
        cart_h = max(8.0, 0.2 * px_per_m)

        self._stroke_line(img, self.width * 0.05, rail_y, self.width * 0.95, rail_y, self._RAIL, 2.0)

        left = cart_x - cart_w / 2.0
        top = rail_y - cart_h
        right = cart_x + cart_w / 2.0
        bottom = rail_y
        self._fill_rect(img, left, top, right, bottom, self._CART)
        self._stroke_line(img, left, top, right, top, self._CART_EDGE, 1.0)
        self._stroke_line(img, right, top, right, bottom, self._CART_EDGE, 1.0)
        self._stroke_line(img, right, bottom, left, bottom, self._CART_EDGE, 1.0)
        self._stroke_line(img, left, bottom, left, top, self._CART_EDGE, 1.0)

        pole_len_px = float(self.params.pole_length) * 2.0 * px_per_m
        tip_x = cart_x + pole_len_px * np.sin(float(theta))
        tip_y = top - pole_len_px * np.cos(float(theta))
        self._stroke_line(img, cart_x, top, tip_x, tip_y, self._POLE, 3.0)
        self._fill_circle(img, tip_x, tip_y, 4.0, self._TIP)

        return np.clip(np.rint(img), 0.0, 255.0).astype(np.uint8)
