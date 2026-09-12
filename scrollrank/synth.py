"""
synth.py — synthetic "ink prediction" images for testing and demos.

Nothing here imitates real Herculaneum data closely; the point is only to
have controllable images where we *know* whether text-like structure is
present, so the scorer can be regression-tested offline.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _glyph(rng: np.random.Generator, h: int, w: int, stroke: int) -> np.ndarray:
    """A random letter-like glyph: 2-4 strokes (bars / diagonals / arcs) in an h x w box."""
    g = np.zeros((h, w), dtype=np.float32)
    n_strokes = rng.integers(2, 5)
    for _ in range(n_strokes):
        kind = rng.integers(0, 4)
        if kind == 0:                                   # vertical bar
            x = rng.integers(0, max(1, w - stroke))
            g[:, x:x + stroke] = 1
        elif kind == 1:                                 # horizontal bar
            y = rng.integers(0, max(1, h - stroke))
            g[y:y + stroke, :] = 1
        elif kind == 2:                                 # diagonal
            for t in np.linspace(0, 1, 4 * max(h, w)):
                y, x = int(t * (h - 1)), int(t * (w - 1))
                if rng.random() < 0.5:
                    x = w - 1 - x
                g[max(0, y - stroke // 2):y + stroke // 2 + 1, max(0, x - stroke // 2):x + stroke // 2 + 1] = 1
        else:                                           # arc
            cy, cx, r = h / 2, w / 2, min(h, w) / 2 - 1
            for a in np.linspace(0, np.pi * rng.uniform(0.8, 1.6), 200):
                y, x = int(cy + r * np.sin(a)), int(cx + r * np.cos(a))
                if 0 <= y < h and 0 <= x < w:
                    g[max(0, y - stroke // 2):y + stroke // 2 + 1, max(0, x - stroke // 2):x + stroke // 2 + 1] = 1
    return g


def make_text_image(shape_mm: tuple[float, float] = (60.0, 60.0), pixel_um: float = 100.0,
                    line_pitch_mm: float = 5.0, letter_h_mm: float = 3.0, letter_w_mm: float = 2.4,
                    column_w_mm: float = 45.0, margin_mm: float = 6.0, tilt_deg: float = 0.0,
                    noise: float = 0.25, blur_px: float = 0.7, seed: int = 0) -> np.ndarray:
    """Rows of random glyphs in one column, plus noise and blur. Values ~[0,1], ink bright."""
    rng = np.random.default_rng(seed)
    px_mm = pixel_um / 1000.0
    H, W = int(shape_mm[0] / px_mm), int(shape_mm[1] / px_mm)
    img = np.zeros((H, W), dtype=np.float32)
    lh, lw = int(letter_h_mm / px_mm), int(letter_w_mm / px_mm)
    pitch = int(line_pitch_mm / px_mm)
    stroke = max(1, int(0.35 / px_mm))
    x_start = int(margin_mm / px_mm)
    x_end = min(W, x_start + int(column_w_mm / px_mm))
    y = int(margin_mm / px_mm)
    while y + lh < H - int(margin_mm / px_mm):
        x = x_start
        while x + lw < x_end:
            if rng.random() < 0.85:                     # occasional word gaps
                img[y:y + lh, x:x + lw] = np.maximum(img[y:y + lh, x:x + lw], _glyph(rng, lh, lw, stroke))
            x += lw + int(0.4 / px_mm)
        y += pitch
    if tilt_deg:
        img = ndi.rotate(img, tilt_deg, reshape=False, order=1, mode="constant")
    img = ndi.gaussian_filter(img, blur_px)
    img = 0.25 + 0.5 * img                              # mimic label-smoothed model output
    img = img + noise * rng.standard_normal(img.shape).astype(np.float32) * 0.25
    return np.clip(img, 0, 1).astype(np.float32)


def make_noise_image(shape_mm: tuple[float, float] = (60.0, 60.0), pixel_um: float = 100.0,
                     blob_sigma_px: float = 2.0, ink_fraction: float = 0.12, seed: int = 1) -> np.ndarray:
    """Blobby random field with a given ink fraction: what a model spits out on non-text papyrus."""
    rng = np.random.default_rng(seed)
    px_mm = pixel_um / 1000.0
    H, W = int(shape_mm[0] / px_mm), int(shape_mm[1] / px_mm)
    field = ndi.gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), blob_sigma_px)
    thr = np.quantile(field, 1 - ink_fraction)
    img = (field > thr).astype(np.float32)
    img = ndi.gaussian_filter(img, 0.7)
    img = 0.25 + 0.5 * img + 0.06 * rng.standard_normal(img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def make_stripes_image(shape_mm: tuple[float, float] = (60.0, 60.0), pixel_um: float = 100.0,
                       pitch_mm: float = 5.0, seed: int = 2) -> np.ndarray:
    """Pure horizontal stripes: periodic like text lines but with no letter-shaped strokes.
    Useful to check that stroke_shape / ink_fraction keep this from scoring as text."""
    rng = np.random.default_rng(seed)
    px_mm = pixel_um / 1000.0
    H, W = int(shape_mm[0] / px_mm), int(shape_mm[1] / px_mm)
    img = np.zeros((H, W), dtype=np.float32)
    pitch = int(pitch_mm / px_mm)
    band = max(1, int(2.5 / px_mm))
    for y in range(0, H, pitch):
        img[y:y + band, :] = 1
    img = 0.25 + 0.5 * ndi.gaussian_filter(img, 0.7)
    img = img + 0.05 * rng.standard_normal(img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)
