"""
letterness.py — CPU-only "letterness" scoring for Vesuvius Challenge ink predictions.

Goal
----
Given a 2D image in which ink is bright (an ink-model prediction, or a raw
surface-volume render where crackle is visible), rank sliding windows of a
fixed physical size (default 20 x 20 mm = 4 cm², the First Letters prize area)
by how much *text-like structure* they contain.  No training, no GPU.

Signals (all classical)
-----------------------
ink_fraction     fraction of "ink" pixels after a global adaptive threshold;
                 scored with a plausibility band (too little = empty, too much
                 = smear / wrong depth / sheet merger).
line_periodicity strength of a dominant period in the autocorrelation of the
                 row projection, searched inside the expected line-pitch range
                 and over a small set of rotation angles (lines may be tilted).
stroke_shape     fraction of ink area belonging to connected components whose
                 height is letter-sized and whose shape is stroke-like (not a
                 filled blob).
anisotropy       row-periodicity vs column-periodicity: real text has strong
                 horizontal banding (lines) and much weaker vertical banding.

The four sub-scores are combined with a weighted geometric mean, so a window
must do reasonably well on *all* of them to score high — this is what keeps
noisy "confetti" predictions from ranking above real letters.

Everything is expressed in millimetres and converted to pixels through the
image's pixel size, so the same code works on 2.4 µm, 7.9 µm or 9.4 µm renders.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import disk, white_tophat
from skimage.measure import block_reduce, label, regionprops


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class ScoreConfig:
    pixel_size_um: float = 9.362           # input pixel size (µm)
    working_um: float = 100.0              # analysis resolution (~0.1 mm/px)
    window_mm: float = 20.0                # window side (20 mm -> 4 cm²)
    stride_mm: float = 5.0                 # window stride
    line_pitch_mm: tuple[float, float] = (3.0, 9.0)      # expected line spacing
    letter_height_mm: tuple[float, float] = (1.5, 5.0)   # expected letter height
    angles_deg: tuple[float, ...] = (-10, -6, -3, 0, 3, 6, 10)
    stroke_width_mm: float = 0.6           # scale of the top-hat filter (letter stroke)
    ink_band: tuple[float, float] = (0.04, 0.35)  # plausible ink fraction band
    # Weights measured, not assumed (see docs/feature_selection.md and
    # docs/benchmark_pherc0139.md).
    #
    # `line_periodicity` separates text from papyrus with no overlap on the five
    # real segments available (1.000 vs 0.22-0.28 at the 95th percentile), and
    # removing it collapses the benchmark. It carries the score — alone.
    #
    # `anisotropy` was kept in v0.6 on a 0.89-vs-0.61 gap measured on two text
    # segments. On six annotated segments of PHerc0139 with real predictions its
    # removal changed Average Precision by +0.001 (4 wins, 2 losses, p=0.44): it
    # does nothing. A score that is periodicity times the ink gate is more
    # honest than one that pretends to combine two things.
    #
    # `ink_fraction` is NOT here: on real predictions it takes a single distinct
    # value across 440 windows, so it cannot order anything. It is kept as a
    # multiplicative gate below — its job is to veto empty windows and smears,
    # which it does, not to rank.
    #
    # `stroke_shape` is NOT here either: its sign is not stable. On synthetic
    # data it points the right way (text 1.00 vs noise 0.24) but on real data it
    # inverts (text 0.49 vs papyrus 0.71-0.92), because fibre lattices survive
    # the top-hat as thin fragmented structures while blurred model predictions
    # do not. Three papyrus segments against two text segments is too little to
    # justify flipping its sign, so it is computed, reported, and given no vote.
    weights: dict = field(default_factory=lambda: {
        "line_periodicity": 1.0,
    })
    invert: bool = False                   # set True if ink is dark in the input
    auto_mask: bool = True                 # treat exact-zero pixels as outside the mesh
    min_coverage: float = 0.80             # min valid fraction for a window to be scored
    top_k: int = 10


@dataclass
class WindowScore:
    row: int
    col: int
    # bounding box in ORIGINAL image pixels (y0, x0, y1, x1)
    y0: int
    x0: int
    y1: int
    x1: int
    # same box in millimetres
    y0_mm: float
    x0_mm: float
    y1_mm: float
    x1_mm: float
    score: float
    ink_fraction: float
    line_periodicity: float
    stroke_shape: float
    anisotropy: float
    best_angle_deg: float
    best_pitch_mm: float
    raw_ink_fraction: float
    coverage: float


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------
def load_image(path: str | Path) -> np.ndarray:
    """Load a 2D grayscale image as float32. Handles TIFF (incl. stacks) and PNG/JPG."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(str(path))
    else:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        arr = np.asarray(Image.open(path))
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # (z, y, x) stack -> max projection ; (y, x, c) RGB -> mean over channels
        if arr.shape[-1] in (3, 4) and arr.shape[0] > 4:
            arr = arr[..., :3].mean(axis=-1)
        else:
            arr = arr.max(axis=0)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {arr.shape}")
    return arr.astype(np.float32)


def robust_normalize(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    """Map robust percentiles to [0, 1]. Also absorbs the compressed output range of
    label-smoothed models (e.g. background ~0.25, ink ~0.75)."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def downsample(img: np.ndarray, pixel_size_um: float, working_um: float) -> tuple[np.ndarray, float]:
    """Block-mean downsample to roughly `working_um` per pixel. Returns (image, actual_um)."""
    factor = max(1, int(round(working_um / pixel_size_um)))
    if factor == 1:
        return img, pixel_size_um
    small = block_reduce(img, block_size=(factor, factor), func=np.mean, cval=0.0)
    return small.astype(np.float32), pixel_size_um * factor


# ----------------------------------------------------------------------------
# Sub-scores
# ----------------------------------------------------------------------------
def _autocorr_1d(x: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation (lag 0 == 1) via FFT with zero padding."""
    x = x - x.mean()
    n = len(x)
    if n < 4 or np.allclose(x, 0):
        return np.zeros(n, dtype=np.float32)
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return np.zeros(n, dtype=np.float32)
    return (ac / ac[0]).astype(np.float32)


def _projection_periodicity(patch: np.ndarray, axis: int, pmin_px: int, pmax_px: int,
                            n_strips: int = 3) -> tuple[float, int]:
    """Periodicity of the projection along `axis`, measured robustly on K strips.

    axis=1 -> project each ROW (mean over x): detects horizontal text lines.
    axis=0 -> project each COLUMN (mean over y): detects vertical banding.

    The window is cut into `n_strips` strips perpendicular to the projection
    direction (for rows: left / middle / right).  Real text lines run across
    the whole window, so (a) the strip projections are mutually correlated at
    lag 0 ("coherence") and (b) their autocorrelations add up coherently.
    Smoothed noise produces autocorrelation wiggles that are large in any single
    strip but uncorrelated between strips, so both measures collapse.

    Returns (score, peak_lag_px) with score = prominence(mean autocorr) x coherence.
    """
    if axis == 1:
        strips = np.array_split(patch, n_strips, axis=1)      # left/middle/right
    else:
        strips = np.array_split(patch, n_strips, axis=0)      # top/middle/bottom
    projs = []
    for s in strips:
        p = s.mean(axis=axis).astype(np.float64)
        p = p - ndi.gaussian_filter1d(p, sigma=max(1.0, pmax_px))   # detrend
        sd = p.std()
        projs.append(p / sd if sd > 1e-9 else np.zeros_like(p))
    n = len(projs[0])
    pmax_px = min(pmax_px, n // 2)
    if pmax_px <= pmin_px + 1:
        return 0.0, 0
    # cross-strip coherence at lag 0 (mean pairwise Pearson correlation, clipped at 0)
    cors = []
    for i in range(len(projs)):
        for j in range(i + 1, len(projs)):
            cors.append(float(np.mean(projs[i] * projs[j])))
    coherence = float(np.clip(np.mean(cors), 0.0, 1.0)) if cors else 0.0
    ac = np.mean([_autocorr_1d(p) for p in projs], axis=0)
    seg = ac[pmin_px:pmax_px + 1]
    if seg.size < 3:
        return 0.0, 0
    # local maxima only (avoid picking the decaying flank of lag 0)
    inner = seg[1:-1]
    is_max = (inner >= seg[:-2]) & (inner >= seg[2:])
    if not is_max.any():
        return 0.0, 0
    # Score each candidate by PROMINENCE: peak value minus the deepest trough
    # between lag 1 and the peak.  Periodic text lines give a deep negative
    # trough at half a pitch and a strong positive peak at one pitch
    # (prominence ~0.6-1.0); blobby noise decays monotonically from lag 0 with
    # small wiggles (prominence ~0.05-0.2) even when its raw autocorrelation
    # value at short lags is still large.
    best_prom, best_lag = 0.0, 0
    for k in np.where(is_max)[0]:
        lag = pmin_px + 1 + k
        trough = float(ac[1:lag].min()) if lag > 1 else 0.0
        prom = float(inner[k] - trough)
        if prom > best_prom:
            best_prom, best_lag = prom, int(lag)
    return best_prom * coherence, best_lag


def _rotate_crop(padded: np.ndarray, cy: int, cx: int, half: int, margin: int, angle: float) -> np.ndarray:
    """Rotate a (2*(half+margin))² patch around its centre and crop the central (2*half)² square."""
    y0, y1 = cy - half - margin, cy + half + margin
    x0, x1 = cx - half - margin, cx + half + margin
    big = padded[y0:y1, x0:x1]
    if angle != 0.0:
        big = ndi.rotate(big, angle, reshape=False, order=1, mode="nearest")
    return big[margin:margin + 2 * half, margin:margin + 2 * half]


def _stroke_shape_score(binary: np.ndarray, px_mm: float, letter_h_mm: tuple[float, float]) -> float:
    """Fraction of ink area in letter-sized, stroke-like connected components."""
    if binary.sum() == 0:
        return 0.0
    lab = label(binary, connectivity=2)
    hmin, hmax = letter_h_mm
    total = 0.0
    good = 0.0
    for r in regionprops(lab):
        area = r.area
        total += area
        y0, x0, y1, x1 = r.bbox
        h_mm = (y1 - y0) * px_mm
        w_mm = (x1 - x0) * px_mm
        if not (hmin * 0.6 <= h_mm <= hmax * 1.3):
            continue
        if w_mm > hmax * 3.0:               # long horizontal smear, not a letter
            continue
        if r.area < 4:                       # speckle
            continue
        # Strokes are thin and wiggly: perimeter² / (4π·area) is ~1 for a disc,
        # ~2 for an ellipse-ish blob, and >3 for letter-like stroke clusters.
        complexity = (r.perimeter ** 2) / (4.0 * math.pi * max(area, 1.0))
        if complexity >= 3.0:
            good += area
    return float(good / total) if total > 0 else 0.0


def _band_score(x: float, lo: float, hi: float) -> float:
    """1 inside [lo, hi], decaying outside — asymmetrically.

    Below the band the decay width is `lo` itself, so an empty window scores 0
    rather than being let through: with a symmetric decay of the band's width
    (0.31 here) a coverage of 0.00 still scored 0.87, and the gate waved through
    exactly the windows it exists to stop. Above the band the decay keeps the
    band's width, since a smear is a matter of degree.
    """
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return float(max(0.0, 1.0 - (lo - x) / max(lo, 1e-6)))
    return float(max(0.0, 1.0 - (x - hi) / max(hi - lo, 1e-6)))


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def score_image(img: np.ndarray, cfg: ScoreConfig, mask: np.ndarray | None = None
                ) -> tuple[list[WindowScore], np.ndarray, dict]:
    """Score sliding windows of an ink image.

    Returns (windows_sorted_by_score_desc, score_grid[rows, cols], meta).
    """
    if cfg.invert:
        img = img.max() - img
    img = robust_normalize(img)
    small, px_um = downsample(img, cfg.pixel_size_um, cfg.working_um)
    px_mm = px_um / 1000.0
    factor = px_um / cfg.pixel_size_um

    # Rendered segments are rectangles only partly covered by the mesh; the rest
    # is exact zero. Those pixels must not enter the threshold, the ink fraction
    # or the projections, otherwise empty area reads as "no ink" and drags every
    # statistic with it.
    if mask is None and cfg.auto_mask:
        mask = (img > 0)
    if mask is not None:
        cov, _ = downsample(mask.astype(np.float32), cfg.pixel_size_um, cfg.working_um)
        m_small = cov > 0.5
    else:
        cov = np.ones_like(small, dtype=np.float32)
        m_small = np.ones_like(small, dtype=bool)

    H, W = small.shape
    win = max(8, int(round(cfg.window_mm / px_mm)))
    win = min(win, H, W)
    half = win // 2
    win = 2 * half
    stride = max(1, int(round(cfg.stride_mm / px_mm)))
    margin = int(math.ceil(0.18 * win))

    # LOCAL CONTRAST, not global brightness. A white top-hat keeps only what is
    # brighter than its surroundings at the scale of a letter stroke, and removes
    # the slow brightness variation of the papyrus itself. Without it, a global
    # Otsu threshold on a raw render selects ~50-65% of the pixels — it splits
    # papyrus texture into light and dark halves rather than finding marks — and
    # every downstream statistic becomes meaningless.
    radius = max(2, int(round(cfg.stroke_width_mm * 2.5 / px_mm)))
    feat = white_tophat(small, disk(radius)).astype(np.float32)
    hi = float(np.percentile(feat[m_small], 99.5)) if m_small.any() else 1.0
    feat = np.clip(feat / max(hi, 1e-6), 0.0, 1.0)
    vals = feat[m_small]
    try:
        thr = float(threshold_otsu(vals)) if vals.size > 16 and vals.std() > 1e-4 else 0.5
    except ValueError:
        thr = 0.5
    thr = max(thr, 0.10)

    padded = np.pad(feat, half + margin, mode="reflect")
    padded_cov = np.pad(cov, half + margin, mode="reflect")
    pmin = max(2, int(round(cfg.line_pitch_mm[0] / px_mm)))
    pmax = max(pmin + 2, int(round(cfg.line_pitch_mm[1] / px_mm)))

    ys = list(range(half, H - half + 1, stride)) or [half]
    xs = list(range(half, W - half + 1, stride)) or [half]
    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    results: list[WindowScore] = []
    wsum = sum(v for v in cfg.weights.values() if v > 0) or 1.0

    for i, cy in enumerate(ys):
        for j, cx in enumerate(xs):
            # skip windows mostly outside the valid mask
            cwin = cov[cy - half:cy + half, cx - half:cx + half]
            coverage = float(cwin.mean())
            if coverage < cfg.min_coverage:
                continue
            # ---- ink fraction (unrotated) ----
            patch0 = _rotate_crop(padded, cy + half + margin, cx + half + margin, half, margin, 0.0)
            binary0 = patch0 > thr
            vwin = _rotate_crop(padded_cov, cy + half + margin, cx + half + margin,
                                half, margin, 0.0) > 0.5
            nvalid = int(vwin.sum())
            raw_ink = float((binary0 & vwin).sum() / nvalid) if nvalid else 0.0
            s_ink = _band_score(raw_ink, *cfg.ink_band)
            # ---- line periodicity over angles ----
            best_r, best_lag, best_angle, best_col_r = 0.0, 0, 0.0, 0.0
            for ang in cfg.angles_deg:
                patch = patch0 if ang == 0.0 else _rotate_crop(
                    padded, cy + half + margin, cx + half + margin, half, margin, float(ang))
                r, lag = _projection_periodicity(patch, axis=1, pmin_px=pmin, pmax_px=pmax)
                if r > best_r:
                    best_r, best_lag, best_angle = r, lag, float(ang)
                    best_col_r, _ = _projection_periodicity(patch, axis=0, pmin_px=pmin, pmax_px=pmax)
            s_period = float(np.clip(best_r / 0.45, 0.0, 1.0))
            # ---- anisotropy: rows should beat columns ----
            # +0.05 keeps the ratio from degenerating to 1.0 when both are ~0
            s_aniso = float(best_r / (best_r + best_col_r + 0.05))
            # ---- stroke shape on the best-angle binary patch ----
            patch_b = patch0 if best_angle == 0.0 else _rotate_crop(
                padded, cy + half + margin, cx + half + margin, half, margin, best_angle)
            s_stroke = _stroke_shape_score(patch_b > thr, px_mm, cfg.letter_height_mm)
            # ---- combine (weighted geometric mean with a small floor) ----
            eps = 0.02
            all_subs = {"ink_fraction": s_ink, "line_periodicity": s_period,
                        "stroke_shape": s_stroke, "anisotropy": s_aniso}
            subs = {k: v for k, v in all_subs.items() if cfg.weights.get(k, 0.0) > 0}
            if subs:
                logsum = sum(cfg.weights[k] * math.log(max(v, eps)) for k, v in subs.items())
                score = float(math.exp(logsum / wsum))
            else:
                score = 0.0
            # ink_fraction acts as a GATE, not as a ranking term: a window whose
            # ink coverage is implausible (empty, or a smear from a wrong depth or
            # a merged sheet) is suppressed whatever its geometry looks like.
            if "ink_fraction" not in subs:
                score *= s_ink
            grid[i, j] = score
            # window box in original pixels
            oy0, ox0 = int((cy - half) * factor), int((cx - half) * factor)
            oy1, ox1 = int((cy + half) * factor), int((cx + half) * factor)
            results.append(WindowScore(
                row=i, col=j, y0=oy0, x0=ox0, y1=oy1, x1=ox1,
                y0_mm=oy0 * cfg.pixel_size_um / 1000, x0_mm=ox0 * cfg.pixel_size_um / 1000,
                y1_mm=oy1 * cfg.pixel_size_um / 1000, x1_mm=ox1 * cfg.pixel_size_um / 1000,
                score=score, ink_fraction=s_ink, line_periodicity=s_period,
                stroke_shape=s_stroke, anisotropy=s_aniso, best_angle_deg=best_angle,
                best_pitch_mm=best_lag * px_mm, raw_ink_fraction=raw_ink,
                coverage=coverage))

    results.sort(key=lambda w: w.score, reverse=True)
    meta = {
        "input_shape": [int(v) for v in img.shape],
        "working_pixel_um": px_um,
        "downsample_factor": factor,
        "window_px_working": win,
        "stride_px_working": stride,
        "ink_threshold": thr,
        "n_windows_scored": len(results),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
    }
    return results, grid, meta


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------
def save_heatmap(grid: np.ndarray, small_shape: tuple[int, int], out_png: str | Path,
                 stride: int, half: int) -> None:
    """Upsample the window grid to the working-resolution canvas and save as PNG."""
    from PIL import Image
    H, W = small_shape
    canvas = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            cy, cx = half + i * stride, half + j * stride
            canvas[cy - half:cy + half, cx - half:cx + half] += grid[i, j]
            count[cy - half:cy + half, cx - half:cx + half] += 1
    canvas = np.where(count > 0, canvas / np.maximum(count, 1), 0)
    im = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    im.save(out_png)


def save_overlay(img_small: np.ndarray, windows: Iterable[WindowScore], factor: float,
                 out_png: str | Path, top_k: int = 10) -> None:
    """Draw the top-K window boxes on the (downsampled) input image."""
    from PIL import Image, ImageDraw
    base = (np.clip(img_small, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(base).convert("RGB")
    dr = ImageDraw.Draw(im)
    for k, w in enumerate(list(windows)[:top_k]):
        box = [w.x0 / factor, w.y0 / factor, w.x1 / factor, w.y1 / factor]
        color = (255, 40, 40) if k == 0 else (255, 180, 0)
        dr.rectangle(box, outline=color, width=2)
        dr.text((box[0] + 3, box[1] + 3), f"#{k + 1} {w.score:.2f}", fill=color)
    im.save(out_png)


def write_json(windows: list[WindowScore], meta: dict, out_json: str | Path, top_k: int) -> None:
    payload = {"meta": meta, "top_windows": [asdict(w) for w in windows[:top_k]]}
    Path(out_json).write_text(json.dumps(payload, indent=2))
