"""
benchmark.py — quantitative evaluation of the ranking, not of a single score.

ScrollRank is a triage tool: the question that matters is not "how high does
the best window score" but "if a human can only look at K windows, how many of
the windows that actually contain text do we hand them". This module answers
that, using a segment for which the Vesuvius Challenge team has published
hand-made ink annotations.

Inputs
------
prediction : a 2D ink-probability image (e.g. an ink_9um output)
label      : the hand annotation of the SAME segment, any resolution
             (it is resampled onto the prediction's grid)

Outputs
-------
Recall@K (at the level of distinct text REGIONS, after non-maximum suppression),
precision@K and Average Precision for
  * the full ScrollRank score,
  * every leave-one-out ablation of the four sub-scores,
  * two baselines: raw ink fraction alone, and random ordering,
plus a precision-recall curve.

Definition of a positive
------------------------
A window is positive if the fraction of its area covered by annotated ink is at
least `min_label_frac` (default 0.5 %). This is deliberately generous: the
annotations are sparse, so a window with two annotated letters counts.

Why grid regions, and why NMS
-----------------------------
Windows overlap heavily (10 mm window, 2 mm stride), so one patch of text
produces hundreds of positive windows and plain window-level recall@10 is
mechanically tiny however good the ranking is. And a human would never be shown
ten near-identical windows of the same spot. So before counting, the ranking is
passed through greedy non-maximum suppression (windows overlapping an already
kept one above `iou_thresh` are dropped), and recall counts how many annotated
REGIONS the top K surviving windows reach. That is the operational question:
"with a budget of K looks, how many of the text areas do we find".

Regions are cells of a fixed grid (default: half a window on a side), not
connected components of the annotation. Components were tried first and failed:
on a densely written segment the annotations merge into a handful of blobs — on
PHerc0139 w035 the whole segment collapsed to 7 regions, so even random ordering
reached all of them within 25 windows and every ranking scored 1.000. A fixed
grid keeps the number of targets proportional to the area examined, so the
metric stays informative on dense and sparse text alike, at the cost of being a
coarse localisation criterion rather than an object-level one.

An important caveat, stated here because it bounds what the numbers mean: the
labels mark only strokes the annotators were sure of. A window with no
annotation may still contain text. Recall is therefore measured fairly, while
precision is a LOWER BOUND — some "false positives" are unannotated text.
Comparisons between rankings on the same data remain valid, because every
ranking is penalised by the same missing annotations.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .letterness import ScoreConfig, WindowScore, load_image, score_image

SUBSCORES = ("ink_fraction", "line_periodicity", "stroke_shape", "anisotropy")


# ----------------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------------
def resample_label(label: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resample a binary label onto the prediction's pixel grid (area-preserving)."""
    lab = (label > 0).astype(np.float32)
    zy = target_shape[0] / lab.shape[0]
    zx = target_shape[1] / lab.shape[1]
    out = ndi.zoom(lab, (zy, zx), order=1)
    # zoom can be off by a pixel; pad or crop to be exact
    fixed = np.zeros(target_shape, dtype=np.float32)
    h = min(target_shape[0], out.shape[0])
    w = min(target_shape[1], out.shape[1])
    fixed[:h, :w] = out[:h, :w]
    return fixed > 0.5


def estimate_shift(pred: np.ndarray, label: np.ndarray, max_shift_px: int = 40
                   ) -> tuple[int, int, float]:
    """Integer (dy, dx) that best aligns label to prediction, by FFT cross-correlation.

    Both inputs are binarised first: we correlate "where the model says ink" with
    "where the annotator says ink". Returns (dy, dx, peak_ratio); peak_ratio is
    the correlation peak divided by the mean, so values near 1 mean no real peak
    and the shift should not be trusted.
    """
    a = pred.astype(np.float32)
    a = (a > np.percentile(a, 85)).astype(np.float32)
    b = label.astype(np.float32)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0, 0, 1.0
    a = a - a.mean()
    b = b - b.mean()
    F = np.fft.rfft2(a) * np.conj(np.fft.rfft2(b))
    cc = np.fft.irfft2(F, s=a.shape)
    cc = np.fft.fftshift(cc)
    cy, cx = cc.shape[0] // 2, cc.shape[1] // 2
    win = cc[cy - max_shift_px:cy + max_shift_px + 1, cx - max_shift_px:cx + max_shift_px + 1]
    if win.size == 0:
        return 0, 0, 1.0
    iy, ix = np.unravel_index(np.argmax(win), win.shape)
    peak = float(win[iy, ix])
    ratio = peak / (float(np.abs(cc).mean()) + 1e-9)
    return int(iy - max_shift_px), int(ix - max_shift_px), ratio


def apply_shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return ndi.shift(mask.astype(np.float32), (dy, dx), order=0, mode="constant", cval=0) > 0.5


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def average_precision(labels: np.ndarray) -> float:
    """AP of a ranked binary list (labels[i] = 1 if the i-th ranked item is positive)."""
    if labels.sum() == 0:
        return float("nan")
    tp = np.cumsum(labels)
    prec = tp / np.arange(1, len(labels) + 1)
    return float((prec * labels).sum() / labels.sum())


def pr_curve(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp = np.cumsum(labels)
    prec = tp / np.arange(1, len(labels) + 1)
    rec = tp / max(labels.sum(), 1)
    return rec, prec


def nms(order: np.ndarray, boxes: np.ndarray, iou_thresh: float = 0.2,
        limit: int = 400) -> list[int]:
    """Greedy non-maximum suppression over the ranked window boxes."""
    kept: list[int] = []
    for idx in order:
        y0, x0, y1, x1 = boxes[idx]
        ok = True
        for j in kept:
            Y0, X0, Y1, X1 = boxes[j]
            iy = max(0, min(y1, Y1) - max(y0, Y0))
            ix = max(0, min(x1, X1) - max(x0, X0))
            inter = iy * ix
            if inter == 0:
                continue
            union = (y1 - y0) * (x1 - x0) + (Y1 - Y0) * (X1 - X0) - inter
            if union > 0 and inter / union > iou_thresh:
                ok = False
                break
        if ok:
            kept.append(int(idx))
            if len(kept) >= limit:
                break
    return kept


def rank_metrics(order: np.ndarray, positive: np.ndarray, boxes: np.ndarray,
                 win_regions: list[set], n_regions: int, iou_thresh: float = 0.2,
                 ks=(5, 10, 25, 50, 100)) -> dict:
    """Window-level AP plus region-level recall/precision after NMS."""
    lab = positive[order].astype(np.int32)
    n, P = len(lab), int(positive.sum())
    out = {"n_windows": n, "n_positive_windows": P, "n_regions": n_regions,
           "average_precision": average_precision(lab)}
    kept = nms(order, boxes, iou_thresh=iou_thresh, limit=max(ks) if ks else 100)
    found: set = set()
    hits = 0
    for rank, idx in enumerate(kept, start=1):
        found |= win_regions[idx]
        if positive[idx]:
            hits += 1
        if rank in ks:
            out[f"recall@{rank}"] = len(found) / n_regions if n_regions else float("nan")
            out[f"precision@{rank}"] = hits / rank
    for k in ks:                                   # ranking shorter than k
        out.setdefault(f"recall@{k}", len(found) / n_regions if n_regions else float("nan"))
        out.setdefault(f"precision@{k}", hits / max(len(kept), 1))
    out["n_kept_after_nms"] = len(kept)
    return out


# ----------------------------------------------------------------------------
# Re-ranking with different weight sets
# ----------------------------------------------------------------------------
def combined(w: WindowScore, weights: dict, eps: float = 0.02) -> float:
    tot = sum(weights.values())
    if tot <= 0:
        return 0.0
    s = sum(wt * math.log(max(getattr(w, name), eps)) for name, wt in weights.items())
    return float(math.exp(s / tot))


def rank_by(windows: list[WindowScore], weights: dict) -> np.ndarray:
    vals = np.array([combined(w, weights) for w in windows])
    return np.argsort(-vals, kind="stable")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run(prediction_path: str | Path, label_path: str | Path, cfg: ScoreConfig,
        out_dir: str | Path, min_label_frac: float = 0.005, align: bool = True,
        iou_thresh: float = 0.2, region_mm: float | None = None,
        cover_frac: float = 0.5, seed: int = 0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if region_mm is None:
        region_mm = cfg.window_mm / 2.0
    pred = load_image(prediction_path)
    label_raw = load_image(label_path)
    label = resample_label(label_raw, pred.shape)

    info_align = {"applied": False, "dy": 0, "dx": 0, "peak_ratio": None}
    if align:
        dy, dx, ratio = estimate_shift(pred, label)
        info_align.update(dy=dy, dx=dx, peak_ratio=round(ratio, 2))
        # only trust a clear peak and a small shift
        if ratio > 3.0 and (dy or dx):
            label = apply_shift(label, dy, dx)
            info_align["applied"] = True

    windows, _grid, meta = score_image(pred, cfg)
    if not windows:
        raise RuntimeError("no windows scored — check pixel size and window size")

    # label coverage per window, in ORIGINAL prediction pixels
    integral = np.cumsum(np.cumsum(label.astype(np.float64), axis=0), axis=1)

    def box_sum(y0, x0, y1, x1):
        y0 = max(0, min(y0, label.shape[0] - 1)); y1 = max(1, min(y1, label.shape[0]))
        x0 = max(0, min(x0, label.shape[1] - 1)); x1 = max(1, min(x1, label.shape[1]))
        tot = integral[y1 - 1, x1 - 1]
        if y0 > 0: tot -= integral[y0 - 1, x1 - 1]
        if x0 > 0: tot -= integral[y1 - 1, x0 - 1]
        if y0 > 0 and x0 > 0: tot += integral[y0 - 1, x0 - 1]
        return float(tot), (y1 - y0) * (x1 - x0)

    label_frac = np.empty(len(windows), dtype=np.float32)
    for i, w in enumerate(windows):
        s, area = box_sum(w.y0, w.x0, w.y1, w.x1)
        label_frac[i] = s / max(area, 1)
    positive = label_frac >= min_label_frac

    # Distinct "places to look" are defined on a FIXED GRID of window-sized
    # cells, not by connected components of the annotation.
    #
    # Connected components fail on dense text: a page of writing merges into a
    # handful of blobs (7 on PHerc0139 w035), and with so few targets even a
    # random ordering finds them all within 25 windows, so recall@K saturates
    # at 1.000 for every ranking and measures nothing. A grid keeps the number
    # of targets proportional to the annotated area, so "how much of the text
    # do K looks reach" stays a real question on dense and sparse text alike.
    cell_px = max(8, int(round(region_mm * 1000 / cfg.pixel_size_um)))
    gy = (np.arange(label.shape[0]) // cell_px)[:, None]
    gx = (np.arange(label.shape[1]) // cell_px)[None, :]
    ny, nx = int(gy.max()) + 1, int(gx.max()) + 1
    cell_id = (gy * nx + gx + 1).astype(np.int64)
    # a cell counts as a target only if it holds enough annotated ink
    counts = np.bincount(cell_id.ravel(), weights=label.ravel().astype(np.float64))
    areas = np.bincount(cell_id.ravel())
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(areas > 0, counts / np.maximum(areas, 1), 0.0)
    is_target = frac >= min_label_frac
    region_id = np.where(is_target[cell_id], cell_id, 0)
    target_ids = sorted({int(v) for v in np.unique(region_id) if v > 0})
    lut = np.zeros(int(cell_id.max()) + 1, dtype=np.int64)
    for new_id, old_id in enumerate(target_ids, start=1):
        lut[old_id] = new_id
    region_id = lut[region_id]
    n_regions = len(target_ids)
    region_area = np.bincount(region_id.ravel(), minlength=n_regions + 1)

    boxes = np.array([[w.y0, w.x0, w.y1, w.x1] for w in windows], dtype=np.int64)
    # A window "reaches" a region only if it covers at least `cover_frac` of that
    # cell: merely clipping a corner is not a look at that place.
    win_regions: list[set] = []
    for w in windows:
        sub = region_id[max(0, w.y0):w.y1, max(0, w.x0):w.x1]
        ids, cnt = np.unique(sub, return_counts=True)
        reached = set()
        for v, c in zip(ids, cnt):
            if v > 0 and region_area[v] and c / region_area[v] >= cover_frac:
                reached.add(int(v))
        win_regions.append(reached)

    rankings: dict[str, np.ndarray] = {}
    full = dict(cfg.weights)
    rankings["ScrollRank (full)"] = rank_by(windows, full)
    voting = [k for k, v in full.items() if v > 0]
    if len(voting) > 1:                      # leave-one-out only makes sense with >1 voter
        for name in voting:
            w2 = {k: v for k, v in full.items() if k != name}
            rankings[f"ablation: senza {name}"] = rank_by(windows, w2)
    for name in SUBSCORES:
        rankings[f"solo {name}"] = rank_by(windows, {name: 1.0})
    raw = np.array([w.raw_ink_fraction for w in windows])
    rankings["baseline: frazione grezza di inchiostro"] = np.argsort(-raw, kind="stable")
    rng = np.random.default_rng(seed)
    rnd = np.arange(len(windows)); rng.shuffle(rnd)
    rankings["baseline: casuale"] = rnd

    results = {name: rank_metrics(order, positive, boxes, win_regions, n_regions,
                                  iou_thresh=iou_thresh)
               for name, order in rankings.items()}

    # Diagnostics. A sub-score that is saturated on nearly every window carries no
    # ordering information: its "ranking" is then just the scan order, and any AP
    # it reports is an artefact. Reporting saturation and the rank correlation
    # against the full score makes such rows readable instead of misleading.
    diagnostics = {"saturation": {}, "rank_corr_vs_full": {}}
    full_order = rankings["ScrollRank (full)"]
    full_rank = np.empty(len(windows), dtype=np.float64)
    full_rank[full_order] = np.arange(len(windows))
    for name in SUBSCORES:
        v = np.array([getattr(w, name) for w in windows])
        diagnostics["saturation"][name] = {
            "frac_at_1.00": round(float((v >= 0.999).mean()), 3),
            "frac_at_0.02_floor": round(float((v <= 0.021).mean()), 3),
            "distinct_values": int(len(np.unique(np.round(v, 4)))),
        }
    for name, order in rankings.items():
        r = np.empty(len(windows), dtype=np.float64)
        r[order] = np.arange(len(windows))
        a, b = r - r.mean(), full_rank - full_rank.mean()
        den = float(np.sqrt((a * a).sum() * (b * b).sum()))
        diagnostics["rank_corr_vs_full"][name] = round(float((a * b).sum() / den), 3) if den else 0.0

    # precision-recall curve for the main rankings
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for name in ["ScrollRank (full)", "solo line_periodicity", "solo ink_fraction",
                     "baseline: frazione grezza di inchiostro", "baseline: casuale"]:
            if name not in rankings:
                continue
            rec, prec = pr_curve(positive[rankings[name]].astype(np.int32))
            ax.plot(rec, prec, lw=1.8 if "full" in name else 1.1,
                    ls="--" if "baseline" in name else "-", label=name)
        base = positive.mean()
        ax.axhline(base, color="#888", ls=":", lw=1)
        ax.text(0.02, base + 0.01, f"prevalenza {base:.2f}", fontsize=8, color="#666")
        ax.set_xlabel("recall"); ax.set_ylabel("precisione"); ax.set_ylim(0, 1.02)
        ax.set_title("Ranking delle finestre — curva precisione/recall")
        ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(out_dir / "pr_curve.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    report = {
        "prediction": str(prediction_path),
        "label": str(label_path),
        "alignment": info_align,
        "positives": {
            "min_label_frac": min_label_frac,
            "n_positive_windows": int(positive.sum()),
            "n_windows": int(len(windows)),
            "prevalence": round(float(positive.mean()), 4),
            "n_regions": int(n_regions),
            "region_mm": region_mm,
            "region_grid": [int(ny), int(nx)],
            "cover_frac": cover_frac,
            "iou_thresh": iou_thresh,
        },
        "scoring_meta": meta,
        "results": results,
        "diagnostics": diagnostics,
        "caveat": ("Labels are sparse and mark only strokes the annotators were sure of, "
                   "so precision is a lower bound. Comparisons between rankings on the "
                   "same data are unaffected."),
    }
    (out_dir / "benchmark.json").write_text(json.dumps(report, indent=2))
    return report


def format_table(report: dict) -> str:
    ks = [k for k in ("recall@10", "recall@25", "recall@50", "recall@100")
          if any(k in v for v in report["results"].values())]
    head = f"{'ranking':<44}{'AP':>7}" + "".join(f"{k:>12}" for k in ks)
    lines = [head, "-" * len(head)]
    for name, m in report["results"].items():
        row = f"{name:<44}{m['average_precision']:>7.3f}"
        for k in ks:
            row += f"{m.get(k, float('nan')):>12.3f}"
        lines.append(row)
    p = report["positives"]
    lines.append("-" * len(head))
    lines.append(f"{p['n_regions']} regioni annotate distinte; {p['n_positive_windows']} "
                 f"finestre positive su {p['n_windows']} (prevalenza {p['prevalence']:.3f})")
    lines.append(f"regioni = celle di griglia da {p['region_mm']:.0f} mm "
                 f"({p['region_grid'][0]}x{p['region_grid'][1]}) con almeno "
                 f"{p['min_label_frac']*100:.1f}% di area annotata")
    lines.append("recall@K = frazione di regioni raggiunte dalle prime K finestre "
                 f"non sovrapposte (IoU < {p['iou_thresh']})")
    d = report.get("diagnostics")
    if d:
        lines.append("")
        lines.append("saturazione dei sub-punteggi (una feature satura non ordina nulla):")
        for k, v in d["saturation"].items():
            lines.append(f"  {k:<20} a 1.00: {v['frac_at_1.00']:.2f}   "
                         f"al pavimento: {v['frac_at_0.02_floor']:.2f}   "
                         f"valori distinti: {v['distinct_values']}")
    return "\n".join(lines)
