"""
concordance.py — do independent runs agree on WHERE the signal is?

A high score on one prediction is not evidence. Models trained from different
seeds make different mistakes but should make the same discoveries, so the
question that separates signal from noise is whether they point at the same
place.

This was not an abstract concern. On segment 025628 of PHerc1447 the mean of
four runs produced a window at 0.636 — above the operating threshold — and it
dissolved under this test: seed 42 put its best window at (2508, 0) and seed 43
at (912, 1596), regions that do not overlap at all. The candidate was an artefact
of averaging two uncorrelated peaks. On PHerc0139 w035, where text is present,
the same two seeds produce score fields that correlate at 0.78.

Three numbers are reported, all over the window grid rather than over pixels,
because the question is about places to look:

spearman        rank correlation of the score fields, computed over windows
                where at least one run scored above zero. Robust to the two runs
                using different score ranges. This is the headline number.

top_decile_iou  intersection over union of the top 10% of windows of each run.
                Answers "do they agree on the interesting places" rather than
                "do they agree everywhere".

top1_iou        overlap of the single best window. Informative when high, but
                unreliable when low: on w035 it is only 0.10 because the best
                windows are adjacent rather than identical, while the fields as
                a whole clearly agree.

Reference values, same settings (10 mm windows, 2 mm stride):

    PHerc0139 w035, seeds 42/43 — text present     spearman 0.78
    a score field against a shuffled copy          spearman 0.09

There is no universal threshold. Compare a segment's concordance against a
segment where the answer is known, and treat a candidate that only one run sees
as unproven rather than as a finding.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from .letterness import ScoreConfig, load_image, score_image


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    m = (a > 0) | (b > 0)
    a, b = a[m], b[m]
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den else float("nan")


def _top_iou(a: np.ndarray, b: np.ndarray, q: float = 90.0) -> float:
    ta, tb = np.percentile(a, q), np.percentile(b, q)
    A, B = a >= ta, b >= tb
    union = int((A | B).sum())
    return float((A & B).sum() / union) if union else float("nan")


def _box_iou(a, b) -> float:
    iy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
    ix = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
    inter = iy * ix
    if inter == 0:
        return 0.0
    ua = (a.y1 - a.y0) * (a.x1 - a.x0)
    ub = (b.y1 - b.y0) * (b.x1 - b.x0)
    return float(inter / (ua + ub - inter))


def run(paths: list[str | Path], cfg: ScoreConfig, out_dir: str | Path | None = None,
        top_k: int = 5) -> dict:
    if len(paths) < 2:
        raise ValueError("servono almeno due predizioni dello stesso segmento")

    runs = {}
    shape = None
    for p in paths:
        img = load_image(p)
        if shape is None:
            shape = img.shape
        elif img.shape != shape:
            raise ValueError(f"{p} ha forma {img.shape}, attesa {shape}")
        ws, grid, _ = score_image(img, cfg)
        runs[str(p)] = {"windows": ws, "grid": grid}

    names = list(runs)
    pairs = []
    for x, y in itertools.combinations(names, 2):
        gx, gy = runs[x]["grid"], runs[y]["grid"]
        wx, wy = runs[x]["windows"], runs[y]["windows"]
        pairs.append({
            "a": Path(x).name, "b": Path(y).name,
            "spearman": round(_spearman(gx, gy), 3),
            "top_decile_iou": round(_top_iou(gx, gy), 3),
            "top1_iou": round(_box_iou(wx[0], wy[0]), 3) if wx and wy else float("nan"),
            "top1_a": [wx[0].x0, wx[0].y0, wx[0].x1, wx[0].y1] if wx else None,
            "top1_b": [wy[0].x0, wy[0].y0, wy[0].x1, wy[0].y1] if wy else None,
            "best_score_a": round(wx[0].score, 3) if wx else None,
            "best_score_b": round(wy[0].score, 3) if wy else None,
        })

    sp = [p["spearman"] for p in pairs if p["spearman"] == p["spearman"]]
    report = {
        "n_runs": len(names),
        "runs": [Path(n).name for n in names],
        "pairs": pairs,
        "spearman_mean": round(float(np.mean(sp)), 3) if sp else float("nan"),
        "spearman_min": round(float(np.min(sp)), 3) if sp else float("nan"),
        "reference": {"w035_text_present": 0.78, "shuffled_control": 0.09},
        "note": ("Un candidato visto da una sola corsa non e' una scoperta. "
                 "Confrontare con un segmento in cui la risposta e' nota."),
    }
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "concordance.json").write_text(json.dumps(report, indent=2))
    return report


def format_report(report: dict) -> str:
    lines = [f"{report['n_runs']} corse: " + ", ".join(report["runs"]), ""]
    head = f"{'coppia':<46}{'spearman':>10}{'top10%':>9}{'top1':>7}"
    lines += [head, "-" * len(head)]
    for p in report["pairs"]:
        lines.append(f"{p['a'][:22] + ' / ' + p['b'][:22]:<46}"
                     f"{p['spearman']:>10.3f}{p['top_decile_iou']:>9.3f}{p['top1_iou']:>7.3f}")
    lines.append("-" * len(head))
    lines.append(f"spearman medio {report['spearman_mean']:.3f}, "
                 f"minimo {report['spearman_min']:.3f}")
    r = report["reference"]
    lines.append(f"riferimenti: testo presente (w035) {r['w035_text_present']:.2f}, "
                 f"controllo rimescolato {r['shuffled_control']:.2f}")
    return "\n".join(lines)
