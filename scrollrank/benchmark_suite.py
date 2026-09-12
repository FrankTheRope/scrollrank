"""
benchmark_suite.py — the same benchmark over many segments, with paired statistics.

A single segment cannot settle whether a feature earns its place. On PHerc0139
w035 the full score reached AP 0.643 and `line_periodicity` alone 0.663; the
order flipped on the other seed. Differences of that size are indistinguishable
from noise on one segment, and reading them as a result is how a pipeline ends
up carrying features nobody has shown to help.

This module runs `benchmark.run` over a list of segments and reports, for every
ranking, the mean and standard deviation of Average Precision ACROSS segments,
plus a paired comparison against the full score: the mean paired difference, how
many segments each ranking wins, and a Wilcoxon signed-rank p-value. Paired,
because the segments differ enormously among themselves — comparing two rankings
on the same segment removes that variance, comparing pooled means does not.

Two modes
---------
`predictions`   real model predictions vs hand annotations. Measures the whole
                chain: how well the ranking finds text in what a model produced.

`labels`        the annotation used as its own prediction. Measures the scorer
                alone under ideal detection — an upper bound. If the scorer does
                poorly here, no ink model can rescue it, and the limitation is in
                the window statistics, not in the detector.

Manifest format (JSON list):

    [{"name": "pherc0139-w035",
      "prediction": "data/w035_pred/seed42.tif",
      "label": "ink_9um/.../w035_inklabels.tif",
      "pixel_size_um": 34.0}, ...]

In `labels` mode `prediction` may be omitted; the label is used for both.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .benchmark import run as run_one
from .letterness import ScoreConfig

FULL = "ScrollRank (full)"


def _signed_rank_p(diff: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; nan when there is nothing to test."""
    d = diff[np.abs(diff) > 1e-12]
    if d.size < 3:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def run_suite(manifest: list[dict], out_dir: str | Path, mode: str = "predictions",
              window_mm: float = 10.0, stride_mm: float = 2.0,
              working_um: float = 100.0, region_mm: float | None = None,
              min_label_frac: float = 0.005, pitch_mm: tuple[float, float] = (3.0, 9.0),
              letter_mm: tuple[float, float] = (1.5, 5.0)) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_segment: list[dict] = []
    for entry in manifest:
        name = entry["name"]
        label = entry["label"]
        pred = label if mode == "labels" else entry["prediction"]
        cfg = ScoreConfig(
            pixel_size_um=float(entry["pixel_size_um"]), working_um=working_um,
            window_mm=window_mm, stride_mm=stride_mm, line_pitch_mm=pitch_mm,
            letter_height_mm=letter_mm, auto_mask=False,
        )
        try:
            rep = run_one(pred, label, cfg, out_dir / name,
                          min_label_frac=min_label_frac, region_mm=region_mm)
        except Exception as e:                       # keep the suite going
            per_segment.append({"name": name, "error": str(e)})
            print(f"  {name}: ERRORE {e}")
            continue
        ap = {k: v["average_precision"] for k, v in rep["results"].items()}
        per_segment.append({
            "name": name,
            "n_windows": rep["positives"]["n_windows"],
            "n_regions": rep["positives"]["n_regions"],
            "prevalence": rep["positives"]["prevalence"],
            "alignment_peak_ratio": rep["alignment"]["peak_ratio"],
            "average_precision": ap,
            "saturation": rep["diagnostics"]["saturation"],
        })
        print(f"  {name}: {rep['positives']['n_windows']} finestre, "
              f"{rep['positives']['n_regions']} regioni, AP(full)={ap.get(FULL, float('nan')):.3f}")

    ok = [s for s in per_segment if "average_precision" in s]
    if not ok:
        raise RuntimeError("nessun segmento valutato con successo")

    names = list(ok[0]["average_precision"].keys())
    matrix = {n: np.array([s["average_precision"][n] for s in ok], dtype=float) for n in names}
    full = matrix[FULL]

    aggregate = {}
    for n in names:
        v = matrix[n]
        d = v - full
        aggregate[n] = {
            "ap_mean": round(float(np.nanmean(v)), 4),
            "ap_std": round(float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0, 4),
            "paired_diff_vs_full_mean": round(float(np.nanmean(d)), 4),
            "paired_diff_vs_full_std": round(float(np.nanstd(d, ddof=1)) if d.size > 1 else 0.0, 4),
            "wins_vs_full": int((d > 0).sum()),
            "losses_vs_full": int((d < 0).sum()),
            "wilcoxon_p_vs_full": round(_signed_rank_p(d), 4),
        }

    # how often each sub-score is saturated, averaged over segments
    sat_keys = list(ok[0]["saturation"].keys())
    saturation = {k: {
        "frac_at_1.00_mean": round(float(np.mean([s["saturation"][k]["frac_at_1.00"] for s in ok])), 3),
        "distinct_values_median": int(np.median([s["saturation"][k]["distinct_values"] for s in ok])),
    } for k in sat_keys}

    report = {
        "mode": mode,
        "n_segments": len(ok),
        "n_failed": len(per_segment) - len(ok),
        "settings": {"window_mm": window_mm, "stride_mm": stride_mm,
                     "region_mm": region_mm if region_mm is not None else window_mm / 2,
                     "min_label_frac": min_label_frac,
                     "pitch_mm": list(pitch_mm), "letter_mm": list(letter_mm)},
        "aggregate": aggregate,
        "saturation": saturation,
        "per_segment": per_segment,
        "caveat": ("Paired differences are read across segments, so segment-to-segment "
                   "variance is removed; the Wilcoxon p-value needs at least a handful "
                   "of segments to mean anything and is reported as nan below three."),
    }
    (out_dir / "suite.json").write_text(json.dumps(report, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = sorted(names, key=lambda n: -aggregate[n]["ap_mean"])
        vals = [aggregate[n]["ap_mean"] for n in order]
        errs = [aggregate[n]["ap_std"] for n in order]
        colors = ["#1F77B4" if n == FULL else
                  ("#C0392B" if n.startswith("baseline") else "#7F8C8D") for n in order]
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 1.6))
        ax.barh(range(len(order)), vals, xerr=errs, color=colors, edgecolor="#333",
                error_kw={"ecolor": "#333", "capsize": 3, "lw": 1})
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8.5)
        ax.invert_yaxis(); ax.set_xlabel("Average Precision (media su %d segmenti, ±1σ)" % len(ok))
        ax.set_title("Ranking delle finestre — modalità '%s'" % mode, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(out_dir / "suite_ap.png", dpi=150); plt.close(fig)
    except ImportError:
        pass

    return report


def format_suite(report: dict) -> str:
    a = report["aggregate"]
    order = sorted(a, key=lambda n: -a[n]["ap_mean"])
    head = f"{'ranking':<44}{'AP medio':>10}{'sigma':>8}{'diff vs full':>14}{'W/L':>8}{'p':>8}"
    lines = [head, "-" * len(head)]
    for n in order:
        r = a[n]
        p = r["wilcoxon_p_vs_full"]
        ps = "  --  " if p != p else f"{p:.3f}"
        lines.append(f"{n:<44}{r['ap_mean']:>10.3f}{r['ap_std']:>8.3f}"
                     f"{r['paired_diff_vs_full_mean']:>+14.3f}"
                     f"{r['wins_vs_full']:>4d}/{r['losses_vs_full']:<3d}{ps:>8}")
    lines.append("-" * len(head))
    lines.append(f"{report['n_segments']} segmenti valutati"
                 + (f", {report['n_failed']} falliti" if report["n_failed"] else "")
                 + f"; modalita' '{report['mode']}'")
    lines.append("")
    lines.append("saturazione media dei sub-punteggi:")
    for k, v in report["saturation"].items():
        lines.append(f"  {k:<20} a 1.00: {v['frac_at_1.00_mean']:.2f}   "
                     f"valori distinti (mediana): {v['distinct_values_median']}")
    return "\n".join(lines)


def discover_labels(root: str | Path, pixel_size_um: float) -> list[dict]:
    """Build a `labels`-mode manifest from every *_inklabels.tif under `root`."""
    root = Path(root)
    out = []
    for f in sorted(root.rglob("*_inklabels.tif")):
        out.append({"name": f.parent.name, "label": str(f), "pixel_size_um": pixel_size_um})
    return out
