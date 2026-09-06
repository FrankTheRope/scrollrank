"""Screen the ink predictions the Challenge team published for the raw
z_dbg_gen segments of PHerc1447 (three models: gp, s5, tracer_ft; both
directions) with ScrollScout, CPU only.

    python3 docs/experiments/screen_pherc1447_raw.py [--workers 6]

Input:  data/1447_raw_pred/<segment>/<segment>_<model>[_reverse].jpg
        (mirror of s3://vesuvius-challenge-open-data/PHerc1447/segments/raw/)
Output: out/screen_1447_raw/<segment>/<image>.json + .grid.npy
        out/screen_1447_raw/summary.json, summary.md

Per image: score grid, top windows, max, p95.  Per segment and direction:
cross-model Spearman on the score grids (the concordance test), and every
window above threshold with the value the other two models give in the same
cell.  A window seen by one model only is not a finding.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scrollscout.letterness import ScoreConfig, load_image, score_image  # noqa: E402
from scrollscout.concordance import _spearman  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "1447_raw_pred"
OUT = ROOT / "out" / "screen_1447_raw"
MODELS = ("gp", "s5", "tracer_ft")
PIXEL_UM = 8.64          # preview JPGs are 1:1 with the 8.64 µm volume (x.tif at scale 0.05 matches)
THRESH = 0.6
CFG = ScoreConfig(pixel_size_um=PIXEL_UM, window_mm=10.0, stride_mm=2.0, top_k=10)


def score_one(path: str) -> dict:
    p = Path(path)
    out = OUT / p.parent.name
    out.mkdir(parents=True, exist_ok=True)
    js = out / (p.stem + ".json")
    if js.exists() and (out / (p.stem + ".grid.npy")).exists():
        return json.loads(js.read_text())
    img = load_image(p)
    ws, grid, meta = score_image(img, CFG)
    scores = np.array([w.score for w in ws]) if ws else np.zeros(1)
    top = [dict(row=w.row, col=w.col, score=round(w.score, 3), period=round(w.line_periodicity, 3),
                stroke=round(w.stroke_shape, 3), aniso=round(w.anisotropy, 3), ink=round(w.raw_ink_fraction, 3),
                pitch_mm=round(w.best_pitch_mm, 2), angle=w.best_angle_deg, cov=round(w.coverage, 2),
                box_px=[w.x0, w.y0, w.x1, w.y1]) for w in ws[:20]]
    rec = dict(image=p.name, segment=p.parent.name, shape=list(img.shape),
               n_windows=int(meta["n_windows_scored"]), max=round(float(scores.max()), 3),
               p95=round(float(np.percentile(scores, 95)), 3), n_above=int((scores > THRESH).sum()),
               nonzero_frac=round(float((img > 0).mean()), 3), top=top)
    np.save(out / (p.stem + ".grid.npy"), grid)
    js.write_text(json.dumps(rec, indent=1))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    segs = sorted(d for d in DATA.iterdir() if d.name.startswith("z_dbg_gen") and list(d.glob("*.jpg")))
    paths = [str(p) for s in segs for p in sorted(s.glob("*.jpg"))]
    print(f"{len(segs)} segments, {len(paths)} images", flush=True)
    recs = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(score_one, paths)):
            recs[r["image"]] = r
            print(f"[{i + 1}/{len(paths)}] {r['image']}: max {r['max']:.3f} p95 {r['p95']:.3f} "
                  f">{THRESH}: {r['n_above']} (n={r['n_windows']})", flush=True)

    summary = []
    for s in segs:
        for direction, suffix in (("fwd", ""), ("rev", "_reverse")):
            names = {m: f"{s.name}_{m}{suffix}.jpg" for m in MODELS}
            if not all(n in recs for n in names.values()):
                continue
            grids = {m: np.load(OUT / s.name / (Path(n).stem + ".grid.npy")) for m, n in names.items()}
            pairs = {}
            for x, y in itertools.combinations(MODELS, 2):
                pairs[f"{x}/{y}"] = round(_spearman(grids[x], grids[y]), 3)
            sp = [v for v in pairs.values() if v == v]
            cands = []
            for m in MODELS:
                for w in recs[names[m]]["top"]:
                    if w["score"] <= THRESH:
                        continue
                    others = {o: round(float(grids[o][w["row"], w["col"]]), 3) for o in MODELS if o != m}
                    cands.append(dict(model=m, score=w["score"], pitch_mm=w["pitch_mm"], angle=w["angle"],
                                      stroke=w["stroke"], box_px=w["box_px"], others=others,
                                      supported=max(others.values()) >= 0.5))
            summary.append(dict(segment=s.name, direction=direction,
                                n_windows=recs[names["gp"]]["n_windows"],
                                max={m: recs[names[m]]["max"] for m in MODELS},
                                p95={m: recs[names[m]]["p95"] for m in MODELS},
                                spearman=pairs, spearman_mean=round(float(np.mean(sp)), 3) if sp else None,
                                spearman_min=round(float(np.min(sp)), 3) if sp else None,
                                candidates=cands))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    lines = ["| segment | dir | windows | max gp | max s5 | max tracer_ft | p95 gp | p95 s5 | p95 tracer_ft | "
             "Spearman gp/s5 | gp/tr | s5/tr | mean | windows >0.6 | supported |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        sp = r["spearman"]
        lines.append(f"| {r['segment']} | {r['direction']} | {r['n_windows']} | "
                     f"{r['max']['gp']:.3f} | {r['max']['s5']:.3f} | {r['max']['tracer_ft']:.3f} | "
                     f"{r['p95']['gp']:.3f} | {r['p95']['s5']:.3f} | {r['p95']['tracer_ft']:.3f} | "
                     f"{sp['gp/s5']:.2f} | {sp['gp/tracer_ft']:.2f} | {sp['s5/tracer_ft']:.2f} | {r['spearman_mean']:.2f} | "
                     f"{len(r['candidates'])} | {sum(c['supported'] for c in r['candidates'])} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
