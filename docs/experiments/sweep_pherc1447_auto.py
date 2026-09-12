"""Score the ink_9um depth sweep on the eleven PHerc1447 auto_grown segments
rendered locally (docs/RENDERING.md) and run on Kaggle
(docs/experiments/kaggle_ink9um_sweep.sh).

    python3 docs/experiments/sweep_pherc1447_auto.py [--workers 6]
    SCROLL=1203 python3 docs/experiments/sweep_pherc1447_auto.py   # same for PHerc1203 (9.362 µm)

Input:  data/1447_sweep/preds/<segment>_s<seed>_d<depth>[_reverse].tif
        work/1447_render/<segment>_render.zarr        (mask: where the mesh is)
Output: out/sweep_1447_auto/<segment>/<image>_a<angle>.json + .grid.npy
        out/sweep_1447_auto/summary.json, summary.md

The prediction is nonzero wherever a 128-px patch touched the mesh, which is
much more than the mesh itself on these sparse renders, so the valid mask
comes from the render (any nonzero voxel in the stack), not from the
prediction.  Scored at 0°, +45°, -45° because fibre direction varies between
these meshes.  Per segment, depth and direction: seed-42/seed-43 Spearman
(the concordance test), max and p95 per seed, and every window above 0.6
with the other seed's value in the same cell.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scrollrank.letterness import ScoreConfig, load_image, score_image  # noqa: E402
from scrollrank.concordance import _spearman  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCROLLS = {  # pred dir, render dir, out dir, voxel size (µm)
    "1447": (ROOT / "data" / "1447_sweep" / "preds", ROOT / "work" / "1447_render", ROOT / "out" / "sweep_1447_auto", 8.64),
    "1203": (ROOT / "data" / "1203_sweep", ROOT / "work" / "1203_render", ROOT / "out" / "sweep_1203_auto", 9.362),
}
import os as _os
PRED, REND, OUT, PIXEL_UM = SCROLLS[_os.environ.get("SCROLL", "1447")]
THRESH = 0.6
ANGLES = (0, 45, -45)
CFG = ScoreConfig(pixel_size_um=PIXEL_UM, window_mm=10.0, stride_mm=2.0, top_k=10, auto_mask=False)
NAME = re.compile(r"(?P<seg>auto_grown_\d+)_s(?P<seed>\d+)_d(?P<depth>\d+)(?P<rev>_reverse)?\.tif$")


def render_mask(seg: str) -> np.ndarray:
    import zarr
    warnings.filterwarnings("ignore")
    a = zarr.open(str(REND / f"{seg}_render.zarr"), mode="r")["0"]
    m = np.zeros(a.shape[1:], dtype=bool)
    for k in range(0, a.shape[0], 5):
        m |= np.asarray(a[k]) > 0
    return m


def rotate(img: np.ndarray, mask: np.ndarray, angle: int):
    if angle == 0:
        return img, mask
    from scipy.ndimage import rotate as rot
    return (rot(img, angle, reshape=True, order=1, mode="constant", cval=0),
            rot(mask.astype(np.uint8), angle, reshape=True, order=0, mode="constant", cval=0) > 0)


def score_one(job) -> dict:
    path, angle = job
    p = Path(path)
    m = NAME.match(p.name)
    seg = m["seg"]
    out = OUT / seg
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{p.stem}_a{angle:+d}"
    js = out / (stem + ".json")
    if js.exists() and (out / (stem + ".grid.npy")).exists():
        return json.loads(js.read_text())
    img = load_image(p)
    mask = render_mask(seg)
    img, mask = rotate(img, mask, angle)
    ws, grid, meta = score_image(img, CFG, mask=mask)
    scores = np.array([w.score for w in ws]) if ws else np.zeros(1)
    top = [dict(row=w.row, col=w.col, score=round(w.score, 3), period=round(w.line_periodicity, 3),
                stroke=round(w.stroke_shape, 3), ink=round(w.raw_ink_fraction, 3),
                pitch_mm=round(w.best_pitch_mm, 2), angle=w.best_angle_deg, cov=round(w.coverage, 2),
                box_px=[w.x0, w.y0, w.x1, w.y1]) for w in ws[:20]]
    rec = dict(image=p.name, segment=seg, seed=int(m["seed"]), depth=int(m["depth"]),
               direction="rev" if m["rev"] else "fwd", rot=angle,
               n_windows=int(meta["n_windows_scored"]), max=round(float(scores.max()), 3),
               p95=round(float(np.percentile(scores, 95)), 3), n_above=int((scores > THRESH).sum()),
               nonzero_frac=round(float((img[mask] > 0).mean()), 3) if mask.any() else 0.0, top=top)
    np.save(out / (stem + ".grid.npy"), grid)
    js.write_text(json.dumps(rec, indent=1))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    paths = sorted(p for p in PRED.glob("*.tif") if NAME.match(p.name))
    jobs = [(str(p), ang) for p in paths for ang in ANGLES]
    print(f"{len(paths)} predictions x {len(ANGLES)} angles = {len(jobs)} scorings", flush=True)
    recs = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(score_one, jobs)):
            recs[(r["segment"], r["seed"], r["depth"], r["direction"], r["rot"])] = r
            if r["rot"] == 0:
                print(f"[{i + 1}/{len(jobs)}] {r['image']}: max {r['max']:.3f} p95 {r['p95']:.3f} >{THRESH}: {r['n_above']} (n={r['n_windows']})", flush=True)

    segs = sorted({k[0] for k in recs})
    summary = []
    for seg in segs:
        for depth in (0, 7, 14):
            for direction in ("fwd", "rev"):
                for rot in ANGLES:
                    r42, r43 = recs.get((seg, 42, depth, direction, rot)), recs.get((seg, 43, depth, direction, rot))
                    if not (r42 and r43):
                        continue
                    g = {s: np.load(OUT / seg / (Path(r["image"]).stem + f"_a{rot:+d}.grid.npy")) for s, r in ((42, r42), (43, r43))}
                    sp = _spearman(g[42], g[43])
                    cands = []
                    for s, r in ((42, r42), (43, r43)):
                        o = 43 if s == 42 else 42
                        for w in r["top"]:
                            if w["score"] > THRESH:
                                ov = float(g[o][w["row"], w["col"]])
                                cands.append(dict(seed=s, score=w["score"], pitch_mm=w["pitch_mm"], angle=w["angle"],
                                                  stroke=w["stroke"], box_px=w["box_px"], other_seed=round(ov, 3),
                                                  supported=ov >= 0.5))
                    summary.append(dict(segment=seg, depth=depth, direction=direction, rot=rot,
                                        n_windows=r42["n_windows"], spearman=round(sp, 3) if sp == sp else None,
                                        max42=r42["max"], max43=r43["max"], p95_42=r42["p95"], p95_43=r43["p95"],
                                        candidates=cands))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    lines = ["| segment | depth | dir | rot | windows | Spearman s42/s43 | max s42 | max s43 | p95 s42 | p95 s43 | >0.6 | supported |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        sp = "n/a" if r["spearman"] is None else f"{r['spearman']:.2f}"
        lines.append(f"| {r['segment'][11:]} | {r['depth']} | {r['direction']} | {r['rot']:+d} | {r['n_windows']} | {sp} | "
                     f"{r['max42']:.3f} | {r['max43']:.3f} | {r['p95_42']:.3f} | {r['p95_43']:.3f} | "
                     f"{len(r['candidates'])} | {sum(c['supported'] for c in r['candidates'])} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    # compact view: per segment, best concordance and best score over everything
    print("\n| segment | best Spearman (depth/dir/rot) | best score (seed/depth/dir/rot) | supported windows |")
    print("|---|---|---|---|")
    for seg in segs:
        rows = [r for r in summary if r["segment"] == seg]
        bs = max((r for r in rows if r["spearman"] is not None), key=lambda r: r["spearman"], default=None)
        bm = max(rows, key=lambda r: max(r["max42"], r["max43"]))
        seed = 42 if bm["max42"] >= bm["max43"] else 43
        nsup = sum(c["supported"] for r in rows for c in r["candidates"])
        bs_txt = "n/a" if bs is None else f"{bs['spearman']:.2f} ({bs['depth']}/{bs['direction']}/{bs['rot']:+d}, n={bs['n_windows']})"
        print(f"| {seg[11:]} | {bs_txt} | "
              f"{max(bm['max42'], bm['max43']):.3f} (s{seed}/{bm['depth']}/{bm['direction']}/{bm['rot']:+d}) | {nsup} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
