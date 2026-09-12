"""Score the PHerc1667 positive control (docs/positive_control_pherc1667.md).

    python3 docs/experiments/score_control_pherc1667.py [--preds data/1667_ctrl] [--canon work/kaggle_1667_w028/w028_crop_pred_canon.png]

For each of the twelve ink_9um predictions of the w028 crop: ScrollRank
score grid at 10 mm / 2 mm, max, p95, pitch; Spearman against the team's
published prediction of the same crop (same grid, both 26.9 mm wide) and
against the other seed. Prints the table of the document.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scrollrank.letterness import ScoreConfig, load_image, score_image  # noqa: E402
from scrollrank.concordance import _spearman  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="data/1667_ctrl")
    ap.add_argument("--canon", default="work/kaggle_1667_w028/w028_crop_pred_canon.png")
    a = ap.parse_args()
    cfg9 = ScoreConfig(pixel_size_um=9.596, window_mm=10, stride_mm=2, top_k=10, auto_mask=False)
    cfg19 = ScoreConfig(pixel_size_um=19.192, window_mm=10, stride_mm=2, top_k=10, auto_mask=False)
    wsC, gC, _ = score_image(load_image(a.canon), cfg19)
    sc = [w.score for w in wsC]
    print(f"| published canon 2.4 µm | {wsC[0].score:.3f} | {np.percentile(sc, 95):.2f} | {wsC[0].best_pitch_mm:.1f} mm | — | — |")
    res = {}
    for sd in (42, 43):
        for d in (0, 5, 10):
            for rev in ("", "_reverse"):
                p = Path(a.preds) / f"w028_crop_s{sd}_d{d}{rev}.tif"
                ws, g, _ = score_image(load_image(p), cfg9)
                s = np.array([w.score for w in ws])
                res[(sd, d, rev)] = (g, s.max(), np.percentile(s, 95), ws[0].best_pitch_mm)
    for d in (0, 5, 10):
        for rev in ("", "_reverse"):
            for sd in (42, 43):
                g, mx, p95, pitch = res[(sd, d, rev)]
                go = res[(85 - sd, d, rev)][0]
                print(f"| ink_9um s{sd}, depth {d}-{d + 16}, {'rev' if rev else 'fwd'} | {mx:.3f} | {p95:.2f} | {pitch:.1f} | "
                      f"{_spearman(g, gC):.2f} | {_spearman(g, go):.2f} |")
    # the instrument that settles the question: pixel-level AUC of each ink_9um map
    # against the published letters (truth = published prediction >= 128), 38 µm grid
    from sklearn.metrics import roc_auc_score
    canon = load_image(a.canon).astype(np.float32)
    truth = np.kron(canon, np.ones((2, 2), np.float32))          # 19.2 -> 9.6 µm grid

    def pool(x, f=4):
        return x[:x.shape[0] // f * f, :x.shape[1] // f * f].reshape(x.shape[0] // f, f, x.shape[1] // f, f).mean(axis=(1, 3))

    y = (pool(truth) >= 128).ravel()
    print("\n| ink_9um run | pixel AUC vs published letters |")
    for sd in (42, 43):
        for d in (0, 5, 10):
            for rev in ("", "_reverse"):
                img = load_image(Path(a.preds) / f"w028_crop_s{sd}_d{d}{rev}.tif").astype(np.float32)
                print(f"| s{sd} d{d}{' rev' if rev else ' fwd'} | {roc_auc_score(y, pool(img).ravel()):.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
