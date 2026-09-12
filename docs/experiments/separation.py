"""Separation experiment: text-bearing segments vs raw papyrus.

Reproduces the table in docs/feature_selection.md. Expects, relative to the repo
root, the tutorial's w035 predictions in data/w035_pred/ and the PHerc1447
projections in data/1447/ (see docs/baseline_pherc1447.md for how to fetch them).

    python3 docs/experiments/separation.py
"""
import glob
import sys

import numpy as np

sys.path.insert(0, ".")
from scrollrank.letterness import ScoreConfig, load_image, score_image  # noqa: E402

SUBS = ("line_periodicity", "stroke_shape", "anisotropy", "ink_fraction")


def main() -> int:
    cases = [(f"w035 {s} TEXT", f"data/w035_pred/{s}.tif", 34.0)
             for s in ("seed42", "seed43")
             if glob.glob(f"data/w035_pred/{s}.tif")]
    for f in sorted(glob.glob("data/1447/*_avgc.tif")):
        cases.append((f.split("/")[-1].replace("_avgc.tif", "") + " PAPYRUS", f, 8.64))
    if not cases:
        print("no inputs found — see the module docstring")
        return 1

    head = f"{'segment':<30}{'score p95':>11}{'max':>8}"
    head += "".join(f"{s[:9]:>11}" for s in SUBS) + f"{'n':>7}"
    print(head)
    print("-" * len(head))
    for name, path, px in cases:
        cfg = ScoreConfig(pixel_size_um=px, window_mm=10.0, stride_mm=2.0,
                          auto_mask=False)
        ws, _, _ = score_image(load_image(path), cfg)
        s = np.array([w.score for w in ws])
        row = f"{name:<30}{np.percentile(s, 95):>11.3f}{s.max():>8.3f}"
        for sub in SUBS:
            v = np.array([getattr(w, sub) for w in ws])
            row += f"{np.percentile(v, 95):>11.3f}"
        print(row + f"{len(ws):>7d}")
    print("\n95th percentiles over all windows; auto_mask off on both classes so "
          "the two are treated identically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
