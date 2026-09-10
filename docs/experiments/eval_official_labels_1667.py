"""Evaluate PHerc1667 w028 maps against the OFFICIAL public ink labels (ink-labels 2026-07).

    python3 docs/experiments/eval_official_labels_1667.py            # fetch labels, align, score all known maps

The experiments in positive_control_pherc1667.md, mil_pherc1667.md and mil_finetune_pherc1667.md
scored every map against the team's published `canon` *prediction* of w028.  The Challenge asks
ink-detection work to be evaluated on its public ink-labels dataset, which has w028:

    https://huggingface.co/buckets/scrollprize/datasets/tree/ink/1667/w028_20251208130119156_2um
      w028_..._inklabels_v2.tif, w028_..._supervision_mask_v2.tif, w028_..._validation_mask_v2.tif
      (37852 x 30583 at 2.4 um, binary)

Alignment.  The label canvas (37852 x 30583) differs from the Data Browser `flatboi` surface volume
(37420 x 30340) that the crop was cut from, by a translation.  A cross-correlation of the labels with
the canon ds8 preview (done once, by hand) gave the search window used here; this script then grid-
searches the offset inside it at 9.6 um by maximising the public ink_9um map's AUC against the labels,
and checks that the optimum is interior and falls off in every direction (0.886 at the optimum, 0.880
at 8 px, 0.864 at 16 px, 0.80 at 32 px).  Because the offset is fitted on that map, its 0.886 is a
best case; the other maps are scored at the same offset.

Maps.  The maps scored are the ones listed in MAPS below: 2000 x 2800 maps of this crop, or 2000 x 1400
maps of its held-out half.  To score a new map of the crop, add it to MAPS.  The inputs come from
build_control_pherc1667.py (crop, canon preview, Kaggle ink_9um runs) and build_mil_1667.py.

Evaluation.  Pixel AUC at 9.6 um, and at 38 um (4x4 cells at least half supervised), inside the
official supervision mask, on the held-out half of the crop (x >= 1400).  In this crop the
supervision mask equals the validation mask and lies entirely in the held-out half (11.2 % of it,
23.8 % ink), so no training arm ever saw these pixels.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import tifffile
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OFF = ROOT / "data" / "1667_official"
BUCKET = ("https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/1667/"
          "w028_20251208130119156_2um/w028_20251208130119156_{}.tif")
CROP_Y0, CROP_X0, CROP_H, CROP_W, SPLIT = 4600, 600, 2000, 2800, 1400   # crop in flatboi pyramid level 2 (9.596 um)

MAPS = {  # label: path (full-crop 2000x2800 uint8 tif, or held-out-half 2000x1400 float npy)
    "team canon prediction (reference used so far)": "work/mil_1667/mil_1667_w028.npz:truth",
    "public ink_9um, forward": "work/mil_1667/mil_1667_w028.npz:ink9um",
    "public ink_9um, reversed depth order": "data/1667_ctrl/w028_crop_s43_d5_reverse.tif",
    "from scratch: supervised": "work/mil_1667/run42/mil_1667_run/supervised_prob_test.npy",
    "from scratch: mil_oracle": "work/mil_1667/run42/mil_1667_run/mil_oracle_prob_test.npy",
    "from scratch: mil_free": "work/mil_1667/run42/mil_1667_run/mil_free_prob_test.npy",
    "fine-tune: baseline": "work/ft_kaggle/ft_t4/baseline_prob_test.npy",
    "fine-tune: mil_free + distillation": "work/ft_kaggle/ft_t4/mil_free_prob_test.npy",
    "fine-tune: mil_free, no distillation": "work/ft_kaggle/ft_l2sp/mil_free_prob_test.npy",
    "fine-tune: mil_oracle + distillation": "work/ft_kaggle/ft_t4/mil_oracle_prob_test.npy",
    "fine-tune: mil_oracle, no distillation": "work/ft_kaggle/ft_l2sp/mil_oracle_prob_test.npy",
    "fine-tune: supervised on canon prediction": "work/ft_kaggle/ft_t4/supervised_prob_test.npy",
}


def fetch() -> None:
    OFF.mkdir(parents=True, exist_ok=True)
    for kind in ("inklabels_v2", "supervision_mask_v2", "validation_mask_v2"):
        dst = OFF / f"w028_20251208130119156_{kind}.tif"
        if not dst.exists():
            req = urllib.request.Request(BUCKET.format(kind), headers={"User-Agent": "Mozilla/5.0"})
            dst.write_bytes(urllib.request.urlopen(req, timeout=300).read())


def ds(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))


def held_out(path: str) -> np.ndarray:
    if ":" in path:
        f, key = path.split(":")
        return np.load(ROOT / f)[key][:, SPLIT:].astype(np.float64) / 255
    if path.endswith(".tif"):
        return tifffile.imread(ROOT / path)[:, SPLIT:].astype(np.float64) / 255
    return np.load(ROOT / path).astype(np.float64)


def main() -> int:
    fetch()
    lab = ds((tifffile.imread(OFF / "w028_20251208130119156_inklabels_v2.tif") > 0).astype(np.float32), 4)
    sup = ds((tifffile.imread(OFF / "w028_20251208130119156_supervision_mask_v2.tif") > 0).astype(np.float32), 4)
    ink = np.load(ROOT / "work/mil_1667/mil_1667_w028.npz")["ink9um"].astype(np.float32) / 255

    def auc_at(oy, ox):
        L = lab[CROP_Y0 + oy:CROP_Y0 + oy + CROP_H, CROP_X0 + ox:CROP_X0 + ox + CROP_W]
        m = sup[CROP_Y0 + oy:CROP_Y0 + oy + CROP_H, CROP_X0 + ox:CROP_X0 + ox + CROP_W] > 0.5
        y = L[m] >= 0.5
        return roc_auc_score(y, ink[m]) if m.sum() > 5000 and 0 < y.mean() < 1 else float("nan")

    grid = {(oy, ox): auc_at(oy, ox) for oy in range(50, 111, 4) for ox in range(30, 131, 4)}
    grid = {k: v for k, v in grid.items() if v == v}
    b = max(grid, key=grid.get)
    fine = {(oy, ox): auc_at(oy, ox) for oy in range(b[0] - 3, b[0] + 4) for ox in range(b[1] - 3, b[1] + 4)}
    oy, ox = max(fine, key=fine.get)
    falloff = {d: (round(auc_at(oy + d, ox), 4), round(auc_at(oy, ox + d), 4)) for d in (0, 8, 16, 32)}
    print(f"alignment offset (level-2 px, label = crop + origin + offset): ({oy}, {ox}); fall-off {falloff}")

    L = lab[CROP_Y0 + oy:CROP_Y0 + oy + CROP_H, CROP_X0 + ox:CROP_X0 + ox + CROP_W][:, SPLIT:] >= 0.5
    M = sup[CROP_Y0 + oy:CROP_Y0 + oy + CROP_H, CROP_X0 + ox:CROP_X0 + ox + CROP_W][:, SPLIT:] > 0.5
    Lp, Mp = ds(L.astype(np.float32), 4), ds(M.astype(np.float32), 4)
    canon = np.load(ROOT / "work/mil_1667/mil_1667_w028.npz")["truth"][:, SPLIT:] >= 128
    print(f"official labels inside the held-out half: {int(M.sum())} px ({M.mean() * 100:.1f} %), ink {L[M].mean() * 100:.1f} % "
          f"(canon marks {canon[M].mean() * 100:.1f} % of the same pixels as ink)")
    rows = []
    print(f"{'map':48s} {'vs official 9.6um':>17s} {'38um':>7s} {'vs canon, same px':>18s}")
    for name, path in MAPS.items():
        if not (ROOT / path.split(":")[0]).exists():
            print(f"{name:48s} (missing: {path})")
            continue
        m = held_out(path)
        a = roc_auc_score(L[M], m[M])
        k = Mp >= 0.5
        a38 = roc_auc_score(Lp[k] >= 0.5, ds(m, 4)[k])
        c = roc_auc_score(canon[M], m[M]) if "canon prediction (reference" not in name else float("nan")
        rows.append({"map": name, "auc_official": a, "auc_official_38um": a38, "auc_canon_same_px": c})
        print(f"{name:48s} {a:17.4f} {a38:7.4f} {c:18.4f}")
    out = ROOT / "out" / "official_labels_1667.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"offset": [oy, ox], "falloff": falloff, "n_px": int(M.sum()), "rows": rows}, indent=1))
    print("wrote", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
