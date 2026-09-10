"""Figure: PHerc1667 w028, official ink labels next to the public model and two fine-tuned variants.

    python3 docs/experiments/eval_official_labels_1667.py   # first: fetches the labels, writes the offset
    python3 docs/experiments/fig_control_1667.py            # -> docs/img/control_and_finetune_1667.png

The panels show the part of the held-out half that the official labels annotate (their validation
region, a 9.6 x 3.6 mm strip).  The AUC under each title is the one in out/official_labels_1667.json:
the whole annotated region of the held-out half, 9.6 um pixels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_official_labels_1667 import CROP_H, CROP_W, CROP_X0, CROP_Y0, OFF, ROOT, SPLIT, ds, held_out  # noqa: E402

PANELS = [  # (title, key in MAPS of eval_official_labels_1667, path)
    ("team's canon prediction", "team canon prediction (reference used so far)", "work/mil_1667/mil_1667_w028.npz:truth"),
    ("public ink_9um", "public ink_9um, forward", "work/mil_1667/mil_1667_w028.npz:ink9um"),
    ("public ink_9um,\nslices in reversed depth order", "public ink_9um, reversed depth order", "data/1667_ctrl/w028_crop_s43_d5_reverse.tif"),
    ("ink_9um fine-tuned\non row geometry only", "fine-tune: mil_free + distillation", "work/ft_kaggle/ft_t4/mil_free_prob_test.npy"),
    ("ink_9um fine-tuned\non the canon prediction", "fine-tune: supervised on canon prediction", "work/ft_kaggle/ft_t4/supervised_prob_test.npy"),
]


def main() -> int:
    res = json.loads((ROOT / "out" / "official_labels_1667.json").read_text())
    auc = {r["map"]: r["auc_official"] for r in res["rows"]}
    oy, ox = res["offset"]
    win = (slice(CROP_Y0 + oy, CROP_Y0 + oy + CROP_H), slice(CROP_X0 + ox, CROP_X0 + ox + CROP_W))
    lab = ds((tifffile.imread(OFF / "w028_20251208130119156_inklabels_v2.tif") > 0).astype(np.float32), 4)[win][:, SPLIT:] >= 0.5
    sup = ds((tifffile.imread(OFF / "w028_20251208130119156_supervision_mask_v2.tif") > 0).astype(np.float32), 4)[win][:, SPLIT:] > 0.5

    ys, xs = np.nonzero(sup)
    r, c = slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)
    shown = np.where(sup, lab.astype(float), 0.5)[r, c]

    fig, axes = plt.subplots(1, len(PANELS) + 1, figsize=(2.3 * (len(PANELS) + 1), 6.4))
    axes[0].imshow(shown, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("official ink labels\n(ink-labels 2026-07)", fontsize=9, fontweight="bold", loc="left")
    axes[0].text(0, -0.03, "grey: not annotated", transform=axes[0].transAxes, fontsize=8, va="top")
    for ax, (title, key, path) in zip(axes[1:], PANELS):
        m = held_out(path)[r, c]
        ax.imshow(m, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.contour(sup[r, c], levels=[0.5], colors="#33aaff", linewidths=0.5)
        ax.set_title(title, fontsize=9, fontweight="bold", loc="left")
        ax.text(0, -0.03, f"AUC vs official labels: {auc[key]:.3f}", transform=ax.transAxes, fontsize=8, va="top")
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    h, w = shown.shape
    fig.suptitle(f"PHerc1667 w028, held-out half: the strip annotated by the official labels, {h * 0.0096:.1f} x {w * 0.0096:.1f} mm "
                 f"at 9.6 um/px (blue: edge of the annotated region).\nThe public model marks where the letters are, as blobs, "
                 f"and well in one depth order only; neither fine-tuning improves it on the official labels.",
                 fontsize=8, x=0.01, y=0.008, ha="left", va="bottom")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.11, wspace=0.05)
    out = ROOT / "docs" / "img" / "control_and_finetune_1667.png"
    fig.savefig(out, dpi=110)
    print("wrote", out.relative_to(ROOT), f"strip {h}x{w} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
