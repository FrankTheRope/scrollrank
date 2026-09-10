"""Build the data package for the MIL experiment on PHerc1667 w028
(docs/experiments/mil_train_1667.py).

    python3 docs/experiments/build_mil_1667.py

Inputs (produced earlier):
  work/kaggle_1667_w028/1667_render/w028_crop_render.zarr   27 slices, 9.596 µm in x, y and z
                                                            (built by build_control_pherc1667.py)
  work/ctrl_1667/truth_canon_9um.npy                        team's canon prediction, upsampled to the same grid
  data/1667_ctrl/w028_crop_s43_d5.tif                       public ink_9um prediction on the crop (Kaggle run)
Output:
  work/mil_1667/mil_1667_w028.npz with stack, truth, ink9um, bands_oracle, bands_free, meta.

Row bags. For each image row (y) a label 1 = text-row band, 0 = interline band (ink-free), -1 =
ambiguous. They are derived from the row profile of a map restricted to the TRAINING half
(x < split_x = W/2), smoothed (sigma 6 rows); rows above the 60th percentile of the profile are
1, rows below the 30th are 0. bands_oracle uses the truth mask; bands_free uses the public
ink_9um prediction and therefore never sees the truth.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import tifffile
import zarr
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")


def bands_from_profile(prof: np.ndarray, hi: float = 60, lo: float = 30, sigma: float = 6) -> np.ndarray:
    prof = gaussian_filter(prof.astype(np.float64), sigma)
    b = np.full(prof.size, -1, np.int8)
    b[prof >= np.percentile(prof, hi)] = 1
    b[prof <= np.percentile(prof, lo)] = 0
    return b


def main() -> int:
    stack = np.asarray(zarr.open("work/kaggle_1667_w028/1667_render/w028_crop_render.zarr", mode="r")["0"])
    truth = np.load("work/ctrl_1667/truth_canon_9um.npy").astype(np.uint8)
    ink = tifffile.imread("data/1667_ctrl/w028_crop_s43_d5.tif")
    H, W = truth.shape
    assert stack.shape[1:] == (H, W) == ink.shape
    split = W // 2
    b_oracle = bands_from_profile((truth[:, :split] >= 128).mean(axis=1))
    b_free = bands_from_profile(ink[:, :split].astype(np.float32).mean(axis=1))
    t = truth >= 128
    for name, b in (("oracle", b_oracle), ("ink9um-derived", b_free)):
        print(f"{name}: row rows {int((b == 1).sum())}, interline rows {int((b == 0).sum())}, ambiguous {int((b == -1).sum())} | "
              f"ink fraction in row bands {t[b == 1].mean():.3f}, in interlines {t[b == 0].mean():.3f}")
    os.makedirs("work/mil_1667", exist_ok=True)
    meta = dict(pixel_um=9.596, z_um=9.596, shape=list(stack.shape), split_x=split,
                truth="team canon prediction (2.4um model), 0-255, ink if >=128",
                bands="per-row labels: 1=text row band, 0=interline (no ink), -1=ambiguous; derived from training half x<split_x only",
                ink9um="public ink_9um seed43 depth 5-21 forward prediction on the same crop, 0-255")
    np.savez_compressed("work/mil_1667/mil_1667_w028.npz", stack=stack, truth=truth, ink9um=ink,
                        bands_oracle=b_oracle, bands_free=b_free, meta=json.dumps(meta))
    print("wrote work/mil_1667/mil_1667_w028.npz", round(os.path.getsize("work/mil_1667/mil_1667_w028.npz") / 1e6), "MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
