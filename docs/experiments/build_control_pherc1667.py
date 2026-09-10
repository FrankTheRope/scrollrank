"""Build the ink_9um input for the PHerc1667 positive control
(docs/positive_control_pherc1667.md).

    python3 docs/experiments/build_control_pherc1667.py [--out work/kaggle_1667_w028]

Takes pyramid level 2 (9.596 µm) of the 2.399 µm surface volume of segment
w028, every 4th z slice (27 of 109, 9.6 µm apart), on a text-dense crop of
27 x 19 mm, straight from the public bucket (uncompressed chunks, ~700 MB),
and writes a render-style OME-Zarr that docs/experiments/kaggle_ink9um_sweep.sh
accepts as <segment>_render.zarr. Also saves the matching crop of the team's
published prediction (ds8 preview) for the comparison.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np

SEG = "20251208130119-w028_20251208130119156_flatboi"
BASE = f"https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/segments/{SEG}/"
VOL = BASE + "surface-volumes/2.399um-0.22m-78keV-volume-20251217075048.zarr/"
PREVIEW = (BASE + "ink-detection/downsampled/PHerc1667-20251208130119-2.399um-0.22m-78keV-volume-"
           "20251217075048-20260417190342-new_canon_autoresearch_recipe-tile256-stride128-ds8.jpg")
LEVEL = "2"                       # 9.596 µm in x/y
X0, X1, Y0, Y1 = 600, 3400, 4600, 6600   # level-2 pixels; = ds8 region x300-1700, y2300-3300
Z_STEP, Z_FIRST = 4, 2            # 27 slices out of 109, centred on slice 54


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="work/kaggle_1667_w028")
    a = ap.parse_args()
    out = Path(a.out)
    meta = json.loads(urllib.request.urlopen(VOL + f"{LEVEL}/.zarray").read())
    Z, CH, CW = meta["chunks"]
    assert meta["compressor"] is None, "expected raw chunks"
    cy0, cy1, cx0, cx1 = Y0 // CH, (Y1 - 1) // CH, X0 // CW, (X1 - 1) // CW
    keys = [(cy, cx) for cy in range(cy0, cy1 + 1) for cx in range(cx0, cx1 + 1)]
    vol = np.zeros((Z, (cy1 - cy0 + 1) * CH, (cx1 - cx0 + 1) * CW), np.uint8)

    def fetch(k):
        cy, cx = k
        for _ in range(3):
            try:
                raw = urllib.request.urlopen(f"{VOL}{LEVEL}/0/{cy}/{cx}", timeout=60).read()
                return k, np.frombuffer(raw, np.uint8).reshape(Z, CH, CW)
            except Exception:
                pass
        return k, None

    print(f"fetching {len(keys)} chunks (~{len(keys) * Z * CH * CW / 1e6:.0f} MB)")
    with cf.ThreadPoolExecutor(16) as ex:
        for (cy, cx), arr in ex.map(fetch, keys):
            if arr is None:
                raise SystemExit(f"chunk {cy}/{cx} failed")
            vol[:, (cy - cy0) * CH:(cy - cy0 + 1) * CH, (cx - cx0) * CW:(cx - cx0 + 1) * CW] = arr
    oy, ox = Y0 - cy0 * CH, X0 - cx0 * CW
    sub = vol[Z_FIRST::Z_STEP, oy:oy + (Y1 - Y0), ox:ox + (X1 - X0)].copy()
    print("input stack", sub.shape)

    import zarr
    from numcodecs import Blosc
    zpath = out / "1667_render" / "w028_crop_render.zarr"
    shutil.rmtree(out, ignore_errors=True)
    zpath.parent.mkdir(parents=True)
    g = zarr.open_group(str(zpath), mode="w", zarr_format=2)
    comp = Blosc(cname="lz4", clevel=3, shuffle=1)
    lvl = sub
    for i in range(6):
        arr = g.create_array(str(i), shape=lvl.shape, chunks=(lvl.shape[0], 128, 128), dtype="uint8",
                             compressors=comp, chunk_key_encoding={"name": "v2", "separator": "/"})
        arr[:] = lvl
        if i < 5:
            H, W = lvl.shape[1] // 2 * 2, lvl.shape[2] // 2 * 2
            lvl = lvl[:, :H, :W].reshape(lvl.shape[0], H // 2, 2, W // 2, 2).mean(axis=(2, 4)).astype(np.uint8)
    g.attrs.update({
        "canvas_size": [X1 - X0, Y1 - Y0],
        "multiscales": [{"axes": [{"name": n, "type": "space", "unit": "micrometer"} for n in "zyx"],
                         "datasets": [{"path": str(i), "coordinateTransformations": [{"type": "scale", "scale": [9.596 * 2 ** i] * 2}]} for i in range(6)]}],
        "note": f"PHerc1667 {SEG}: level {LEVEL} of the 2.399 um surface volume, z every {Z_STEP}th slice from {Z_FIRST}, crop x{X0}-{X1} y{Y0}-{Y1}",
    })
    shutil.copy("docs/experiments/kaggle_ink9um_sweep.sh", out / "kaggle_ink9um_sweep.sh")
    from PIL import Image
    Image.fromarray(sub[sub.shape[0] // 2]).save(out / "w028_crop_render_mid.png")
    pv = Image.open(io.BytesIO(urllib.request.urlopen(PREVIEW, timeout=120).read()))
    pv.crop((X0 // 2, Y0 // 2, X1 // 2, Y1 // 2)).save(out / "w028_crop_pred_canon.png")
    shutil.make_archive(str(out), "zip", out.parent, out.name)
    print("wrote", out.with_suffix(".zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
