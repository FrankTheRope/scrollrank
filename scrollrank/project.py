"""
project.py — turn a folder of surface-volume TIFF slices into 2D projections.

A rendered segment is a stack of N slices sampled along the surface normal
(typically 21-31). Ink models consume the stack, but for *looking* and for
scoring you need a single 2D image. Which projection is best is not known in
advance, so this writes several:

  mid   the central slice — sharpest, closest to the surface itself
  max   maximum over depth — what the tutorials use; boosts anything bright
        at any depth, including ink-like deposits, but also noise
  avgc  mean of the central 2k+1 slices — usually the most readable of the three
  minc  minimum over the central band — occasionally reveals dark features

Slices are read lazily one at a time, so a big stack does not have to fit in
memory twice.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import tifffile


def load_stack(folder: str | Path) -> tuple[np.ndarray, list[str]]:
    files = sorted(glob.glob(str(Path(folder) / "*.tif")) + glob.glob(str(Path(folder) / "*.tiff")))
    if not files:
        raise FileNotFoundError(f"no TIFF slices in {folder}")
    first = tifffile.imread(files[0])
    stack = np.empty((len(files),) + first.shape, dtype=first.dtype)
    stack[0] = first
    for i, f in enumerate(files[1:], start=1):
        a = tifffile.imread(f)
        if a.shape != first.shape:
            raise ValueError(f"{f} has shape {a.shape}, expected {first.shape}")
        stack[i] = a
    return stack, files


def project(folder: str | Path, out_prefix: str | Path, half_band: int = 4) -> dict:
    stack, files = load_stack(folder)
    n = len(files)
    c = n // 2
    lo, hi = max(0, c - half_band), min(n, c + half_band + 1)
    band = stack[lo:hi]
    outputs = {
        "mid": stack[c],
        "max": stack.max(axis=0),
        "avgc": band.mean(axis=0).astype(stack.dtype),
        "minc": band.min(axis=0),
    }
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, img in outputs.items():
        p = out_prefix.with_name(out_prefix.name + f"_{name}.tif")
        tifffile.imwrite(str(p), img)
        written[name] = str(p)
    valid = float((stack[c] > 0).mean())
    return {
        "n_slices": n, "shape": [int(v) for v in stack[c].shape],
        "dtype": str(stack.dtype), "central_band": [lo, hi],
        "valid_fraction_mid": round(valid, 3), "outputs": written,
    }
