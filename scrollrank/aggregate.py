"""
aggregate.py — combine several ink predictions of the SAME segment.

Typical use: you ran ink_9um with 2 seeds x N checkpoints x M depth windows
(--layer-start/--layer-end) x both directions, and now have a folder of TIFFs
that all share the segment's shape. This module produces:

  mean.tif         average prediction (the "ensemble" image to look at)
  std.tif          disagreement between runs
  consistency.tif  mean / (std + eps), rescaled to 0-255: high where runs agree
                   that there is ink.  Text that only appears in one depth
                   window is a warning sign (surface offset or a false positive).

All inputs are robust-normalized to [0,1] first, so checkpoints with
different output ranges can be mixed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from .letterness import load_image, robust_normalize


def aggregate(paths: list[str | Path], out_dir: str | Path, eps: float = 0.05) -> dict:
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("no input images")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = None
    acc2 = None
    n = 0
    shape = None
    for p in paths:
        img = robust_normalize(load_image(p))
        if shape is None:
            shape = img.shape
            acc = np.zeros(shape, dtype=np.float64)
            acc2 = np.zeros(shape, dtype=np.float64)
        elif img.shape != shape:
            raise ValueError(f"{p} has shape {img.shape}, expected {shape}")
        acc += img
        acc2 += img.astype(np.float64) ** 2
        n += 1

    mean = acc / n
    var = np.maximum(acc2 / n - mean ** 2, 0.0)
    std = np.sqrt(var)
    consistency = mean / (std + eps)
    consistency = consistency / max(float(consistency.max()), 1e-6)

    def to_u8(a: np.ndarray) -> np.ndarray:
        return (np.clip(a, 0, 1) * 255).astype(np.uint8)

    tifffile.imwrite(out_dir / "mean.tif", to_u8(mean))
    tifffile.imwrite(out_dir / "std.tif", to_u8(std / max(float(std.max()), 1e-6)))
    tifffile.imwrite(out_dir / "consistency.tif", to_u8(consistency))
    return {
        "n_inputs": n,
        "shape": [int(s) for s in shape],
        "mean_of_mean": float(mean.mean()),
        "mean_of_std": float(std.mean()),
        "outputs": [str(out_dir / f) for f in ("mean.tif", "std.tif", "consistency.tif")],
    }
