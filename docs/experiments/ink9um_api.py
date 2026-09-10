"""Minimal, verified re-implementation of villa's flat ink_9um inference (infer.py) as a module.

Mirrors vesuvius/ink_detection/inference/infer.py for the public hybrid_3d2d ink_9um checkpoints:

  * layer selection : select_layer_indices(depth, layer_start, layer_end, output_depth=input_depth, direction)
                      -> arange(start, stop), center-cropped to input_depth (upper centre for even excess),
                      reversed for direction 'reverse'. When fewer than input_depth layers are selected the
                      FlatPatchReader places them at depth offset (input_depth - n) // 2 inside a ZERO buffer
                      (so --layer-start 5 --layer-end 21 on a 17-deep model = slices 5..20 + one zero plane at
                      index 16). TargetModel.input_pad_depth_to is None for these checkpoints, so no further pad.
  * preprocessing   : 'tifxyz_robust' = vesuvius.image_proc.intensity.normalization.normalize_robust applied
                      PER PATCH to the float32 padded (D_in, 128, 128) block (zero plane included in the
                      statistics): clip to [p1, p99], subtract median, divide by 1.4826 * MAD.
  * stitching       : stride = round(patch * (1 - overlap)) with a final boundary-aligned position; Hann
                      window (torch.hann_window periodic=False, outer product, / max, floored at 1e-3);
                      sigmoid of the logits per tile, weighted mean of PROBABILITIES (not logits),
                      prob = clip(prob_sum / weight_sum, 0, 1); infer.py then writes uint8 via truncation.
  * amp             : infer.py autocasts to fp16 only on CUDA; on CPU it runs float32 (as here).

The model is built through infer.configure_model so the architecture and weight loading are villa's own.

Usage
-----
    from ink9um_api import load_ink9um, select_layers, predict_map
    model, info = load_ink9um('work/ckpt/hybrid_3d2d-seed43/step-075000.pth', 'cpu')
    layers = select_layers(stack, 5, 21)                 # (17, H, W) uint8
    prob = predict_map(model, layers, 'cpu')             # (H, W) float32 in [0, 1]

Run as a script to reproduce the verification against work/mil_1667/mil_1667_w028.npz (see __main__).
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.ink_detection.inference import infer as villa_infer
from vesuvius.image_proc.intensity.normalization import normalize_robust


# ----------------------------------------------------------------------------- model
def load_ink9um(ckpt_path, device="cpu", amp_dtype="auto"):
    """Return (model, info). model is villa's TargetModel (eval, on device); forward (B,1,D_in,P,P) -> logits (B,1,P,P)."""
    cm = villa_infer.configure_model(argparse.Namespace(checkpoint=str(ckpt_path), amp_dtype=amp_dtype))
    device = torch.device(device)
    model = cm.model.to(device).eval()
    info = {
        "patch_size": int(cm.patch_size),
        "input_depth": int(cm.input_depth),
        "amp_dtype": cm.amp_dtype,               # torch.float16 for the public checkpoints; used only on CUDA
        "preprocessing": cm.preprocessing,       # 'tifxyz_robust'
        "input_pad_depth_to": cm.model.input_pad_depth_to,   # None for the public checkpoints
        "checkpoint": str(ckpt_path),
        "device": str(device),
    }
    return model, info


# ----------------------------------------------------------------------------- layers
def select_layer_indices(depth, layer_start, layer_end, input_depth, reverse=False):
    """Exactly infer.select_layer_indices (clamp, centre crop, optional reverse)."""
    return villa_infer.select_layer_indices(
        int(depth), layer_start=layer_start, layer_end=layer_end,
        output_depth=int(input_depth), direction="reverse" if reverse else "forward")


def select_layers(stack, layer_start, layer_end, reverse=False, input_depth=17):
    """uint8 (D_in, H, W): the slices infer.py feeds, zero-padded to input_depth like FlatPatchReader.read
    (selected block placed at depth offset (D_in - n) // 2, remaining planes zero)."""
    stack = np.asarray(stack)
    assert stack.ndim == 3 and stack.dtype == np.uint8, "stack must be uint8 (D, H, W)"
    D, H, W = stack.shape
    idx = select_layer_indices(D, layer_start, layer_end, input_depth, reverse)
    n = int(idx.size)
    out = np.zeros((int(input_depth), H, W), dtype=np.uint8)
    start = (int(input_depth) - n) // 2
    out[start:start + n] = stack[idx]
    return out


# ----------------------------------------------------------------------------- preprocessing
def preprocess(patch_uint8_DHW):
    """float32 tensor (1, D_in, H, W): infer.py 'tifxyz_robust' = per-patch normalize_robust on the float32 block.
    (The zero padding plane, if any, is part of the block and enters the percentile/median/MAD statistics,
    exactly as in FlatPatchReader.read -> FlatBlockDataset.__getitem__ -> normalize_flat_patch.)"""
    block = np.asarray(patch_uint8_DHW, dtype=np.float32)   # reader casts the raw block to float32
    block = np.ascontiguousarray(normalize_robust(block))    # percentile 1/99 clip, median, 1.4826*MAD
    return torch.from_numpy(block).unsqueeze(0)


# ----------------------------------------------------------------------------- stitching
def hann_weight(patch):
    """infer.compute_importance_map_2d(mode='hann'): outer Hann (periodic=False), / max, floor 1e-3."""
    return villa_infer.compute_importance_map_2d(patch_size=(patch, patch), mode="hann").numpy()


def sliding_positions(length, patch, stride):
    """infer._sliding_positions_1d: range(0, L-P+1, stride) plus a final boundary-aligned position."""
    return villa_infer._sliding_positions_1d(int(length), int(patch), int(stride))


@torch.no_grad()
def predict_map(model, stack_DHW_uint8, device="cpu", patch=128, overlap=0.5, batch=8, roi=None,
                amp_dtype=None, progress=False):
    """Sliding window + Hann blend exactly like infer.py (forward direction, no TTA, no mask).

    stack_DHW_uint8 : output of select_layers, uint8 (D_in, H, W). The tile grid is defined over the FULL (H, W).
    roi             : optional (y0, y1, x0, x1) in full-frame coordinates; only tiles intersecting it are run and
                      the returned map covers the roi (so it equals the crop of a full-frame run).
    amp_dtype       : autocast dtype used only on CUDA (infer.py behaviour); CPU always runs float32.
    Returns float32 (h, w) probabilities in [0, 1] (prob_sum / weight_sum, clipped) before infer.py's uint8 truncation.
    """
    device = torch.device(device)
    stack = np.asarray(stack_DHW_uint8)
    assert stack.ndim == 3 and stack.dtype == np.uint8
    D, H, W = stack.shape
    P = int(patch)
    stride = villa_infer.resolve_patch_stride(patch_size=P, overlap=float(overlap), explicit_stride=None)
    ry0, ry1, rx0, rx1 = (0, H, 0, W) if roi is None else (int(v) for v in roi)
    w2 = hann_weight(P)

    tiles = []
    for y0 in sliding_positions(H, P, stride):
        vh = min(P, H - y0)
        if y0 + vh <= ry0 or y0 >= ry1:
            continue
        for x0 in sliding_positions(W, P, stride):
            vw = min(P, W - x0)
            if x0 + vw <= rx0 or x0 >= rx1:
                continue
            tiles.append((y0, x0, vh, vw))

    acc = np.zeros((ry1 - ry0, rx1 - rx0), np.float32)
    wsum = np.zeros_like(acc)
    autocast = (torch.autocast("cuda", dtype=amp_dtype) if (device.type == "cuda" and amp_dtype is not None)
                else torch.autocast("cpu", enabled=False))
    t0 = time.time()
    for i in range(0, len(tiles), int(batch)):
        chunk = tiles[i:i + int(batch)]
        xs, keep = [], []
        for (y0, x0, vh, vw) in chunk:
            raw = np.zeros((D, P, P), np.uint8)            # reader zero-pads beyond the image (never hit here)
            raw[:, :vh, :vw] = stack[:, y0:y0 + vh, x0:x0 + vw]
            if not raw.any():                                # infer.py skips all-zero raw patches
                continue
            xs.append(preprocess(raw))
            keep.append((y0, x0, vh, vw))
        if not xs:
            continue
        xb = torch.stack(xs).to(device)
        with torch.inference_mode(), autocast:
            logits = model(xb)
            probs = villa_infer.logits_to_probabilities(logits, image_hw=(P, P)).float().cpu().numpy()[:, 0]
        for (y0, x0, vh, vw), pr in zip(keep, probs):
            iy0, iy1 = max(y0, ry0), min(y0 + vh, ry1)
            ix0, ix1 = max(x0, rx0), min(x0 + vw, rx1)
            src = (slice(iy0 - y0, iy1 - y0), slice(ix0 - x0, ix1 - x0))
            dst = (slice(iy0 - ry0, iy1 - ry0), slice(ix0 - rx0, ix1 - rx0))
            acc[dst] += pr[src] * w2[src]
            wsum[dst] += w2[src]
        if progress:
            done = min(i + len(chunk), len(tiles))
            print(f"  {done}/{len(tiles)} tiles, {time.time() - t0:.0f}s", flush=True)
    out = np.zeros_like(acc)
    np.divide(acc, wsum, out=out, where=wsum > 1e-6)
    np.clip(out, 0, 1, out=out)
    return out


def to_uint8_like_infer(prob):
    """infer.iter_probability_tiles: clip, *255, truncate to uint8."""
    return (np.clip(prob, 0, 1) * 255).astype(np.uint8)


# ----------------------------------------------------------------------------- verification
def _pool4(a):
    H, W = a.shape
    H4, W4 = (H // 4) * 4, (W // 4) * 4
    return a[:H4, :W4].reshape(H4 // 4, 4, W4 // 4, 4).mean(axis=(1, 3))


def _safe_auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(bool).ravel(); s = np.asarray(s, np.float64).ravel()
    if y.size == 0 or y.all() or not y.any():
        return None
    return float(roc_auc_score(y, s))


def _main():
    ap = argparse.ArgumentParser(description="Reproduce the Kaggle ink_9um map on the PHerc1667 test half.")
    ap.add_argument("--npz", default="work/mil_1667/mil_1667_w028.npz")
    ap.add_argument("--ckpt", default="work/ckpt/hybrid_3d2d-seed43/step-075000.pth")
    ap.add_argument("--rows", type=int, default=600, help="rows 0..rows of the test half (0 = all)")
    ap.add_argument("--layer-start", type=int, default=5)
    ap.add_argument("--layer-end", type=int, default=21)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default="", help="optional .npy path for the predicted map")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    d = np.load(args.npz)
    stack, truth, ink9um, bands = d["stack"], d["truth"], d["ink9um"], d["bands_oracle"]
    import json
    split_x = int(json.loads(str(d["meta"]))["split_x"])
    H, W = truth.shape
    rows = H if args.rows <= 0 else min(args.rows, H)
    roi = (0, rows, split_x, W)

    model, info = load_ink9um(args.ckpt, "cpu")
    print("info:", {k: str(v) for k, v in info.items()})
    layers = select_layers(stack, args.layer_start, args.layer_end, input_depth=info["input_depth"])
    print("layers:", layers.shape, "nonzero planes:", [int(layers[i].any()) for i in range(layers.shape[0])])

    t0 = time.time()
    prob = predict_map(model, layers, "cpu", patch=info["patch_size"], overlap=0.5, batch=args.batch, roi=roi,
                       progress=True)
    dt = time.time() - t0
    n_tiles = len([1 for y in sliding_positions(H, 128, 64) if y + 128 > 0 and y < rows
                   for x in sliding_positions(W, 128, 64) if x + 128 > split_x])
    print(f"predict_map: {dt:.1f}s for {n_tiles} tiles = {dt / n_tiles:.2f} s/tile")
    if args.save:
        np.save(args.save, prob)

    ours255 = prob * 255.0
    ref = ink9um[roi[0]:roi[1], roi[2]:roi[3]].astype(np.float32)
    tr = truth[roi[0]:roi[1], roi[2]:roi[3]]
    corr = float(np.corrcoef(ours255.ravel(), ref.ravel())[0, 1])
    mad = float(np.abs(ours255 - ref).mean())
    mad_u8 = float(np.abs(to_uint8_like_infer(prob).astype(np.float32) - ref).mean())
    print(f"pearson r = {corr:.5f}   mean|diff| = {mad:.3f} (float) / {mad_u8:.3f} (uint8-truncated)   "
          f"max|diff| = {np.abs(ours255 - ref).max():.1f}")

    tp = _pool4((tr >= 128).astype(np.float32)) >= 0.5
    H4 = tp.shape[0]
    row_in = np.broadcast_to((bands[:H4 * 4] == 1).reshape(H4, 4).all(axis=1)[:, None], tp.shape)
    for name, m in (("ours", prob), ("kaggle", ref / 255.0)):
        pp = _pool4(m.astype(np.float32))
        print(f"{name:7s} AUC all = {_safe_auc(tp, pp):.4f}   AUC rows = {_safe_auc(tp[row_in], pp[row_in]):.4f}   "
              f"(cells {pp.size}, row cells {int(row_in.sum())})")


if __name__ == "__main__":
    _main()
