#!/usr/bin/env python
"""
MIL vs supervised ink detection on a PHerc1667 (segment w028) surface-volume crop.

Experiment
----------
Input: 27 depth slices (9.6 um voxels, axis order z,y,x) of a flattened papyrus sheet; text rows run
along x.  Question: how much ink signal can a small 2D U-Net recover from ROW-level labels only,
compared with the same network trained on per-pixel labels and with the public ink_9um model?
Three arms share the same network, initialisation, patch sequence and evaluation:

  supervised : per-pixel BCE against the team's canon ink prediction (truth >= 128).  Upper bound.
  mil_oracle : multiple-instance learning.  The only labels are per-ROW bags (1 = text-row band,
               0 = interline band = certain negatives, -1 = ambiguous/ignored), derived from truth on
               the training half and broadcast along x into a per-pixel bag map.
  mil_free   : identical loss, but the row bags come from the public ink_9um prediction (truth-free).

MIL loss  =  w_neg   * BCE->0 on every bag==0 pixel of the batch
          +  w_pos   * BCE->1 on the top-q fraction of bag==1 pixels of each patch (skipped if empty)
          +  w_prior * hinge keeping the mean bag==1 probability of each patch inside [lo, hi]
The MIL branch never touches the truth array (the sampler is built without it for MIL arms).

Protocol: training patches lie entirely in x < split_x; per-slice normalisation statistics come from
that half; evaluation is Hann-weighted sliding-window inference (patch, stride patch/2) on x >= split_x
only, reported for the FINAL-epoch model (the per-epoch monitor AUC is logging only).  Metrics are
computed on a 4x4 mean-pooled grid (pooled truth >= 0.5 is ink): AUC over all test cells and AUC
restricted to pooled rows whose four source rows are all bands_oracle == 1 ("within rows"); the
ink_9um baseline goes through the identical Evaluator.  Augmentation: H/V flips only (no rotations).
Outputs in --out: results.json, <arm>_prob_test.npy (float16), <arm>_prob_test.png, panel_test.png.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ARMS = ("supervised", "mil_oracle", "mil_free")
FINGERPRINT_KEYS = ("smoke", "patch", "epochs", "steps_per_epoch", "batch_size", "lr", "seed", "topq",
                    "w_neg", "w_pos", "w_prior", "prior_lo", "prior_hi", "base_ch", "split_x", "stack_shape")


# ----------------------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="mil_1667_w028.npz")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--arm", default="all", choices=ARMS + ("all",))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topq", type=float, default=0.3, help="top fraction of bag==1 pixels pushed to 1, in (0,1]")
    p.add_argument("--w-neg", type=float, default=1.0)
    p.add_argument("--w-pos", type=float, default=1.0)
    p.add_argument("--w-prior", type=float, default=0.5)
    p.add_argument("--prior-lo", type=float, default=0.2)
    p.add_argument("--prior-hi", type=float, default=0.6)
    p.add_argument("--base-ch", type=int, default=32, help="U-Net width at the first level")
    p.add_argument("--infer-batch", type=int, default=32)
    p.add_argument("--monitor-patches", type=int, default=8, help="fixed test patches for the per-epoch AUC monitor")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--smoke", action="store_true", help="tiny crop, 2 epochs x 5 steps, patch 64, batch 4")
    args = p.parse_args()
    if args.smoke:
        args.patch, args.batch_size, args.epochs, args.steps_per_epoch = 64, 4, 2, 5
        args.monitor_patches = min(args.monitor_patches, 4)
    if args.patch % 8:
        p.error("--patch must be a multiple of 8 (three 2x poolings)")
    if not 0 < args.topq <= 1:
        p.error("--topq must be in (0, 1]")
    if not 0 <= args.prior_lo <= args.prior_hi <= 1:
        p.error("--prior-lo/--prior-hi must satisfy 0 <= lo <= hi <= 1")
    return args


def pick_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {name} requested but CUDA is not available")
    return torch.device(name)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------------- data
def load_data(path, smoke):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    stack, truth, ink9um = d["stack"], d["truth"], d["ink9um"]
    bands_oracle, bands_free = d["bands_oracle"].astype(np.int8), d["bands_free"].astype(np.int8)
    split_x = int(meta["split_x"])
    if smoke:
        # NOTE: the band labels were derived on the full training half (x < split_x of the full crop),
        # which overlaps the smoke test half x in [300, 600): smoke metrics are NOT protocol-clean.
        stack, truth, ink9um = stack[:, :400, :600], truth[:400, :600], ink9um[:400, :600]
        bands_oracle, bands_free, split_x = bands_oracle[:400], bands_free[:400], 300
    C, H, W = stack.shape
    assert truth.shape == (H, W) == ink9um.shape and bands_oracle.shape == (H,) == bands_free.shape
    assert 0 < split_x < W
    return {"stack": np.ascontiguousarray(stack), "truth": np.ascontiguousarray(truth),
            "ink9um": np.ascontiguousarray(ink9um), "bands_oracle": np.ascontiguousarray(bands_oracle),
            "bands_free": np.ascontiguousarray(bands_free), "split_x": split_x, "meta": meta}


def norm_stats(stack, split_x):
    """Per-slice mean/std over the TRAINING half only (x < split_x); one slice at a time (small transient)."""
    C = stack.shape[0]
    mean, std = np.empty(C, np.float32), np.empty(C, np.float32)
    for c in range(C):
        s = stack[c, :, :split_x].astype(np.float32)
        mean[c] = s.mean()
        std[c] = max(float(s.std(dtype=np.float64)), 1e-3)
    return mean, std


def normalise(x_u8, mean_t, std_t, device):
    x = torch.from_numpy(np.ascontiguousarray(x_u8)).to(device, non_blocking=True).float()
    return (x - mean_t) / std_t


class PatchSampler:
    """Random training patches strictly inside x < split_x, with H/V flips (no rotations).

    Returns uint8 stack patches, a per-pixel bag map (per-row band label broadcast along x; -1
    everywhere when bands is None) and, ONLY when truth_bin was given (supervised arm), the truth
    patch.  MIL arms construct the sampler with truth_bin=None, so truth cannot reach the MIL loss.
    """

    def __init__(self, stack, truth_bin, bands, split_x, patch, seed):
        H = stack.shape[1]
        if split_x < patch or H < patch:
            raise ValueError(f"patch {patch} does not fit in the training half ({H} x {split_x})")
        self.stack, self.truth_bin, self.bands = stack, truth_bin, bands
        self.split_x, self.patch = split_x, patch
        self.rng = np.random.default_rng(seed)

    def sample(self, B):
        C, H, _ = self.stack.shape
        p, with_truth = self.patch, self.truth_bin is not None
        ys = self.rng.integers(0, H - p + 1, size=B)
        xs = self.rng.integers(0, self.split_x - p + 1, size=B)  # x0 + p <= split_x
        flips = self.rng.random((B, 2)) < 0.5
        x = np.empty((B, C, p, p), np.uint8)
        bag = np.empty((B, p, p), np.int8)
        t = np.empty((B, p, p), np.uint8) if with_truth else None
        for i, (y0, x0) in enumerate(zip(ys, xs)):
            xi = self.stack[:, y0:y0 + p, x0:x0 + p]
            bi = np.full((p, p), -1, np.int8) if self.bands is None else \
                np.broadcast_to(self.bands[y0:y0 + p, None], (p, p))
            ti = self.truth_bin[y0:y0 + p, x0:x0 + p] if with_truth else None
            if flips[i, 0]:  # horizontal flip (along x): rows stay rows
                xi, bi = xi[:, :, ::-1], bi[:, ::-1]
                ti = None if ti is None else ti[:, ::-1]
            if flips[i, 1]:  # vertical flip (along y)
                xi, bi = xi[:, ::-1, :], bi[::-1, :]
                ti = None if ti is None else ti[::-1, :]
            x[i], bag[i] = xi, bi
            if with_truth:
                t[i] = ti
        return x, bag, t


# ----------------------------------------------------------------------------- model
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        g = 8 if cout % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    """Plain 2D U-Net: 27 slices as input channels, 3 down/up levels, GroupNorm, one logit map."""

    def __init__(self, in_ch=27, base=32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1, self.enc2, self.enc3 = ConvBlock(in_ch, c1), ConvBlock(c1, c2), ConvBlock(c2, c3)
        self.bott = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.up3, self.dec3 = nn.ConvTranspose2d(c4, c3, 2, stride=2), ConvBlock(2 * c3, c3)
        self.up2, self.dec2 = nn.ConvTranspose2d(c3, c2, 2, stride=2), ConvBlock(2 * c2, c2)
        self.up1, self.dec1 = nn.ConvTranspose2d(c2, c1, 2, stride=2), ConvBlock(2 * c1, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1).squeeze(1)  # (B, H, W) logits


def autocast_ctx(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# ----------------------------------------------------------------------------- losses
def supervised_loss(logits, target):
    """Per-pixel BCE-with-logits against the binary truth patch (float 0/1)."""
    loss = F.binary_cross_entropy_with_logits(logits, target)
    return loss, {"bce": loss}


def mil_loss(logits, bag, topq, w_neg, w_pos, w_prior, lo, hi):
    """Bag-level loss.  Sees only the logits and the per-pixel bag map (1 / 0 / -1), never truth.

    Returns (total, terms).  A term whose pixel set is empty is skipped (per patch for the positive
    terms); if no term is active `terms` is empty and the caller must skip the optimisation step.
    """
    terms = {}
    total = logits.new_zeros(())
    neg = bag == 0
    if bool(neg.any()):  # (a) interline pixels are certain negatives
        ln = logits[neg]
        l_neg = F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
        terms["neg"] = l_neg
        total = total + w_neg * l_neg
    pos_terms, prior_terms = [], []
    for i in range(logits.shape[0]):  # per patch: positive bag may be empty -> skip that patch
        pm = bag[i] == 1
        n = int(pm.sum())
        if n == 0:
            continue
        lp = logits[i][pm]
        k = min(n, max(1, int(math.ceil(topq * n))))
        top = torch.topk(lp, k).values  # top-q of probabilities == top-q of logits (monotone)
        pos_terms.append(F.binary_cross_entropy_with_logits(top, torch.ones_like(top)))  # (b)
        p_mean = torch.sigmoid(lp).mean()  # (c) soft prior on the mean bag probability
        prior_terms.append(F.relu(lo - p_mean) + F.relu(p_mean - hi))
    if pos_terms:
        l_pos, l_prior = torch.stack(pos_terms).mean(), torch.stack(prior_terms).mean()
        terms["pos"], terms["prior"] = l_pos, l_prior
        total = total + w_pos * l_pos + w_prior * l_prior
    return total, terms


# ----------------------------------------------------------------------------- evaluation
def safe_auc(y, s):
    y = np.asarray(y).astype(bool).ravel()
    s = np.asarray(s, np.float64).ravel()
    if y.size == 0 or y.all() or not y.any():
        return None
    return float(roc_auc_score(y, s))


def pool4(a):
    H, W = a.shape
    H4, W4 = (H // 4) * 4, (W // 4) * 4
    return a[:H4, :W4].reshape(H4 // 4, 4, W4 // 4, 4).mean(axis=(1, 3))


class Evaluator:
    """Identical metric for every arm and the baseline: 4x4 pooled AUC on the test half.

    'within rows' = pooled rows whose four source rows are all bands_oracle == 1 (strict rule).
    """
    row_rule = "all 4 source rows bands_oracle==1"

    def __init__(self, truth, bands_oracle, split_x):
        truth_t = (truth[:, split_x:] >= 128).astype(np.float32)
        self.truth_pooled = pool4(truth_t) >= 0.5
        H4 = self.truth_pooled.shape[0]
        row_in = (bands_oracle[:H4 * 4] == 1).reshape(H4, 4).all(axis=1)
        self.row_mask = np.broadcast_to(row_in[:, None], self.truth_pooled.shape)

    def __call__(self, prob_test):
        pp = pool4(np.asarray(prob_test, np.float32))
        assert pp.shape == self.truth_pooled.shape
        return {"auc_all": safe_auc(self.truth_pooled, pp),
                "auc_rows": safe_auc(self.truth_pooled[self.row_mask], pp[self.row_mask]),
                "n_cells": int(pp.size), "n_row_cells": int(self.row_mask.sum()),
                "ink_frac_cells": float(self.truth_pooled.mean())}


class Monitor:
    """Fixed subset of test patches for a cheap per-epoch AUC (never used for model selection)."""

    def __init__(self, stack, truth_bin, split_x, patch, n_patches, seed):
        rng = np.random.default_rng(seed + 1)
        C, H, W = stack.shape
        self.x = self.t = None
        if n_patches <= 0 or W - split_x < patch:
            return
        ys = rng.integers(0, H - patch + 1, size=n_patches)
        xs = rng.integers(split_x, W - patch + 1, size=n_patches)
        self.x = np.stack([stack[:, y:y + patch, x:x + patch] for y, x in zip(ys, xs)])
        self.t = np.stack([truth_bin[y:y + patch, x:x + patch] for y, x in zip(ys, xs)]).ravel()

    @torch.no_grad()
    def auc(self, model, mean_t, std_t, device, infer_batch):
        if self.x is None:
            return None
        model.eval()
        probs = []
        for i in range(0, len(self.x), infer_batch):
            with autocast_ctx(device):
                logits = model(normalise(self.x[i:i + infer_batch], mean_t, std_t, device))
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        return safe_auc(self.t, np.concatenate(probs).ravel())


def tile_starts(n, p, s):
    assert n >= p, f"window {p} larger than extent {n}"
    starts = list(range(0, n - p + 1, s))
    if starts[-1] != n - p:
        starts.append(n - p)
    return starts


@torch.no_grad()
def predict_test_half(model, stack, split_x, mean_t, std_t, patch, device, infer_batch):
    """Sliding window (stride patch/2) over x >= split_x; Hann-weighted average of logits -> probabilities."""
    model.eval()
    C, H, W = stack.shape
    p, s = patch, patch // 2
    Wt = W - split_x
    w1 = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(p) + 0.5) / p)  # half-sample-shifted Hann, > 0 everywhere
    w2 = np.clip(np.outer(w1, w1), 1e-2, None).astype(np.float32)
    tiles = [(y, x) for y in tile_starts(H, p, s) for x in tile_starts(Wt, p, s)]
    acc = np.zeros((H, Wt), np.float32)
    wsum = np.zeros((H, Wt), np.float32)
    for i in range(0, len(tiles), infer_batch):
        chunk = tiles[i:i + infer_batch]
        xb = np.stack([stack[:, y:y + p, split_x + x:split_x + x + p] for y, x in chunk])
        with autocast_ctx(device):
            logits = model(normalise(xb, mean_t, std_t, device))
        for (y, x), lg in zip(chunk, logits.float().cpu().numpy()):
            acc[y:y + p, x:x + p] += lg * w2
            wsum[y:y + p, x:x + p] += w2
    assert wsum.min() > 0
    return expit(acc / wsum).astype(np.float32)


# ----------------------------------------------------------------------------- training
def train_and_eval_arm(arm, data, mean, std, args, device, evaluator, monitor):
    stack, split_x = data["stack"], data["split_x"]
    C = stack.shape[0]
    supervised = arm == "supervised"
    truth_bin = (data["truth"] >= 128).astype(np.uint8) if supervised else None  # MIL arms: no truth at all
    bands = {"supervised": None, "mil_oracle": data["bands_oracle"], "mil_free": data["bands_free"]}[arm]

    seed_all(args.seed)  # identical initialisation across arms
    model = UNet2D(C, args.base_ch).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    sampler = PatchSampler(stack, truth_bin, bands, split_x, args.patch, args.seed)  # same patch sequence per arm
    mean_t = torch.from_numpy(mean).view(1, C, 1, 1).to(device)
    std_t = torch.from_numpy(std).view(1, C, 1, 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=args.lr * 0.01)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"\n=== arm {arm}: {n_params:,} params | device {device} | amp {use_amp} | "
          f"{args.epochs} epochs x {args.steps_per_epoch} steps x batch {args.batch_size} of {args.patch}px")

    log = []
    t_train0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sums, n_ok, n_skip = defaultdict(float), 0, 0
        for _ in range(args.steps_per_epoch):
            x_u8, bag_i8, t_u8 = sampler.sample(args.batch_size)
            with autocast_ctx(device):
                logits = model(normalise(x_u8, mean_t, std_t, device))
            logits = logits.float()  # losses in fp32
            if supervised:
                loss, terms = supervised_loss(logits, torch.from_numpy(t_u8).to(device).float())
            else:
                loss, terms = mil_loss(logits, torch.from_numpy(bag_i8).to(device), args.topq,
                                       args.w_neg, args.w_pos, args.w_prior, args.prior_lo, args.prior_hi)
            if terms:
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                sums["loss"] += float(loss.detach())
                for k, v in terms.items():
                    sums[k] += float(v.detach())
                n_ok += 1
            else:
                n_skip += 1  # whole batch had no usable bag pixels (all rows ambiguous)
            sched.step()  # tied to the step budget, also on skipped batches
        means = {k: sums[k] / max(n_ok, 1) for k in ("loss", "bce", "neg", "pos", "prior") if k in sums}
        mon = monitor.auc(model, mean_t, std_t, device, args.infer_batch)
        lr_now = opt.param_groups[0]["lr"]
        term_str = " ".join(f"{k}={v:.4f}" for k, v in means.items() if k != "loss")
        print(f"[{arm}] epoch {ep:3d}/{args.epochs} loss={means.get('loss', float('nan')):.4f} ({term_str}) "
              f"monitor_auc={fmt(mon)} lr={lr_now:.2e} skipped={n_skip} {time.time() - t0:.1f}s")
        log.append({"epoch": ep, "loss": means.get("loss"), "terms": {k: v for k, v in means.items() if k != "loss"},
                    "monitor_auc": mon, "lr": lr_now, "skipped_steps": n_skip, "seconds": time.time() - t0})
    train_s = time.time() - t_train0

    t0 = time.time()
    prob = predict_test_half(model, stack, split_x, mean_t, std_t, args.patch, device, args.infer_batch)
    prob16 = prob.astype(np.float16)  # the saved artefact; metrics are computed on exactly this array
    eval_s = time.time() - t0
    metrics = evaluator(prob16.astype(np.float32))
    print(f"[{arm}] test-half AUC all={fmt(metrics['auc_all'])} within-rows={fmt(metrics['auc_rows'])} "
          f"(train {train_s:.1f}s, inference {eval_s:.1f}s)")
    return {**metrics, "params": n_params, "train_seconds": train_s, "eval_seconds": eval_s, "epochs": log}, prob16


# ----------------------------------------------------------------------------- outputs
def fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def to_u8(a):
    return (np.clip(np.asarray(a, np.float32), 0, 1) * 255 + 0.5).astype(np.uint8)


def save_panel(images, labels, path, width=700):
    resample = getattr(getattr(Image, "Resampling", Image), "BOX")
    strip, gap, tiles = 22, 12, []
    for img, lab in zip(images, labels):
        im = Image.fromarray(to_u8(img))
        im = im.resize((width, max(1, round(im.height * width / im.width))), resample)
        canvas = Image.new("L", (width, im.height + strip), 255)
        canvas.paste(im, (0, strip))
        ImageDraw.Draw(canvas).text((6, 4), lab, fill=0)
        tiles.append(canvas)
    out = Image.new("L", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), max(t.height for t in tiles)), 128)
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + gap
    out.save(path)


def markdown_table(baseline, arm_results):
    lines = ["| arm | AUC all | AUC within rows |", "|---|---|---|",
             f"| ink_9um baseline | {fmt(baseline['auc_all'])} | {fmt(baseline['auc_rows'])} |"]
    for arm in ARMS:
        if arm in arm_results:
            lines.append(f"| {arm} | {fmt(arm_results[arm]['auc_all'])} | {fmt(arm_results[arm]['auc_rows'])} |")
    return "\n".join(lines)


def load_merged_results(path, fingerprint):
    """Existing results.json (arms run in separate invocations) minus arms trained under another config."""
    results = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                results = json.load(f)
        except (OSError, ValueError):
            results = {}
    kept = {}
    for arm, r in results.get("arms", {}).items():
        if r.get("fingerprint") == fingerprint:
            kept[arm] = r
        else:
            print(f"warning: dropping stale '{arm}' entry from {path} (different config)")
    results["arms"] = kept
    return results


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)  # no-op under IPython %run
    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step")  # AMP may skip a step
    seed_all(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    t_all0 = time.time()

    data = load_data(args.data, args.smoke)
    stack, truth, ink9um, split_x = data["stack"], data["truth"], data["ink9um"], data["split_x"]
    C, H, W = stack.shape
    print(f"data {args.data}: stack {stack.shape}, split_x={split_x} (train x<{split_x}, test x>={split_x}), "
          f"device={device}, torch={torch.__version__}, smoke={args.smoke}")
    if args.smoke:
        print("smoke: bands were derived on the full training half, which overlaps the smoke test half; "
              "metrics are not protocol-clean and carry no signal (10 optimiser steps)")
    mean, std = norm_stats(stack, split_x)
    evaluator = Evaluator(truth, data["bands_oracle"], split_x)
    monitor = Monitor(stack, (truth >= 128).astype(np.uint8), split_x, args.patch, args.monitor_patches, args.seed)

    baseline = evaluator(ink9um[:, split_x:].astype(np.float32) / 255.0)
    print(f"ink_9um baseline: AUC all={fmt(baseline['auc_all'])} within-rows={fmt(baseline['auc_rows'])} "
          f"({baseline['n_row_cells']}/{baseline['n_cells']} pooled cells inside oracle rows)")

    config = {**vars(args), "device": str(device), "split_x": split_x, "stack_shape": [C, H, W],
              "stride": args.patch // 2, "row_rule": Evaluator.row_rule, "protocol_clean": not args.smoke,
              "cudnn_deterministic": True, "norm_mean_per_slice": mean.tolist(), "norm_std_per_slice": std.tolist(),
              "torch": torch.__version__}
    fingerprint = {k: config[k] for k in FINGERPRINT_KEYS}
    results_path = os.path.join(args.out, "results.json")
    results = load_merged_results(results_path, fingerprint)
    results["config"], results["baseline_ink9um"] = config, baseline

    arms = ARMS if args.arm == "all" else (args.arm,)
    for arm in arms:
        res, prob16 = train_and_eval_arm(arm, data, mean, std, args, device, evaluator, monitor)
        results["arms"][arm] = {**res, "fingerprint": fingerprint}
        results["params"] = res["params"]
        np.save(os.path.join(args.out, f"{arm}_prob_test.npy"), prob16)
        Image.fromarray(to_u8(prob16)).save(os.path.join(args.out, f"{arm}_prob_test.png"))
        results["table_markdown"] = markdown_table(baseline, results["arms"])
        results["total_seconds"] = time.time() - t_all0
        with open(results_path, "w") as f:  # rewritten after every arm so a timeout leaves partial results
            json.dump(results, f, indent=2)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # panel and table over every arm present (this run + compatible earlier runs in --out)
    present = [a for a in ARMS if a in results["arms"] and os.path.exists(os.path.join(args.out, f"{a}_prob_test.npy"))]
    maps = [np.load(os.path.join(args.out, f"{a}_prob_test.npy")).astype(np.float32) for a in present]
    save_panel([truth[:, split_x:] / 255.0, ink9um[:, split_x:] / 255.0] + maps,
               ["truth (test half)", "ink_9um (test half)"] + present, os.path.join(args.out, "panel_test.png"))
    table = markdown_table(baseline, results["arms"])
    results["table_markdown"], results["total_seconds"] = table, time.time() - t_all0
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nparams={results.get('params')}  pooled 4x4  test half x>={split_x}\n{table}")
    print(f"\nwrote {results_path} ({results['total_seconds']:.1f}s total)")


if __name__ == "__main__":
    main()
