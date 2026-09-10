#!/usr/bin/env python
"""
Geometry-guided fine-tuning of the public ink_9um model on a PHerc1667 crop (final script).

EXPERIMENT  mil_1667_w028.npz holds 27 depth slices (9.6 um, z,y,x) of a PHerc1667 w028 sheet crop (text rows
  along x), the team's published prediction as `truth`, the Kaggle output of this checkpoint as `ink9um`, and
  per-row band labels (1 text row / 0 interline / -1 ambiguous) derived on the TRAINING half only.  Question:
  can ROW-LEVEL geometry alone, with no per-pixel ink label, fine-tune villa's hybrid_3d2d ink_9um checkpoint
  into a better crop-specific detector?

ARMS (--arm; each starts from the pristine checkpoint with identical data, seed and evaluation)
  baseline    no training.  Stage-0 control: it must reproduce the published 0.7714 / 0.7623 within 0.005
              (checked automatically outside --smoke) and it is the reference for every delta.
  mil_free    MIL fine-tuning with bands_free (bands derived from the public map: truth-free).  MAIN ARM.
  mil_oracle  same loss with bands_oracle (bands derived from truth on the training half).
  supervised  per-pixel BCE against truth >= 128.  Upper bound, not a legitimate method.

TRUTH-FREEDOM OF mil_free  Nothing that touches the mil_free weights or decides when they stop moving is
  derived from `truth`: the sampler is built with truth_bin=None, the loss sees only bag labels, and the
  guards below read bands_FREE and the frozen public checkpoint (the "teacher") only.  Truth appears in the
  mil_free arm in exactly one place, the final evaluation on the test half, plus one clearly marked
  `auc_truth_diag` curve entry that no guard, no stopping rule and no selection path reads.

LOSS  MIL, exactly mil_train_1667.py's conventions (the MIL samplers are built without truth, so no truth
  array is reachable from those arms): w_neg * BCE->0 over the batch's bag==0 pixels + w_pos * BCE->pos_target
  over the top-q fraction of each patch's bag==1 logits + w_prior * hinge on each patch's mean bag==1
  probability inside [prior_lo, prior_hi].  --pos-target 1.0 gives bit-exact parity with mil_train_1667.py;
  the default 0.9 protects the checkpoint's deliberate under-confidence (its BCE was smoothed to 0.25/0.75)
  at the cost of an irreducible floor H(0.9) = 0.325 in the logged `pos` term.

DRIFT SAFEGUARDS (task U2).  The cheapest minimiser of the MIL loss is a horizontal row-stripe detector, so
  both safeguards are implemented, selected with --anchor {none,l2sp,distill,both} (default both):
  l2sp     per-group lambda/2 * ||theta - theta0||^2 toward the pre-trained weights (--w-anchor), replacing
           weight decay, PLUS a hard per-group trust region (--no-trust-region to disable): after every step
           a group whose relative drift exceeds its U2 threshold (2 % stem/encoder, 10 % decoder/head) is
           projected back onto that ball.  The projection is what binds -- under AdamW the L2-SP gradient at
           the 2 % encoder threshold is ~15x smaller than the task gradient, so the penalty alone is not a
           constraint.
  distill  self-KD from a frozen deepcopy: BCE(student_logits, sigmoid(teacher_logits)), weight 1.0 averaged
           over bag<=0 pixels (interline + ambiguous) and --w-kd-unsel over the bag==1 pixels the top-q term
           did NOT select, as two separate means so the region weights do not track batch composition.  It
           never argues with the MIL positive evidence but forbids the flat stripe solution.  Off for the
           supervised arm (per-pixel evidence everywhere).

GUARDS / STOPPING (class Guard).  This script defends a 0.77-AUC checkpoint against a weak self-supervised
  loss, so the stopping rule has to stop a run that is really degrading without stopping a healthy one on
  monitor noise.  Every enforced signal is truth-free and measured rather than guessed:
  weight space   while the projection is on, the drift VALUE cannot raise an alarm (the projection pins it
                 at DRIFT_MAX), so the alarm is the projection's ACTIVITY: a violation is raised when the
                 trust region clipped on more than --max-clip-rate of the steps in the window.  With the
                 projection off (--no-trust-region, or --anchor none/distill) the raw per-group drift is
                 instead checked every --drift-every steps, not only at curve points.
  agreement      teacher-vs-student ranking AUC over the INTERLINE pixels of the held-out monitor tiles
                 (teacher labels at its own threshold): the letter-level information the MIL loss cannot see.
                 Floor --min-agreement.
  stripe         row-vs-interline AUC of the row-mean probabilities relative to the same statistic for the
                 teacher: a rise (--max-stripe-delta) is the degenerate stripe detector, a fall
                 (--max-stripe-drop) is the loss of row contrast.
  noise band     a threshold crossing is only a VIOLATION when it exceeds the monitor's own spread: the
                 statistic is bootstrapped over the monitor's tiles (--guard-boot resamples) and the crossing
                 must be larger than --guard-sigma of those standard deviations.  A crossing inside the band
                 is printed as a note, and it still blocks the clean snapshot, but it never stops the arm.
  hysteresis     a violation must repeat at --guard-patience consecutive curve points.
  resolution     the two monitors do NOT have the same power, and the bootstrap says so.  Measured on this
                 crop at --monitor-tiles 6: sigma(agreement) = 0.003-0.042 and sigma(stripe_delta) =
                 0.07-0.28, i.e. the stripe score's 2-sigma band (+-0.15..0.55) is far wider than the
                 thresholds it is compared against, so it can only catch a catastrophic swing.  Repeating
                 the same 400-step run at --monitor-tiles 16 halves the stripe spread (0.14 -> 0.07) but
                 leaves the agreement spread unchanged (2 sigma = 0.057 vs 0.056 at step 100), because that
                 one is dominated by real tile-to-tile heterogeneity of the student, not by how many tiles
                 were drawn -- so more tiles is not a way to sharpen the agreement guard, and tightening its
                 threshold would only manufacture false alarms.  Both sigmas are printed and stored at every
                 curve point, so the resolution of a run's guards is always visible in its own log.
  action         the first enforced verdict is NOT a stop: the weights go back to the last snapshot that
                 crossed nothing (verified after restore, and replaced by the pristine weights if it still
                 crosses), the optimiser state is dropped and every lr is multiplied by --guard-lr-factor.
                 Only after --guard-lr-drops such reductions does the arm stop.  --no-rollback turns all
                 enforcement off and leaves the monitors as pure logging.
  calibration    the two enforced thresholds are placed between two MEASURED regimes, not guessed:
                 clip rate is 0 % at every curve point of the default 400-step decoder run and 100 % at
                 every curve point of an --lr 0.5 run (the alarm sits at 25 %, i.e. mid-range with a 4x
                 margin on both sides); interline agreement wobbles 0.90-0.99 around its 0.95 floor on the
                 default run, every crossing inside 2 sigma, while the --lr 0.5 runs read 0.20-0.88, i.e.
                 3-40 sigma below it.
  The guards run for EVERY fine-tuning arm (a supervised arm that destroys interline ranking is degrading
  too), with the arm's OWN bands: bands_free for mil_free, bands_oracle for mil_oracle / supervised.

LIMITATION  passing the guards is NOT evidence that an arm helped.  The default 400-step mil_free run raises
  no violation at all (400/400 steps) and still finishes 0.044 AUC BELOW the pristine checkpoint on the test
  half: the truth-free monitors resolve a collapse, not a slow leak, and --monitor-tiles 16 does not change
  that (verified).  The verdict on an arm is the test-half table, never the absence of a guard event.

INPUT / STITCHING  villa's infer.py through ink9um_api.py (verified r = 0.99998 against the Kaggle map): raw
  uint8 layers 5..20 in a 17-deep zero buffer, normalize_robust per 128x128 patch, Hann blend of
  PROBABILITIES at stride 64 on the FULL-frame tile grid.  Augmentation: H/V flips of image, bag map and
  truth patch together (no rotation: the bag labels are row-directional).

PROTOCOL / METRICS  training patches lie entirely in x < split_x - holdout; the FINAL model of each arm is
  scored on x >= split_x on the 4x4 mean-pooled grid (pooled truth >= 0.5 is ink): AUC over all cells, AUC
  within pooled rows whose 4 source rows are all bands_oracle == 1, both repeated on the inner test half
  x >= split_x + patch (control for tile context spilling across the split), the row-vs-interline AUC (a
  stripe detector maxes it out) and the map mean/std.  There is NO model selection: the reported model of an
  arm is its last step that passed the guards (results.json config records selection = "none").  The curve
  logs a truth-free held-out AUC against the public map (`auc_proxy`) and, as a diagnostic only, the same
  AUC against truth (`auc_truth_diag`, printed with a *).  Outputs in --out: results.json (merged across
  invocations by a per-arm fingerprint), <arm>_prob_test.npy (float16), <arm>_prob_test.png, panel_test.png,
  and with --save-ckpt a villa-loadable {'model','config','step'} .pth (skipped for an arm that was rolled
  back to step 0: those weights are the input checkpoint, not a fine-tuned model).

CHANGES vs the first release (all deliberate, all from the verification report)
  * monitors are built per arm and read the arm's own bands (mil_free no longer consumes bands_oracle);
  * curve keys renamed: monitor 'auc' -> 'auc_proxy' (truth-free) + 'auc_truth_diag' (diagnostic);
  * the drift alarm is trust-region clip activity, not the pinned drift value; guards apply to every arm;
  * new flags --guard-patience/--guard-sigma/--guard-boot/--guard-lr-drops/--guard-lr-factor/--max-clip-rate
    /--max-stripe-drop/--drift-every, and --curve-every now defaults to 50;
  * the results table carries a `steps / guard` column, and results.json now separates `steps_run` (steps
    the optimiser actually took) from `weights_from_step` (the step the SCORED weights come from, 0 =
    pristine); the saved checkpoint is named by the latter and is skipped entirely when it is 0;
  * TRAIN_KEYS fingerprints the guard settings, so results.json entries written by the previous version are
    evicted rather than merged.

KAGGLE  see docs/experiments/KAGGLE_mil_finetune.md for the verified T4 cell sequence.
"""
from __future__ import annotations

import argparse, contextlib, copy, json, math, os, sys, time  # noqa: E401
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:  # ink9um_api imports villa; villa itself is only a path entry, but its runtime deps must be installed
    from ink9um_api import load_ink9um, preprocess, predict_map, select_layers  # noqa: E402
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(f"cannot import ink9um_api ({e}).  Copy ink9um_api.py next to this script, put villa's "
                     f"vesuvius/src on PYTHONPATH and install its runtime deps:  pip install -q pynrrd "
                     f"donfig zarr numcodecs nest_asyncio timm einops  (timm pulls torchvision, "
                     f"huggingface_hub and safetensors; see docs/experiments/KAGGLE_mil_finetune.md)")

ARMS = ("baseline", "mil_free", "mil_oracle", "supervised")
MIL_ARMS = ("mil_free", "mil_oracle")
TERM_ORDER = ("loss", "bce", "neg", "pos", "prior", "kd", "l2sp")
# results.json merges arms across invocations only when their fingerprint matches.  The baseline arm does no
# training, so its fingerprint holds the data / inference keys only and survives hyper-parameter sweeps.
BASE_KEYS = ("smoke", "seed", "split_x", "stack_shape", "layer_start", "layer_end", "overlap", "patch",
             "ckpt_id", "data_id", "device", "infer_batch", "deterministic")
# everything that can change the trained weights, the guard verdict or the step the arm stops at
TRAIN_KEYS = ("train", "steps", "batch_size", "lr", "topq", "pos_target", "w_neg", "w_pos", "w_prior",
              "prior_lo", "prior_hi", "anchor", "w_anchor", "w_kd", "w_kd_unsel", "trust_region_active",
              "train_x_hi", "holdout_w", "monitor_tiles", "curve_every", "rollback", "min_agreement",
              "max_stripe_delta", "max_stripe_drop", "max_clip_rate", "drift_every", "guard_patience",
              "guard_sigma", "guard_boot", "guard_lr_drops", "guard_lr_factor")

# group -> (parameter-name prefixes relative to cm.model, lr multiplier (head = 1.0), L2-SP lambda).
_E, _D = "model.network.shared_encoder.", "model.network.shared_decoder."
_dec = lambda *k: tuple(f"{_D}{w}.{i}." for i in k for w in ("transpconvs", "stages"))
GROUPS = OrderedDict([
    ("stem_3d",         (("model.depth_fusion.",), 0.1, 1.0)),
    ("encoder_shallow", ((f"{_E}stem.",) + tuple(f"{_E}stages.{i}." for i in (0, 1, 2)), 0.1, 1.0)),
    ("encoder_deep",    (tuple(f"{_E}stages.{i}." for i in (3, 4, 5)), 0.03, 1.0)),
    ("decoder_deep",    (_dec(0, 1), 0.25, 1e-1)),
    ("decoder_mid",     (_dec(2), 0.25, 1e-1)),
    ("decoder_fine",    (_dec(3, 4), 0.5, 1e-2)),
    ("head",            (("model.network.task_heads.ink.",), 1.0, 1e-2)),
])
DRIFT_MAX = {"stem_3d": 0.02, "encoder_shallow": 0.02, "encoder_deep": 0.02,
             "decoder_deep": 0.10, "decoder_mid": 0.10, "decoder_fine": 0.10, "head": 0.10}
TRAIN_SETS = {"head": ("decoder_fine", "head"),
              "decoder": ("decoder_deep", "decoder_mid", "decoder_fine", "head"),
              "all": tuple(GROUPS)}
LR_DEFAULT = {"head": 1e-4, "decoder": 2e-4, "all": 1e-4}   # base = head lr; group lr = base * multiplier

# ----------------------------------------------------------------------------- CLI / setup
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="mil_1667_w028.npz")
    p.add_argument("--ckpt", required=True, help="pristine ink_9um checkpoint (step-075000.pth)")
    p.add_argument("--out", required=True)
    p.add_argument("--arm", default="all", choices=ARMS + ("all",))
    p.add_argument("--train", default="decoder", choices=tuple(TRAIN_SETS), help="trainable groups (U2 stages)")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None, help=f"HEAD lr (others scaled by group); default {LR_DEFAULT}")
    p.add_argument("--anchor", default="both", choices=("none", "l2sp", "distill", "both"),
                   help="drift safeguard: L2-SP + trust region, teacher self-distillation, both (default) or none")
    p.add_argument("--w-anchor", type=float, default=1.0, help="scale of the per-group L2-SP penalty")
    p.add_argument("--w-kd", type=float, default=1.0, help="scale of the self-distillation term")
    p.add_argument("--w-kd-unsel", type=float, default=0.2, help="KD weight on unselected text-row pixels")
    p.add_argument("--no-trust-region", dest="trust_region", action="store_false",
                   help="disable the hard per-group drift projection (L2-SP alone barely binds under AdamW)")
    p.add_argument("--no-rollback", dest="rollback", action="store_false",
                   help="never act on a guard verdict: monitors and violations are logged only")
    # --- guards (all truth-free; see the GUARDS section of the module docstring)
    p.add_argument("--min-agreement", type=float, default=0.95, help="teacher/student interline agreement floor")
    p.add_argument("--max-stripe-delta", type=float, default=0.05, help="max RISE of the stripe score vs teacher")
    p.add_argument("--max-stripe-drop", type=float, default=0.25, help="max FALL of the stripe score vs teacher")
    p.add_argument("--max-clip-rate", type=float, default=0.25,
                   help="max fraction of steps in a window on which the trust region may clip (drift alarm)")
    p.add_argument("--drift-every", type=int, default=1,
                   help="steps between raw drift checks when the trust-region projection is OFF")
    p.add_argument("--guard-patience", type=int, default=2,
                   help="consecutive violating curve points before the guard acts")
    p.add_argument("--guard-sigma", type=float, default=2.0,
                   help="a monitor crossing counts only if it exceeds this many bootstrap sigmas of the monitor")
    p.add_argument("--guard-boot", type=int, default=128, help="bootstrap resamples of the monitor tiles")
    p.add_argument("--guard-lr-drops", type=int, default=1,
                   help="lr reductions (x --guard-lr-factor) tried, after a rollback, before stopping the arm")
    p.add_argument("--guard-lr-factor", type=float, default=0.3, help="lr multiplier applied on a guard verdict")
    p.add_argument("--topq", type=float, default=0.3)
    p.add_argument("--pos-target", type=float, default=0.9, help="1.0 = exact mil_train_1667.py parity")
    p.add_argument("--w-neg", type=float, default=1.0)
    p.add_argument("--w-pos", type=float, default=1.0)
    p.add_argument("--w-prior", type=float, default=0.5)
    p.add_argument("--prior-lo", type=float, default=0.2)
    p.add_argument("--prior-hi", type=float, default=0.6)
    p.add_argument("--patch", type=int, default=128, help="must stay 128: InstanceNorm statistics are per-extent")
    p.add_argument("--layer-start", type=int, default=5)
    p.add_argument("--layer-end", type=int, default=21)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--infer-batch", type=int, default=0, help="0 = 32 on CUDA, 8 on CPU")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--curve-every", type=int, default=50, help="steps between curve points (AUC, drift, guards)")
    p.add_argument("--monitor-tiles", type=int, default=6)
    p.add_argument("--holdout-w", type=int, default=256, help="train-half strip the sampler never draws (monitor)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", help="cudnn deterministic (slower on T4)")
    p.add_argument("--save-ckpt", action="store_true", help="save villa-loadable fine-tuned weights per arm")
    p.add_argument("--smoke", action="store_true", help="400x600 crop, split_x 300, 6 steps, batch 2")
    a = p.parse_args()
    if a.smoke:
        a.batch_size, a.steps, a.monitor_tiles = 2, 6, 2
        a.log_every = a.curve_every = 3
    if a.lr is None:
        a.lr = LR_DEFAULT[a.train]
    if not 0 < a.topq <= 1:
        p.error("--topq must be in (0, 1]")
    if not 0 < a.pos_target <= 1:
        p.error("--pos-target must be in (0, 1]")
    if not 0 <= a.prior_lo <= a.prior_hi <= 1:
        p.error("--prior-lo/--prior-hi must satisfy 0 <= lo <= hi <= 1")
    if a.guard_patience < 1:
        p.error("--guard-patience must be >= 1")
    if not 0 < a.guard_lr_factor < 1:
        p.error("--guard-lr-factor must be in (0, 1)")
    if a.drift_every < 1:
        p.error("--drift-every must be >= 1")
    return a

def pick_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {name} requested but CUDA is not available")
    return torch.device(name)

def seed_all(seed, deterministic=False):
    """Same seeds and data order everywhere; cudnn autotuning is left on for the fixed shapes on CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = torch.cuda.is_available() and not deterministic

def fmt(v, n=4):
    return "n/a" if v is None else f"{v:.{n}f}"

def amp_ctx(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()

def file_id(path):
    st = os.stat(path)
    return f"{os.path.basename(path)}:{st.st_size}:{int(st.st_mtime)}"

# ----------------------------------------------------------------------------- data
def load_data(path, smoke):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    stack, truth, ink9um = d["stack"], d["truth"], d["ink9um"]
    bands_o, bands_f = d["bands_oracle"].astype(np.int8), d["bands_free"].astype(np.int8)
    split_x = int(meta["split_x"])
    if smoke:
        stack, truth, ink9um = stack[:, :400, :600], truth[:400, :600], ink9um[:400, :600]
        bands_o, bands_f, split_x = bands_o[:400], bands_f[:400], 300
    D, H, W = stack.shape
    assert truth.shape == (H, W) == ink9um.shape and bands_o.shape == (H,) == bands_f.shape
    assert 0 < split_x < W
    return {"stack": np.ascontiguousarray(stack), "truth": np.ascontiguousarray(truth),
            "ink9um": np.ascontiguousarray(ink9um), "bands_oracle": bands_o, "bands_free": bands_f,
            "split_x": split_x, "meta": meta}

class PatchSampler:
    """Random training patches entirely inside x < x_hi, with H/V flips (no rotations).

    Yields the RAW uint8 block (D_in, p, p) infer.py would build, the per-pixel bag map (row band broadcast
    along x) and, only when truth_bin was passed, the truth patch.  The MIL arms construct the sampler with
    truth_bin=None, so no truth array is reachable from the MIL loss.
    """

    def __init__(self, layers, truth_bin, bands, x_hi, patch, seed):
        H = layers.shape[1]
        if x_hi < patch or H < patch:
            raise ValueError(f"patch {patch} does not fit in the training area ({H} x {x_hi})")
        self.layers, self.truth_bin, self.bands = layers, truth_bin, bands
        self.x_hi, self.patch = x_hi, patch
        self.rng = np.random.default_rng(seed)

    def sample(self, B):
        D, H, _ = self.layers.shape
        p, with_truth = self.patch, self.truth_bin is not None
        x = np.empty((B, D, p, p), np.uint8)
        bag = np.empty((B, p, p), np.int8)
        t = np.empty((B, p, p), np.uint8) if with_truth else None
        i, attempts, budget = 0, 0, 200 * B + 200
        while i < B:
            attempts += 1
            if attempts > budget:
                raise RuntimeError(f"no non-empty {p}x{p} patch in x < {self.x_hi} after {budget} draws")
            y0 = int(self.rng.integers(0, H - p + 1))
            x0 = int(self.rng.integers(0, self.x_hi - p + 1))
            xi = self.layers[:, y0:y0 + p, x0:x0 + p]
            if not xi.any():                       # infer.py skips all-zero raw patches; so do we
                continue
            bi = (np.full((p, p), -1, np.int8) if self.bands is None
                  else np.broadcast_to(self.bands[y0:y0 + p, None], (p, p)))
            ti = self.truth_bin[y0:y0 + p, x0:x0 + p] if with_truth else None
            if self.rng.random() < 0.5:            # flip along x: rows stay rows
                xi, bi = xi[:, :, ::-1], bi[:, ::-1]
                ti = None if ti is None else ti[:, ::-1]
            if self.rng.random() < 0.5:            # flip along y: the bag map flips with the image
                xi, bi = xi[:, ::-1, :], bi[::-1, :]
                ti = None if ti is None else ti[::-1, :]
            x[i], bag[i] = xi, bi
            if with_truth:
                t[i] = ti
            i += 1
        return x, bag, t

def to_batch(raw_u8, device):
    """RAW uint8 (B, D, p, p) -> (B, 1, D, p, p) float, exactly infer.py's per-patch preprocessing."""
    return torch.stack([preprocess(np.ascontiguousarray(r)) for r in raw_u8]).to(device)

# ----------------------------------------------------------------------------- losses / safeguards
def mil_loss(logits, bag, a):
    """Bag-level loss.  Sees only logits and the bag map (1 / 0 / -1).  Returns (total, terms, selected)."""
    terms, total = {}, logits.new_zeros(())
    selected = torch.zeros_like(bag, dtype=torch.bool)
    neg = bag == 0
    if bool(neg.any()):
        ln = logits[neg]
        l_neg = F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
        terms["neg"], total = l_neg, total + a.w_neg * l_neg
    pos_terms, prior_terms = [], []
    for i in range(logits.shape[0]):
        pm = bag[i] == 1
        n = int(pm.sum())
        if n == 0:
            continue
        lp = logits[i][pm]
        k = min(n, max(1, int(math.ceil(a.topq * n))))
        top = torch.topk(lp, k)
        pos_terms.append(F.binary_cross_entropy_with_logits(top.values,
                                                            torch.full_like(top.values, a.pos_target)))
        p_mean = torch.sigmoid(lp).mean()
        prior_terms.append(F.relu(a.prior_lo - p_mean) + F.relu(p_mean - a.prior_hi))
        sel = torch.zeros(n, dtype=torch.bool, device=logits.device)
        sel[top.indices] = True
        selected[i][pm] = sel
    if pos_terms:
        l_pos, l_prior = torch.stack(pos_terms).mean(), torch.stack(prior_terms).mean()
        terms["pos"], terms["prior"] = l_pos, l_prior
        total = total + a.w_pos * l_pos + a.w_prior * l_prior
    return total, terms, selected

def kd_loss(student_logits, teacher_logits, bag, selected, w_unsel):
    """Soft-target BCE toward the frozen teacher where the MIL loss gives no letter-level information.

    Two SEPARATE means (U2's two lambdas), so the region weights do not track batch composition:
    weight 1.0 over bag <= 0 (interline + ambiguous) and w_unsel over unselected bag == 1 pixels.
    """
    per = F.binary_cross_entropy_with_logits(student_logits, torch.sigmoid(teacher_logits), reduction="none")
    out = student_logits.new_zeros(())
    m0, m1 = bag <= 0, (bag == 1) & ~selected
    if bool(m0.any()):
        out = out + per[m0].mean()
    if w_unsel > 0 and bool(m1.any()):
        out = out + w_unsel * per[m1].mean()
    return out

def l2sp(params, anchor, lambdas):
    """sum_g lambda_g / 2 * ||theta_g - theta0_g||^2 over the trainable parameters."""
    total = None
    for name, p, g in params:
        d = (p - anchor[name]).pow(2).sum() * (0.5 * lambdas[g])
        total = d if total is None else total + d
    return total

def group_of(name):
    for g, (prefixes, _, _) in GROUPS.items():
        if any(name.startswith(pre) for pre in prefixes):
            return g
    return None

def group_drift(params, anchor, base_norms):
    acc = {g: 0.0 for g in base_norms}
    with torch.no_grad():
        for name, p, g in params:
            if g in acc:
                acc[g] += float((p.detach() - anchor[name]).pow(2).sum())
    return {g: math.sqrt(v) / max(base_norms[g], 1e-12) for g, v in acc.items()}

def project_trust_region(by_group, anchor, base_norms, radii):
    """Hard per-group trust region: what actually bounds the drift (the L2-SP gradient does not, under Adam)."""
    clipped = []
    with torch.no_grad():
        for g, plist in by_group.items():
            sq = sum(float((p - anchor[n]).pow(2).sum()) for n, p in plist)
            r = radii[g] * base_norms[g]
            if r > 0 and sq > r * r:
                s = r / math.sqrt(sq)
                for n, p in plist:
                    p.mul_(s).add_(anchor[n], alpha=1.0 - s)
                clipped.append(g)
    return clipped

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
    """One metric for every arm and for the published map: 4x4 pooled AUC on the test half."""
    row_rule = "pooled row counts as text row iff all 4 source rows have bands_oracle==1"

    def __init__(self, truth, bands_oracle, split_x, patch):
        self.truth_pooled = pool4((truth[:, split_x:] >= 128).astype(np.float32)) >= 0.5
        H4, W4 = self.truth_pooled.shape
        b = bands_oracle[:H4 * 4].reshape(H4, 4)
        self.band_pos = np.broadcast_to((b == 1).all(axis=1)[:, None], self.truth_pooled.shape)
        self.band_neg = np.broadcast_to((b == 0).all(axis=1)[:, None], self.truth_pooled.shape)
        self.row_mask = self.band_pos
        # inner control: pooled columns whose source pixels are >= patch away from the split, i.e. produced
        # by tiles that saw no training-half context at all
        self.inner = np.zeros_like(self.truth_pooled, bool)
        self.inner[:, min(W4, patch // 4):] = True

    def __call__(self, prob_test):
        pp = pool4(np.asarray(prob_test, np.float32))
        assert pp.shape == self.truth_pooled.shape, f"{pp.shape} != {self.truth_pooled.shape}"
        sel = self.band_pos | self.band_neg
        rin = self.row_mask & self.inner
        return {"auc_all": safe_auc(self.truth_pooled, pp),
                "auc_rows": safe_auc(self.truth_pooled[self.row_mask], pp[self.row_mask]),
                "auc_all_inner": safe_auc(self.truth_pooled[self.inner], pp[self.inner]),
                "auc_rows_inner": safe_auc(self.truth_pooled[rin], pp[rin]),
                "auc_row_vs_interline": safe_auc(self.band_pos[sel], pp[sel]),
                "prob_mean": float(np.asarray(prob_test, np.float32).mean()),
                "prob_std": float(np.asarray(prob_test, np.float32).std()),
                "n_cells": int(pp.size), "n_row_cells": int(self.row_mask.sum()),
                "ink_frac_cells": float(self.truth_pooled.mean())}

class TileMonitor:
    """Fixed tiles for the learning curve; 'heldout' also carries the guards, 'test_peek' is a peek.

    Every statistic a guard reads is TRUTH-FREE: it uses the arm's own band labels (bands_free for mil_free)
    and the frozen teacher's own probabilities.  `auc_proxy` (vs the public ink9um map thresholded at 0.5) is
    the truth-free learning-curve AUC; `auc_truth_diag` is the same AUC against truth, kept as a DIAGNOSTIC
    that no guard, stopping rule or selection path reads.
    """

    def __init__(self, layers, truth_bin, proxy_bin, bands, x_lo, x_hi, patch, n, seed, name):
        self.name, self.raw, self.patch = name, None, patch
        self.t_probs = self.t_thr = self.t_stripe = None
        self.n_rows_pos = self.n_rows_neg = 0
        self.has_signal = False
        H = layers.shape[1]
        if n <= 0 or x_hi - x_lo < patch or H < patch:
            return
        rng = np.random.default_rng(seed)
        ys = rng.integers(0, H - patch + 1, size=n)
        xs = rng.integers(x_lo, x_hi - patch + 1, size=n)
        self.ys, self.xs = ys.tolist(), xs.tolist()
        self.raw = np.stack([layers[:, y:y + patch, x:x + patch] for y, x in zip(ys, xs)])
        self.truth = np.stack([truth_bin[y:y + patch, x:x + patch] for y, x in zip(ys, xs)]).astype(bool)
        self.proxy = np.stack([proxy_bin[y:y + patch, x:x + patch] for y, x in zip(ys, xs)]).astype(bool)
        self.bands = np.stack([np.broadcast_to(bands[y:y + patch, None], (patch, patch)) for y in ys])
        rows = self.bands[:, :, 0]
        self.n_rows_pos, self.n_rows_neg = int((rows == 1).sum()), int((rows == 0).sum())
        self.has_signal = self.n_rows_pos > 0 and self.n_rows_neg > 0

    def describe(self):
        if self.raw is None:
            return f"{self.name}: none"
        return (f"{self.name}: {len(self.raw)} tiles at x={self.xs} y={self.ys} "
                f"({self.n_rows_pos} text rows / {self.n_rows_neg} interline rows)")

    def probs(self, model, device, batch, amp_dtype):
        if self.raw is None:
            return None
        was_training = model.training
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(self.raw), batch):
                with amp_ctx(device, amp_dtype):
                    logits = model(to_batch(self.raw[i:i + batch], device))
                out.append(torch.sigmoid(logits.float()).cpu().numpy()[:, 0])
        model.train(was_training)
        return np.concatenate(out)

    def set_teacher(self, tp):
        """Freeze the reference the truth-free statistics are measured against (the pristine checkpoint)."""
        self.t_probs = tp
        if tp is None:
            return
        inter = self.bands == 0
        if inter.any():
            ti = tp[inter]
            self.t_thr = 0.5 if (ti >= 0.5).any() and (ti < 0.5).any() else float(np.quantile(ti, 0.7))
        s = self.stripe_tiles(tp)
        if s:
            self.t_stripe = safe_auc(np.concatenate([a for a, _, _ in s]), np.concatenate([b for _, b, _ in s]))

    def agreement_tiles(self, probs):
        """Per tile: (teacher label at its own threshold, student probability) over INTERLINE pixels."""
        if probs is None or self.t_probs is None or self.t_thr is None:
            return []
        out = []
        for i in range(len(self.raw)):
            m = self.bands[i] == 0
            if m.any():
                out.append((self.t_probs[i][m] >= self.t_thr, probs[i][m]))
        return out

    def stripe_tiles(self, probs):
        """Per tile: (row is a text row, student row-mean probability, teacher row-mean probability)."""
        if probs is None:
            return []
        out = []
        for i in range(len(self.raw)):
            b = self.bands[i][:, 0]
            m = b >= 0
            if not m.any():
                continue
            tp = (self.t_probs[i].mean(axis=1)[m] if self.t_probs is not None
                  else np.zeros(int(m.sum()), np.float32))
            out.append((b[m] == 1, probs[i].mean(axis=1)[m], tp))
        return out

    def stats(self, probs):
        """Truth-free curve statistics (+ the labelled truth diagnostic)."""
        if probs is None:
            return {}
        d = {"auc_proxy": safe_auc(self.proxy, probs), "auc_truth_diag": safe_auc(self.truth, probs)}
        ag = self.agreement_tiles(probs)
        if ag:
            d["agreement_auc"] = safe_auc(np.concatenate([a for a, _ in ag]),
                                          np.concatenate([b for _, b in ag]))
        st = self.stripe_tiles(probs)
        if st:
            d["stripe_auc"] = safe_auc(np.concatenate([a for a, _, _ in st]),
                                       np.concatenate([b for _, b, _ in st]))
            if self.t_stripe is not None and d["stripe_auc"] is not None:
                d["stripe_delta"] = d["stripe_auc"] - self.t_stripe
        return d

# ----------------------------------------------------------------------------- guards
def boot_sigma(tiles, stat, n_boot, rng):
    """Spread of a tile-aggregated statistic under resampling of the monitor's OWN tiles, with replacement.

    `tiles` is one tuple of equal-length arrays per monitor tile and `stat` is applied to the concatenation
    of a resample, so the returned standard deviation is how far this reading moves for no reason other than
    which tiles the monitor happened to draw.  None when it cannot be estimated (< 2 tiles, too few defined
    replicates).  This is what turns a hard-coded floor into a measured one: a crossing smaller than a few of
    these sigmas is noise, not degradation.
    """
    n = len(tiles)
    if n < 2 or n_boot < 8:
        return None
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = stat(*[np.concatenate([tiles[i][c] for i in idx]) for c in range(len(tiles[0]))])
        if v is not None:
            vals.append(v)
    return float(np.std(vals)) if len(vals) >= max(8, n_boot // 2) else None

def _stripe_delta_stat(y, ps, pt):
    a, b = safe_auc(y, ps), safe_auc(y, pt)
    return None if a is None or b is None else a - b

class Guard:
    """Truth-free stopping rule for one fine-tuning arm.

    A reading that crosses a threshold is a CROSSING; a crossing that also exceeds --guard-sigma bootstrap
    sigmas of that monitor's own tile spread is a VIOLATION.  Crossings block the clean snapshot (so a state
    the monitors already disliked can never become the rollback target) but never stop anything; violations
    have to repeat at --guard-patience consecutive curve points before the guard acts, and the first action
    is a rollback plus an lr reduction, not a stop.  Weight-space drift is watched through the trust region's
    clip activity, because the projection pins the drift value at its bound and a pinned value can never
    exceed it.
    """

    def __init__(self, args, arm, mon, use_tr, seed):
        self.a, self.arm, self.mon, self.use_tr = args, arm, mon, use_tr
        self.rng = np.random.default_rng(seed)
        self.streak, self.drops, self.events = 0, 0, []
        self.data_on = mon is not None and mon.raw is not None and mon.has_signal

    def signals(self):
        s = [f"trust-region clip rate > {self.a.max_clip_rate:.0%}" if self.use_tr else
             f"per-group drift > DRIFT_MAX every {self.a.drift_every} step(s)"]
        if self.data_on:
            s += [f"interline agreement < {self.a.min_agreement}",
                  f"stripe delta > {self.a.max_stripe_delta:+.2f} or < {-self.a.max_stripe_drop:+.2f}"]
        return s

    def crossings(self, st):
        """Raw threshold crossings, with no noise model: (name, gap, message).

        A crossing is not enough to stop an arm, but it IS enough to disqualify the state as a rollback
        target -- the snapshot the guard restores must be one that crossed nothing.
        """
        out, eps = [], 1e-6        # eps: a rolled-back model sits at agreement 1 - 1e-16, not a crossing
        if not self.data_on or not st:
            return out
        if st.get("agreement_auc") is not None and st["agreement_auc"] < self.a.min_agreement - eps:
            out.append(("agreement", self.a.min_agreement - st["agreement_auc"],
                        f"agreement {st['agreement_auc']:.3f} < {self.a.min_agreement}"))
        if st.get("stripe_delta") is not None:
            d = st["stripe_delta"]
            if d > self.a.max_stripe_delta + eps:
                out.append(("stripe_rise", d - self.a.max_stripe_delta,
                            f"stripe_delta {d:+.3f} > {self.a.max_stripe_delta:+.3f} (stripe detector)"))
            if -d > self.a.max_stripe_drop + eps:
                out.append(("stripe_drop", -d - self.a.max_stripe_drop,
                            f"stripe_delta {d:+.3f} < {-self.a.max_stripe_drop:+.3f} (row contrast lost)"))
        return out

    def sigmas(self, st, probs):
        """The monitor's own bootstrap spread for the statistics it produced (logged at every curve point)."""
        out = {}
        if not self.data_on or probs is None:
            return out
        if st.get("agreement_auc") is not None:
            out["agreement"] = boot_sigma(self.mon.agreement_tiles(probs), safe_auc, self.a.guard_boot,
                                          self.rng)
        if st.get("stripe_delta") is not None:
            out["stripe"] = boot_sigma(self.mon.stripe_tiles(probs), _stripe_delta_stat, self.a.guard_boot,
                                       self.rng)
        return out

    def data_verdict(self, st, probs, sig=None):
        """(violations, crossings, diagnostics) from the arm's own truth-free monitor statistics.

        A crossing becomes a violation only when it is larger than --guard-sigma of the monitor's measured
        spread; otherwise it is reported as a note and the arm keeps training.
        """
        viol, cross, diag = [], [], {}
        cr = self.crossings(st)
        if not cr:
            return viol, cross, diag
        if sig is None:
            sig = self.sigmas(st, probs)
        k = self.a.guard_sigma
        for name, gap, msg in cr:
            cross.append(msg)
            s = sig.get("agreement" if name == "agreement" else "stripe")
            if s is None or gap > max(k * s, 1e-6):
                viol.append(msg + (f" (gap {gap:.3f} > {k:g}*sigma={k * s:.3f})" if s is not None
                                   else " (monitor spread not estimable)"))
            else:
                diag.setdefault("within_noise", []).append(
                    f"{msg}: gap {gap:.3f} <= {k:g}*sigma={k * s:.3f}")
        return viol, cross, diag

    def weight_verdict(self, clip_rate, window, drift):
        """Weight-space alarm: clip ACTIVITY while the projection is on, the raw drift when it is off."""
        if self.use_tr:
            if clip_rate is not None and clip_rate > self.a.max_clip_rate:
                return [f"trust region clipped on {clip_rate:.0%} of the last {window} steps "
                        f"> {self.a.max_clip_rate:.0%}"]
            return []
        over = [g for g in drift if drift[g] > DRIFT_MAX[g] + 1e-6]
        return ["drift " + " ".join(f"{g} {drift[g] * 100:.1f}%" for g in sorted(over))] if over else []

    def escalate(self, viol, step, enforce, immediate=False):
        """Advance / reset the hysteresis counter and decide what happens.  -> 'none' | 'lr_drop' | 'stop'."""
        self.streak = (self.a.guard_patience if immediate else self.streak + 1) if viol else 0
        act = "none"
        if viol and enforce and self.streak >= self.a.guard_patience:
            act = "lr_drop" if self.drops < self.a.guard_lr_drops else "stop"
            if act == "lr_drop":
                self.drops += 1
            self.streak = 0
        if viol:
            self.events.append({"step": step, "violations": viol, "streak_action": act,
                                "enforced": bool(enforce)})
        return act

# ----------------------------------------------------------------------------- one arm
def build_monitors(arm, layers, data, args, split_x, W):
    """Monitor tiles for one arm, built with the arm's OWN bands.

    mil_free gets bands_free, so no truth-derived array can reach its guards; mil_oracle and supervised are
    truth-informed arms by construction and get bands_oracle.
    """
    bands = data["bands_free"] if arm == "mil_free" else data["bands_oracle"]
    truth_bin = (data["truth"] >= 128).astype(np.uint8)
    proxy_bin = (data["ink9um"] >= 128).astype(np.uint8)       # truth-free pseudo-labels: the public map
    mons = [TileMonitor(layers, truth_bin, proxy_bin, bands, max(0, data["train_x_hi"]), split_x, args.patch,
                        args.monitor_tiles, args.seed + 1, "heldout"),
            TileMonitor(layers, truth_bin, proxy_bin, bands, split_x, W, args.patch,
                        args.monitor_tiles, args.seed + 2, "test_peek")]
    return [m for m in mons if m.raw is not None], ("bands_free" if arm == "mil_free" else "bands_oracle")

def run_arm(arm, data, layers, args, device, evaluator, amp_dtype):
    split_x, patch = data["split_x"], args.patch
    seed_all(args.seed, args.deterministic)                   # identical init / patch order for every arm
    model, info = load_ink9um(args.ckpt, device)
    assert info["patch_size"] == patch and info["input_depth"] == layers.shape[0], (info, layers.shape)
    running = [n for n, _ in model.named_buffers() if "running_" in n]
    res = {"arm": arm, "running_stat_buffers": len(running), "groups_trained": [], "trainable_params": 0,
           "curve": [], "train_seconds": 0.0, "steps_run": 0, "weights_from_step": 0, "guard": None,
           "arm_status": "not trained (baseline)"}

    if arm != "baseline":
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        monitors, band_src = build_monitors(arm, layers, data, args, split_x, data["truth"].shape[1])
        sel = next((m for m in monitors if m.name == "heldout"), None)
        train_groups = TRAIN_SETS[args.train]
        for name, p in model.named_parameters():
            p.requires_grad_(group_of(name) in train_groups)
        params = [(n, p, group_of(n)) for n, p in model.named_parameters() if p.requires_grad]
        by_group = {g: [(n, p) for n, p, gg in params if gg == g] for g in train_groups}
        use_l2sp = args.anchor in ("l2sp", "both")
        use_tr = args.trust_region and use_l2sp      # the hard projection is the binding half of L2-SP
        use_kd = args.anchor in ("distill", "both") and arm in MIL_ARMS   # supervised has per-pixel evidence
        teacher = copy.deepcopy(model).eval().requires_grad_(False)
        anchor = {n: p.detach() for n, p in teacher.named_parameters()}
        base_norms = {g: math.sqrt(sum(float(anchor[n].pow(2).sum()) for n, _ in by_group[g]))
                      for g in train_groups}
        lambdas = {g: GROUPS[g][2] * args.w_anchor for g in GROUPS}
        pgroups = [{"params": [p for _, p in by_group[g]], "lr": args.lr * GROUPS[g][1], "name": g}
                   for g in train_groups if by_group[g]]
        opt = torch.optim.AdamW(pgroups, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)  # L2-SP replaces wd
        warm = max(1, args.steps // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: ((s + 1) / warm if s < warm else
                            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm)))))
        scaler = None
        if device.type == "cuda":                 # torch.amp.GradScaler is 2.4+; older wheels have the alias
            scaler = (torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler")
                      else torch.cuda.amp.GradScaler())
        guard = Guard(args, arm, sel, use_tr, args.seed + 7)
        res.update({"groups_trained": list(train_groups),
                    "trainable_params": int(sum(p.numel() for _, p, _ in params)),
                    "lr_per_group": {g: args.lr * GROUPS[g][1] for g in train_groups},
                    "baseline_group_norms": base_norms, "l2sp": use_l2sp, "kd": use_kd,
                    "trust_region_active": use_tr, "monitor_bands": band_src,
                    "monitors": [m.describe() for m in monitors],
                    "guard_signals": guard.signals(), "guard_data_active": guard.data_on,
                    "guard_lr_drops": 0})
        # no BatchNorm and no running-stat buffers anywhere: train() and eval() are bit-identical here
        model.train(not running)
        truth_bin = (data["truth"] >= 128).astype(np.uint8) if arm == "supervised" else None
        bands = {"mil_free": data["bands_free"], "mil_oracle": data["bands_oracle"]}.get(arm)
        sampler = PatchSampler(layers, truth_bin, bands, data["train_x_hi"], patch, args.seed)
        w_kd = args.w_kd if use_kd else 0.0
        for m in monitors:
            m.set_teacher(m.probs(teacher, device, args.infer_batch, amp_dtype))
        print(f"\n=== {arm}: train '{args.train}' = {res['trainable_params']:,} params in {len(train_groups)} "
              f"groups | lr(head) {args.lr:g} | {args.steps} steps x batch {args.batch_size} | anchor "
              f"{args.anchor} (l2sp {(args.w_anchor if use_l2sp else 0):g}{'+TR' if use_tr else ''}, kd {w_kd:g}) "
              f"| device {device}", flush=True)
        print(f"[{arm}] monitor bands: {band_src} | " + " | ".join(m.describe() for m in monitors) +
              f"\n[{arm}] guards ({'enforced' if args.rollback else 'LOGGED ONLY (--no-rollback)'}, "
              f"patience {args.guard_patience}, {args.guard_sigma:g} sigma, warmup {warm} steps): "
              + "; ".join(guard.signals()) +
              (f"\n[{arm}] guard: no held-out monitor with both text and interline rows -- the data-driven "
               f"guards are INACTIVE for this run, only the weight-space check is enforced"
               if not guard.data_on else "") +
              (f"\n[{arm}] NOTE: --anchor {args.anchor} disables the hard trust-region projection; the raw "
               f"per-group drift is checked every {args.drift_every} step(s) instead"
               if args.trust_region and not use_tr else "") +
              f"\n[{arm}] curve AUCs: auc_proxy is truth-free (vs the public map); *truth is a DIAGNOSTIC "
              f"that no guard or selection path reads", flush=True)

        def restore(state):
            with torch.no_grad():
                for nm, p, _ in params:
                    p.copy_(state[nm].to(p.device))
            opt.state.clear()                     # stale Adam moments would undo the rollback in a few steps

        t0, run, clip_hits, clip_win, win0 = time.time(), {}, 0, 0, 0
        snap, snap_step = {n: anchor[n] for n, _, _ in params}, 0   # step 0 = pristine weights
        w_step, stop_reason = 0, None   # w_step: the step the weights in the model come from (0 = pristine)
        for step in range(1, args.steps + 1):
            raw, bag_np, t_np = sampler.sample(args.batch_size)
            xb = to_batch(raw, device)
            bag = torch.from_numpy(bag_np).to(device)
            with amp_ctx(device, amp_dtype):
                logits = model(xb)[:, 0]
                if w_kd > 0:
                    with torch.no_grad():
                        t_logits = teacher(xb)[:, 0].float()
            logits = logits.float()
            if arm == "supervised":
                loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(t_np).to(device).float())
                terms, selected = {"bce": loss}, torch.zeros_like(bag, dtype=torch.bool)
            else:
                loss, terms, selected = mil_loss(logits, bag, args)
            if terms:                                     # else: no usable bag pixels -- skip the update only
                if w_kd > 0:
                    l_kd = kd_loss(logits, t_logits, bag, selected, args.w_kd_unsel)
                    terms["kd"], loss = l_kd, loss + w_kd * l_kd
                if use_l2sp:
                    l_sp = l2sp(params, anchor, lambdas)
                    terms["l2sp"], loss = l_sp, loss + l_sp
                terms["loss"] = loss
                opt.zero_grad(set_to_none=True)
                stepped = True
                if scaler is not None:
                    prev_scale = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([p for _, p, _ in params], 1.0)
                    scaler.step(opt)
                    scaler.update()
                    stepped = scaler.get_scale() >= prev_scale     # fp16 overflow -> optimiser step skipped
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for _, p, _ in params], 1.0)
                    opt.step()
                if stepped:
                    sched.step()
                    if use_tr:
                        c = project_trust_region(by_group, anchor, base_norms, DRIFT_MAX)
                        clip_hits += len(c)
                        clip_win += bool(c)
                for k, v in terms.items():
                    run[k] = run.get(k, 0.0) + float(v.detach())
                run["n"] = run.get("n", 0) + 1
                w_step = step
            res["steps_run"] = step

            # weight-space alarm between curve points: with the projection OFF the drift is unbounded and a
            # run can travel hundreds of percent between two curve points, so check it every --drift-every.
            if not use_tr and step % args.drift_every == 0:
                dr_now = group_drift(params, anchor, base_norms)
                wv = guard.weight_verdict(None, 0, dr_now)
                if wv:
                    act = guard.escalate(wv, step, args.rollback, immediate=True)   # a hard bound: no hysteresis
                    print(f"[{arm}] WARNING safeguard (step {step}): {'; '.join(wv)}"
                          f"{'' if args.rollback else '  [not enforced: --no-rollback]'}", flush=True)
                    if act != "none":
                        restore(snap)
                        w_step = snap_step
                        if act == "lr_drop":
                            for pg in opt.param_groups:
                                pg["lr"] *= args.guard_lr_factor
                            sched.base_lrs = [b * args.guard_lr_factor for b in sched.base_lrs]
                            res["guard_lr_drops"] = guard.drops
                            print(f"[{arm}] rolled back to step {snap_step}"
                                  f"{' (pristine)' if snap_step == 0 else ''} and continuing at "
                                  f"{args.guard_lr_factor:g}x lr", flush=True)
                        else:
                            stop_reason = (step, wv)
                            print(f"[{arm}] rolled back to step {snap_step}"
                                  f"{' (pristine)' if snap_step == 0 else ''} and stopped this arm", flush=True)
                            break

            last = step == args.steps
            if step % args.log_every == 0 or last or step % args.curve_every == 0:
                n = max(run.pop("n", 1), 1)
                means = {k: v / n for k, v in run.items()}
                run = {}
                window = max(1, step - win0)
                entry = {"step": step, "lr": {g["name"]: g["lr"] for g in opt.param_groups},
                         "tr_clips": clip_hits, **means}
                if step % args.curve_every == 0 or last:
                    entry["drift"] = group_drift(params, anchor, base_norms)
                    entry["clip_rate"] = (clip_win / window) if use_tr else None
                    sel_probs = None
                    for m in monitors:
                        pm = m.probs(model, device, args.infer_batch, amp_dtype)
                        entry[m.name] = m.stats(pm)
                        if m is sel:
                            sel_probs = pm
                res["curve"].append(entry)
                dr = entry.get("drift", {})
                mon = " ".join(f"{m.name}[9um={fmt(entry[m.name].get('auc_proxy'))} "
                               f"*truth={fmt(entry[m.name].get('auc_truth_diag'))}]"
                               for m in monitors if m.name in entry)
                gst = entry.get(sel.name, {}) if sel is not None and sel.name in entry else {}
                print(f"[{arm}] step {step:4d}/{args.steps} " +
                      " ".join(f"{k}={means[k]:.4f}" for k in TERM_ORDER if k in means) +
                      (f" | {mon}" if mon else "") +
                      (" | drift " + " ".join(f"{g}={dr[g] * 100:.2f}%" for g in sorted(dr)) if dr else "") +
                      (f" tr_clips={clip_hits} clip_rate={entry['clip_rate']:.0%}"
                       if use_tr and entry.get("clip_rate") is not None else "") +
                      (f" | agree={fmt(gst.get('agreement_auc'), 3)} "
                       f"stripe_delta={fmt(gst.get('stripe_delta'), 3)}" if gst else "") +
                      f" | {time.time() - t0:.0f}s", flush=True)
                if "drift" in entry:                      # safeguard verdict on this curve point
                    st = entry.get(sel.name, {}) if sel is not None else {}
                    sig = guard.sigmas(st, sel_probs)     # the monitor's measured noise band, always logged
                    entry["monitor_sigma"] = sig
                    if sig:
                        print(f"[{arm}] monitor spread (bootstrap over the {len(sel.raw)} held-out tiles): "
                              + " ".join(f"sigma[{k}]={fmt(v, 3)}" for k, v in sig.items()), flush=True)
                    dviol, dcross, diag = guard.data_verdict(st, sel_probs, sig)
                    wviol = guard.weight_verdict(entry["clip_rate"], window, dr)
                    viol, cross = wviol + dviol, wviol + dcross
                    enforce = args.rollback and step >= warm
                    entry["guard"] = {"violations": viol, "crossings": cross, "enforced": enforce, **diag}
                    if guard.data_on and not st.get("agreement_auc") and not st.get("stripe_auc"):
                        print(f"[{arm}] guard: no signal at this curve point (the held-out monitor returned "
                              f"neither an agreement nor a stripe statistic)", flush=True)
                    for note in diag.get("within_noise", []):
                        print(f"[{arm}] note: {note} -- inside the monitor's own spread, not a violation",
                              flush=True)
                    act = guard.escalate(viol, step, enforce)
                    if viol:
                        print(f"[{arm}] WARNING safeguard: {'; '.join(viol)} "
                              f"[streak {guard.streak or args.guard_patience}/{args.guard_patience}"
                              f"{'' if enforce else ', NOT enforced: warmup' if step < warm else ''}"
                              f"{'' if args.rollback else ', NOT enforced: --no-rollback'}]", flush=True)
                    entry["guard"]["action"] = act
                    win0, clip_win = step, 0
                    if act != "none":
                        restore(snap)
                        back = snap_step
                        if guard.data_on and snap_step > 0:        # the restored state has to pass, too
                            st2 = sel.stats(sel.probs(model, device, args.infer_batch, amp_dtype))
                            cr2 = guard.crossings(st2)
                            if cr2:
                                restore(anchor)
                                back = 0
                                print(f"[{arm}] the step-{snap_step} snapshot still crosses a guard "
                                      f"threshold ({'; '.join(m for _, _, m in cr2)}) -- falling back to "
                                      f"the pristine weights", flush=True)
                        if back == 0:
                            snap, snap_step = {n: anchor[n] for n, _, _ in params}, 0
                        w_step = back
                        if act == "lr_drop":
                            for pg in opt.param_groups:
                                pg["lr"] *= args.guard_lr_factor
                            sched.base_lrs = [b * args.guard_lr_factor for b in sched.base_lrs]
                            res["guard_lr_drops"] = guard.drops
                            print(f"[{arm}] rolled back to step {back}{' (pristine)' if back == 0 else ''} "
                                  f"and continuing at {args.guard_lr_factor:g}x lr "
                                  f"(drop {guard.drops}/{args.guard_lr_drops})", flush=True)
                        else:
                            stop_reason = (step, viol)
                            print(f"[{arm}] rolled back to step {back}"
                                  f"{' (pristine)' if back == 0 else ''} and stopped this arm", flush=True)
                            break
                    elif not cross:                       # only a point that crossed NOTHING can be a snapshot
                        snap = {nm: p.detach().cpu().clone() for nm, p, _ in params}
                        snap_step = step
        res["train_seconds"] = time.time() - t0
        res["final_drift"] = group_drift(params, anchor, base_norms)
        res["trust_region_clips"] = clip_hits
        res["guard_events"] = guard.events
        res["weights_from_step"] = w_step
        if guard.events:
            res["guard"] = {"step": guard.events[-1]["step"], "violations": guard.events[-1]["violations"],
                            "enforced": bool(args.rollback), "lr_drops": guard.drops,
                            "stopped_at": stop_reason[0] if stop_reason else None,
                            "stopped": stop_reason is not None, "rolled_back_to": w_step}
        pristine = " -- these are the PRISTINE weights, not a fine-tuned model" if w_step == 0 else ""
        res["arm_status"] = (
            f"guard stopped it at step {stop_reason[0]}, scoring step {w_step}{pristine}" if stop_reason else
            f"{res['steps_run']} steps" +
            (f", {guard.drops} guard lr drop(s), scoring step {w_step}{pristine}" if guard.drops else "") +
            ("" if w_step == res["steps_run"] or guard.drops else f", scoring step {w_step}{pristine}"))
        if device.type == "cuda":
            res["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        del teacher, snap
    model.eval()

    t0 = time.time()
    H, W = data["truth"].shape
    prob = predict_map(model, layers, device, patch=patch, overlap=args.overlap, batch=args.infer_batch,
                       roi=(0, H, split_x, W), amp_dtype=amp_dtype, progress=False)
    prob16 = prob.astype(np.float16)                          # metrics are computed on the SAVED array
    res["eval_seconds"] = time.time() - t0
    res.update(evaluator(prob16.astype(np.float32)))
    print(f"[{arm}] test half x>={split_x}: AUC all={fmt(res['auc_all'])} within-rows={fmt(res['auc_rows'])} "
          f"(inner {fmt(res['auc_all_inner'])}/{fmt(res['auc_rows_inner'])}) "
          f"row-vs-interline={fmt(res['auc_row_vs_interline'])} mean={res['prob_mean']:.3f} "
          f"[{res['arm_status']}] (train {res['train_seconds']:.1f}s, inference {res['eval_seconds']:.1f}s)",
          flush=True)
    if args.save_ckpt and arm != "baseline":
        if res["weights_from_step"] == 0:         # these are the INPUT weights, not a fine-tuned model
            print(f"[{arm}] not saving a checkpoint: the arm was rolled back to the pristine weights",
                  flush=True)
        else:
            ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
            w = res["weights_from_step"]
            torch.save({"model": model.model.state_dict(), "config": ck["config"], "step": w},
                       os.path.join(args.out, f"{arm}_step{w}.pth"))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res, prob16

# ----------------------------------------------------------------------------- outputs
def to_u8(a):
    return (np.clip(np.asarray(a, np.float32), 0, 1) * 255 + 0.5).astype(np.uint8)

def save_panel(images, labels, path, width=600):
    resample = getattr(getattr(Image, "Resampling", Image), "BOX")
    strip, gap, tiles = 22, 12, []
    for img, lab in zip(images, labels):
        im = Image.fromarray(to_u8(img))
        im = im.resize((width, max(1, round(im.height * width / im.width))), resample)
        canvas = Image.new("L", (width, im.height + strip), 255)
        canvas.paste(im, (0, strip))
        ImageDraw.Draw(canvas).text((6, 4), lab, fill=0)
        tiles.append(canvas)
    out = Image.new("L", (sum(t.width for t in tiles) + gap * (len(tiles) - 1),
                          max(t.height for t in tiles)), 128)
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + gap
    out.save(path)

def markdown_table(reference, arms, smoke=False):
    """The table the report quotes.  The `steps / guard` column is not decoration: an arm the guard rolled
    back to step 0 carries the INPUT checkpoint, and must never read as a trained result."""
    base = arms.get("baseline")
    ref_note = " (full-frame grid: NOT comparable under --smoke)" if smoke else ""
    lines = ["| arm | AUC all | AUC within rows | delta vs baseline | row-vs-interline AUC | steps / guard |",
             "|---|---|---|---|---|---|",
             f"| kaggle ink_9um map{ref_note} | {fmt(reference['auc_all'])} | {fmt(reference['auc_rows'])} "
             f"| - | {fmt(reference['auc_row_vs_interline'])} | - |"]
    for arm in ARMS:
        r = arms.get(arm)
        if not r:
            continue
        def delta(key):
            if base is None or r.get(key) is None or base.get(key) is None:
                return "n/a"
            return f"{r[key] - base[key]:+.4f}"
        status = r.get("arm_status") or ("-" if arm == "baseline" else f"{r.get('steps_run', 0)} steps")
        status = status.replace(" -- these are the PRISTINE weights, not a fine-tuned model", " = PRISTINE")
        lines.append(f"| {arm} | {fmt(r.get('auc_all'))} | {fmt(r.get('auc_rows'))} | "
                     f"{delta('auc_all')} / {delta('auc_rows')} | {fmt(r.get('auc_row_vs_interline'))} | "
                     f"{status} |")
    return "\n".join(lines)

def fingerprint_of(arm, config):
    keys = BASE_KEYS if arm == "baseline" else BASE_KEYS + TRAIN_KEYS
    return {k: config[k] for k in keys}

def load_merged(path, config):
    results = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                results = json.load(f)
        except (OSError, ValueError):
            results = {}
    kept = {}
    for arm, r in results.get("arms", {}).items():
        if arm in ARMS and r.get("fingerprint") == fingerprint_of(arm, config):
            kept[arm] = r
        else:
            print(f"warning: dropping stale '{arm}' entry from {path} (different config)")
    results["arms"] = kept
    return results

# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    seed_all(args.seed, args.deterministic)
    if not args.infer_batch:
        args.infer_batch = 32 if device.type == "cuda" else 8
    if args.smoke:
        args.infer_batch = min(args.infer_batch, 4)
    amp_dtype = torch.float16 if device.type == "cuda" else None
    os.makedirs(args.out, exist_ok=True)
    t_all = time.time()

    data = load_data(args.data, args.smoke)
    stack, truth, split_x = data["stack"], data["truth"], data["split_x"]
    H, W = truth.shape
    probe, info = load_ink9um(args.ckpt, "cpu")               # authoritative patch size / input depth
    del probe
    args.patch = info["patch_size"]
    layers = select_layers(stack, args.layer_start, args.layer_end, input_depth=info["input_depth"])
    holdout = args.holdout_w
    if holdout < args.patch or split_x - holdout < args.patch:
        if not args.smoke:
            raise SystemExit(f"--holdout-w must be >= --patch ({args.patch}) and leave >= {args.patch} px of "
                             f"training area, so that no evaluation tile (first contributing x0 = "
                             f"{split_x - args.patch + 1}) can contain a trained-on pixel; got {holdout}")
        holdout = 0 if split_x - args.holdout_w < args.patch else args.holdout_w
    data["train_x_hi"] = split_x - holdout
    print(f"data {args.data}: stack {stack.shape} -> layers {layers.shape} "
          f"(nonzero planes {[int(layers[i].any()) for i in range(layers.shape[0])]}), split_x={split_x}, "
          f"train x<{data['train_x_hi']} (held-out strip {holdout}px), test x>={split_x}, device={device}, "
          f"torch={torch.__version__}, smoke={args.smoke}", flush=True)
    if args.smoke:
        print("smoke: NOT protocol-clean -- the bands were derived over the full training half (covers the "
              "smoke test half), the 400x600 crop redefines the Hann tile grid (so the reference row, a "
              "full-frame run, is not comparable and Stage 0 cannot be checked here), 6 optimiser steps "
              "carry no signal and there is no held-out monitor, so the data-driven guards are inactive.  "
              f"Training patches reach x={data['train_x_hi']} > {split_x - args.patch}.", flush=True)

    evaluator = Evaluator(truth, data["bands_oracle"], split_x, args.patch)
    reference = evaluator(data["ink9um"][:, split_x:].astype(np.float32) / 255.0)
    print(f"published kaggle ink_9um map through this evaluator: AUC all={fmt(reference['auc_all'])} "
          f"within-rows={fmt(reference['auc_rows'])} ({reference['n_row_cells']}/{reference['n_cells']} "
          f"pooled cells inside oracle rows, ink fraction {reference['ink_frac_cells']:.3f})", flush=True)

    config = {**vars(args), "device": str(device), "split_x": split_x, "stack_shape": list(stack.shape),
              "train_x_hi": data["train_x_hi"], "holdout_w": holdout, "ckpt_id": file_id(args.ckpt),
              "data_id": file_id(args.data), "stride": round(args.patch * (1 - args.overlap)),
              "row_rule": Evaluator.row_rule, "protocol_clean": not args.smoke, "amp": str(amp_dtype),
              "group_lr_mult": {g: GROUPS[g][1] for g in GROUPS},
              "group_l2sp_lambda": {g: GROUPS[g][2] for g in GROUPS}, "drift_max": DRIFT_MAX,
              "trust_region_active": bool(args.trust_region and args.anchor in ("l2sp", "both")),
              "selection": "none: the reported model of an arm is its last step that passed the guards; no "
                           "monitor, and in particular no truth-derived quantity, selects a step",
              "guard_bands": {"mil_free": "bands_free (truth-free)", "mil_oracle": "bands_oracle",
                              "supervised": "bands_oracle"},
              "torch": torch.__version__, "numpy": np.__version__, "argv": sys.argv}
    rpath = os.path.join(args.out, "results.json")
    results = load_merged(rpath, config)
    results["config"], results["reference_kaggle_ink9um"] = config, reference

    def flush():
        results["table_markdown"] = markdown_table(reference, results["arms"], args.smoke)
        results["total_seconds"] = time.time() - t_all
        with open(rpath, "w") as f:                # rewritten after each arm: a timeout leaves partial results
            json.dump(results, f, indent=2, default=str)

    for arm in (ARMS if args.arm == "all" else (args.arm,)):
        res, prob16 = run_arm(arm, data, layers, args, device, evaluator, amp_dtype)
        if arm == "baseline" and not args.smoke:               # U2 Stage 0: machine-checked input pipeline
            gaps = [abs(res[k] - reference[k]) for k in ("auc_all", "auc_rows")
                    if res.get(k) is not None and reference.get(k) is not None]
            res["stage0_gap"] = max(gaps) if gaps else None
            res["baseline_reproduces"] = bool(gaps) and max(gaps) <= 0.005
            if not res["baseline_reproduces"]:
                print(f"WARNING: baseline deviates from the published ink_9um map by "
                      f"{fmt(res['stage0_gap'])} (> 0.005) -- the input pipeline is not infer.py's", flush=True)
        results["arms"][arm] = {**res, "fingerprint": fingerprint_of(arm, config)}
        np.save(os.path.join(args.out, f"{arm}_prob_test.npy"), prob16)
        Image.fromarray(to_u8(prob16)).save(os.path.join(args.out, f"{arm}_prob_test.png"))
        flush()

    shape, present, maps = (H, W - split_x), [], []
    for a in ARMS:
        f = os.path.join(args.out, f"{a}_prob_test.npy")
        if a in results["arms"] and os.path.exists(f):
            m = np.load(f).astype(np.float32)
            if m.shape == shape:                               # never mix maps from another crop
                present.append(a)
                maps.append(m)
    save_panel([truth[:, split_x:] / 255.0, data["ink9um"][:, split_x:] / 255.0] + maps,
               ["truth (test half)", "kaggle ink_9um"] + present, os.path.join(args.out, "panel_test.png"))
    flush()
    print(f"\n4x4 pooled AUC, test half x>={split_x}, final model of each arm\n"
          f"{results['table_markdown']}\n\nwrote {rpath} ({results['total_seconds']:.1f}s total)")

if __name__ == "__main__":
    main()
