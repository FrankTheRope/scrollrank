"""Where the geometry-guided fine-tuning loses AUC (docs/mil_finetune_pherc1667.md, "Where the loss comes from").

    python3 docs/experiments/mechanism_finetune_1667.py

Needs work/mil_1667/mil_1667_w028.npz (docs/experiments/build_mil_1667.py) and the saved test-half probability
maps of the two Kaggle fine-tuning runs (work/ft_kaggle/ft_t4, work/ft_kaggle/ft_l2sp).  Independent of the
training script.  The reference here is the team's canon prediction (ink if >= 128), because the training half
has no official labels; canon over-marks ink relative to them (35.6 % vs 23.8 % on the labelled pixels).

Prints: the share of canon ink among the "certain negative" (interline) pixels of the training strip, for the
truth-free and the oracle bands; the share of test-half ink inside oracle interlines; pixel AUC at five pooling
scales; AUC by region (row bands / interlines / mixed cells, oracle bands, 38 um cells); mean |Laplacian| of the
maps on ink and on background.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import laplace
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SPLIT, TRAIN_X = 1400, 1144          # held-out half starts at x = 1400; fine-tuning trained on x < 1144
MAPS = {
    "baseline": "work/ft_kaggle/ft_t4/baseline_prob_test.npy",
    "mil_free + distillation": "work/ft_kaggle/ft_t4/mil_free_prob_test.npy",
    "mil_free, no distillation": "work/ft_kaggle/ft_l2sp/mil_free_prob_test.npy",
    "mil_oracle + distillation": "work/ft_kaggle/ft_t4/mil_oracle_prob_test.npy",
    "mil_oracle, no distillation": "work/ft_kaggle/ft_l2sp/mil_oracle_prob_test.npy",
    "supervised on canon": "work/ft_kaggle/ft_t4/supervised_prob_test.npy",
}


def pool(a, k):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


d = np.load(ROOT / "work/mil_1667/mil_1667_w028.npz")
ink = d["truth"] >= 128
bands = {"truth-free (from ink_9um)": d["bands_free"], "oracle (from canon)": d["bands_oracle"]}
for name, b in bands.items():
    inter = np.broadcast_to((b == 0)[:, None], (b.size, TRAIN_X))
    rows = np.broadcast_to((b == 1)[:, None], (b.size, TRAIN_X))
    print(f"{name:26s} canon ink in interline pixels (x < {TRAIN_X}): {ink[:, :TRAIN_X][inter].mean():.3f}, "
          f"in row bands: {ink[:, :TRAIN_X][rows].mean():.3f}")
t = ink[:, SPLIT:]
print(f"test half: share of canon ink inside oracle interlines {t[d['bands_oracle'] == 0].sum() / t.sum():.3f}")

M = {n: np.load(ROOT / p).astype(np.float32) for n, p in MAPS.items()}
print("\npixel AUC vs canon by pooling scale")
for k in (2, 4, 8, 16, 32):
    tp = pool(t.astype(np.float32), k) >= 0.5
    print(f"  {k * 9.596:6.1f} um", {n: round(roc_auc_score(tp.ravel(), pool(m, k).ravel()), 3) for n, m in M.items()})

print("\nAUC by region, oracle bands, 38 um cells")
tp = pool(t.astype(np.float32), 4) >= 0.5
b = d["bands_oracle"][: tp.shape[0] * 4].reshape(tp.shape[0], 4)
regions = {"rows": (b == 1).all(1), "interlines": (b == 0).all(1)}
regions["mixed"] = ~(regions["rows"] | regions["interlines"])
for n, m in M.items():
    pm = pool(m, 4)
    print(f"  {n:28s}", {r: round(roc_auc_score(tp[np.broadcast_to(sel[:, None], tp.shape)],
                                                  pm[np.broadcast_to(sel[:, None], tp.shape)]), 3)
                          for r, sel in regions.items()})

print("\nmean |Laplacian| on canon ink / background")
for n, m in M.items():
    lap = np.abs(laplace(m.astype(np.float64)))
    print(f"  {n:28s} ink {lap[t].mean():.4f}  background {lap[~t].mean():.4f}")
