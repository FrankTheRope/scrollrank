# Results

*Everything ScrollScout claims, on one page, with links to the experiment that
produced each number. Every figure below comes from a command in this
repository run on public Vesuvius Challenge data.*

## What the tool is

ScrollScout does not detect letters. It measures whether an ink prediction is
**geometrically credible as writing**, and ranks 4 cm² windows accordingly, so
that a human — or a GPU budget — is spent on the most promising regions first.
CPU-only, no training, classical signal processing on top of the public
`ink_9um` models.

## Headline numbers

| claim | number | where |
|---|---|---|
| Separation, text vs papyrus (score p95) | **1.00** vs **0.15–0.30** | [feature_selection](feature_selection.md) |
| Same, with predictions on **both** sides | **0.97–0.99** vs **0.36–0.64** | [pherc1447_predictions](pherc1447_predictions.md) |
| Ranking quality vs random (AP, w035) | **0.64** vs **0.37** | [validation_w035](validation_w035.md) |
| Run-to-run agreement, text present (two seeds) | **0.78** Spearman | [pherc1447_predictions](pherc1447_predictions.md) |
| Cross-model agreement, text present (`ink_9um` vs `canon_2um`, 9 vs 2.4 µm) | **0.64** Spearman | this page |
| Run-to-run agreement, PHerc1447 | **0.16–0.37** (shuffled control: 0.09) | same |
| Raw-render noise floor, 3 segments × 4 projections | max **0.505**, `period` never above 0.42 | [baseline_pherc1447](baseline_pherc1447.md) |
| Cross-model agreement, PHerc1447 raw segments (22 segments, `gp`/`s5`/`tracer_ft`) | median **0.15**, max **0.39** | [screening_pherc1447_raw](screening_pherc1447_raw.md) |
| `ink_9um` depth sweep on 11 locally rendered PHerc1447 segments (3 depths, 2 seeds, 3 angles) | **0 windows above 0.6** in 393 scorings | [pherc1447_auto_sweep](pherc1447_auto_sweep.md) |
| `ink_9um` on all 22 PHerc1203 segments, rendered locally at 9.362 µm (3 depths, 2 seeds, 3 angles) | seed Spearman median **0.15**; 10 seed-supported windows, all fibre-aligned, none letter-like by eye | [pherc1203_predictions](pherc1203_predictions.md) |
| **Positive control**: `ink_9um` on a text-dense crop of PHerc1667 (known text, 2.4 µm volume resampled to 9.6 µm) | pixel AUC vs published letters **0.76** forward, ~0.5 in reversed depth order: partial transfer, no glyphs, not depth-flip invariant | [positive_control_pherc1667](positive_control_pherc1667.md) |
| Geometry-only training (MIL from row/interline bands) vs true labels vs `ink_9um`, same crop | **0.55** / 0.66 / 0.77 pixel AUC: geometry alone teaches little at 13 cm²; both arms data-starved | [mil_pherc1667](mil_pherc1667.md) |
| Fine-tuning `ink_9um` itself: geometry-only (2 anchors) / oracle geometry / true labels, same crop | **0.710-0.738** / 0.755-0.760 / **0.791** vs 0.771 baseline: geometry degrades it in all 4 configurations, labels improve it; idea closed | [mil_finetune_pherc1667](mil_finetune_pherc1667.md) |

**Operating threshold:** above 0.6 on a prediction is a candidate; below 0.5 is
background. A candidate seen by only one run is not yet a finding: it goes to a
human, with the runs side by side. The tool ranks; it does not decide.

## Four things this repository establishes

**1. Feature weights measured, not chosen.** The score began as a weighted
geometric mean of four hand-designed sub-scores with hand-picked weights. Under
measurement, two of the four lost their vote:

- `ink_fraction` took a **single distinct value** across 440 windows of a real
  prediction — a constant cannot order anything. Demoted to a multiplicative
  gate that vetoes empty windows and smears.
- `stroke_shape` **inverts sign** between synthetic and real data (text 0.49,
  papyrus 0.71–0.92 on real segments). Computed and reported, given no vote —
  and not inverted either, because three papyrus segments against two text
  segments is too thin to justify flipping a sign.
- `line_periodicity` separates the two classes with **no overlap** on five real
  segments and carries the score.

Separation improved from 0.79–0.83 to 0.97 on text. The change **costs**
performance on synthetic data, where `stroke_shape` happens to be correctly
signed; the regression test was relaxed from 2× to 1.5× over random with the
reason written inside the test.

**2. A benchmark that can fail the tool.** Region-level Recall@K after
non-maximum suppression, Average Precision, leave-one-out ablations of every
sub-score, and two baselines (raw ink fraction, random). Plus cross-segment
paired statistics with a signed-rank test, because a difference of 0.01 on one
segment is not a result. The first version of this benchmark was itself wrong —
connected-component regions collapsed a dense segment to 7 targets and every
ranking scored 1.000 — and the fix is documented rather than quietly applied.

**3. A documented negative result on a Grand-Prize-eligible scroll.** The public
`ink_9um` models (2 seeds, both directions) on the three pre-rendered
`auto_grown` segments of **PHerc. 1447** show no detectable text. One window
crossed the 0.6 threshold and was **set aside on seed disagreement**: seed 42 and
seed 43 pointed at regions with zero overlap. Reproducible in ~10 minutes on a
free Kaggle GPU, with a sanity check on PHerc0139 w035 that reproduces the
tutorial's published predictions before anything else is believed.

**4. Two false-positive modes named and characterised.** Fibre crosshatch
(bright vertical bundles raising shape scores while periodicity stays low), and
fibre-bundle periodicity at ~3.3 mm, close enough to the ~4.7 mm line pitch of
PHerc0139 handwriting to matter. Both found by looking at what the tool ranked
first, not by theory.

## Commands

```
scrollscout score          rank 4 cm² windows; heatmap, overlay, JSON
scrollscout concordance    do independent runs point at the same places?
scrollscout benchmark      Recall@K, AP, ablations, baselines vs annotations
scrollscout benchmark-suite the same across many segments, paired statistics
scrollscout aggregate      mean / std / consistency over an ensemble
scrollscout project        surface-volume slices → 2D projections
scrollscout catalog        the 13 eligible scrolls and their S3 paths
scrollscout synth          synthetic data, to try it with no downloads
```

## Reproducing

```bash
pip install -e ".[dev]" && pytest -q          # 12 tests
python3 docs/experiments/separation.py         # the separation table
```

Each document lists the exact commands and data paths for its own figures.
No Vesuvius Challenge data is committed to this repository: it is CC BY-NC 4.0
and is fetched from the public bucket by the commands shown.

## Limitations, stated plainly

- **Five real segments** is a small sample. `anisotropy` earns its place on a
  0.89-vs-0.61 gap that one more scroll could erase.
- **PHerc0139, PHerc1447, PHerc0800 and PHerc1203** have been examined this way —
  every eligible scroll that has public segments. The other ten have none.
- The PHerc1447 negative concerns **three segments out of sixteen**, one depth
  window, two seeds, one checkpoint. It does not say the scroll has no ink — it
  says these models, on these segments, show none.
- `line_periodicity` needs several consecutive lines inside a window, so
  partially recovered text and sparse annotations are underestimated. On the
  hand-made labels the score drops to 0.22–0.53 for exactly this reason.
- The tool ranks; it does not read. Papyrologists do that.

## Documents

- [`feature_selection.md`](feature_selection.md) — how the weights were measured, and what it cost
- [`benchmark_pherc0139.md`](benchmark_pherc0139.md) — six annotated segments, real predictions: what the within-segment benchmark can and cannot say
- [`pherc1447_predictions.md`](pherc1447_predictions.md) — the GPU session, the negative result, the candidate set aside
- [`screening_pherc1447_raw.md`](screening_pherc1447_raw.md) — 132 published predictions on 22 raw segments screened on CPU: no candidate survives cross-model concordance
- [`pherc1447_auto_sweep.md`](pherc1447_auto_sweep.md) — the eleven unrendered PHerc1447 meshes rendered locally, `ink_9um` at three depths: nothing above threshold, and meshes too fragmentary for 10 mm windows
- [`pherc1203_predictions.md`](pherc1203_predictions.md) — PHerc1203, 22 good meshes at the model's native resolution: no text; rotation to ±45° multiplies the fibre false positive and seed agreement does not defend against it
- [`positive_control_pherc1667.md`](positive_control_pherc1667.md) — the control that was missing: on a scroll read in full, `ink_9um` localises the known text (pixel AUC 0.76) without drawing letters, and only in one depth order (a property of the model, not of the sheet); window rankings were the wrong instrument for this
- [`mil_pherc1667.md`](mil_pherc1667.md) — can writing geometry (rows, interlinear gaps) teach a network what ink looks like? First test: weak signal, overfits at this scale; next is scale and geometry-guided fine-tuning of `ink_9um`
- [`mil_finetune_pherc1667.md`](mil_finetune_pherc1667.md) — the same idea applied as fine-tuning of the public model: every geometry-only configuration degrades it (−0.011 to −0.062) while 13 cm² of true labels improve it (+0.020); and a guard that the loss optimises is blind by construction
- [`validation_w035.md`](validation_w035.md) — validation on real predictions of a known-text segment
- [`baseline_pherc1447.md`](baseline_pherc1447.md) — the raw-render noise floor and the false-positive modes
- [`GPU_INFERENZA.md`](GPU_INFERENZA.md) — running the GPU half on a rented or free machine
