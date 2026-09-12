# Ink predictions on PHerc. 1447: a documented negative result

*Three `auto_grown` segments of a Grand-Prize-eligible scroll, run through the
public `ink_9um` models and through ScrollRank. No text was found. This
document records how, what was measured, and the one candidate that crossed the
threshold and then failed verification — because a negative result nobody can
reproduce is worth nothing, and a candidate nobody retired is worse.*

## Why this run

Two open questions needed the same experiment.

The first was a limitation stated in `feature_selection.md`: the separation
measured there put text-bearing segments at 0.97 and papyrus at 0.19-0.30, but
the two classes were **different kinds of image** — model predictions at 34 µm
against raw renders at 8.64 µm. The separation might partly have measured
"prediction vs render". Only predictions on both sides settle it.

The second was the prize question itself: do the three pre-rendered segments of
PHerc1447 contain ink that current models can see? Nobody had published an
answer.

## Setup

Kaggle notebook, 2× Tesla T4, ~35 minutes of quota total, no cost.

- `villa` cloned; `volume-cartographer` skipped (it fails to build without Ceres
  and is only needed for rendering, which these segments do not require).
  Installed with `pip install -e . --no-deps` plus `pynrrd`, `zarr`, `tifffile`,
  `imagecodecs`, on the notebook's Python 3.12 rather than the 3.14 the package
  requests. It runs.
- Checkpoints: `scrollprize/ink_9um`, seeds 42 and 43, step 075000.
- Inputs: the `8.64um-1.2m-116keV` surface-volume Zarrs of segments
  `20250702235910`, `20250703025628`, `20250703034159` (175, 294, 329 MB;
  31 slices; level 0 at 2980×3240, 4100×4260, 3620×5220).

## Sanity check first

Before trusting anything, the same pipeline was run on **w035 of PHerc0139**,
whose predictions the tutorial publishes: 9.362 µm surface volume, a 3000×3000
central mask, seed 42, both directions. The output shows Greek letters, legible
by eye, matching the published images in character. The pipeline is correct.

This step cost one minute of GPU and is the reason the negative result below can
be believed rather than merely asserted.

## The sweep

3 segments × 2 seeds × both directions = 12 predictions, 9 minutes of GPU.
Default depth window (the script selects 17 source layers automatically),
`--overlap 0.5 --blend-mode hann --batch-size 4`.

## Result

Scored at 10 mm windows, 2 mm stride, `--no-auto-mask`, on the mean of the four
runs per segment:

| input | best score | best `period` |
|---|---|---|
| **PHerc0139 w035 — text present** | **0.97-0.99** | 1.00 |
| PHerc1447 235910 | 0.378 | 0.33 |
| PHerc1447 034159 | 0.362 | 0.33 |
| PHerc1447 025628 | 0.636 | 0.61 |

**The first question is answered.** With predictions on both sides of the
comparison, the separation holds: 0.97 against 0.36-0.64. It was not an artefact
of comparing predictions with renders.

**The second question is answered too, negatively.** Two of the three segments
stay far below the 0.6 operating threshold. `stroke_shape` is 0.00 in every top
window of all three segments: not one connected component has the shape of a
stroke.

## The candidate that failed

Segment 025628 produced a window at **0.636**, above threshold, with `period`
0.61 and a 4.5 mm line pitch — plausible for Greek handwriting, and the highest
periodicity PHerc1447 has ever produced (raw renders never exceeded 0.42).
Visual inspection was ambiguous: diffuse bright patches with a weak vertical
tendency, no strokes, and patch-boundary artefacts near the edge.

The test that retired it was **agreement between seeds**:

| run | best window (x0,y0)-(x1,y1) | score | `period` |
|---|---|---|---|
| seed 42 | (2508, 0)-(3660, 1152) | 0.480 | 0.45 |
| seed 43 | (912, 1596)-(2064, 2748) | 0.633 | 0.59 |

The two models point at regions that do not overlap at all. On w035, where text
is present, the same two seeds produce score fields correlating at 0.78
(Spearman over the window grid; a shuffled control gives 0.09). The 0.636
candidate was an artefact of averaging: two uncorrelated peaks summed into a
place neither seed ranks first.

This is now a command, `scrollrank concordance`, so the check is not something
one has to remember to do. Run on all three segments it gives:

| segment | Spearman | top-decile IoU | top-1 IoU |
|---|---|---|---|
| **PHerc0139 w035 — text present** | **0.778** | 0.354 | 0.101 |
| PHerc1447 025628 | 0.370 | 0.152 | 0.000 |
| PHerc1447 034159 | 0.284 | 0.111 | 0.000 |
| PHerc1447 235910 | 0.163 | 0.059 | 0.000 |
| *shuffled control* | *0.090* | — | — |

All three sit closer to the shuffled control than to the segment with text, and
top-1 IoU is zero in every case: the two seeds never point at overlapping
places. Segment 025628 is the most structured of the three, which is consistent
with it being the one that produced the false positive — enough to explain the
artefact, not enough to support a finding.

## Conclusion

**In the three pre-rendered `auto_grown` segments of PHerc. 1447, the public
`ink_9um` models show no detectable text.** The finding is bounded: it concerns
three segments out of fifteen, one depth window, two seeds, one checkpoint, and
the models' own generalisation to this scroll — which the Challenge lists as an
open problem. It does not say the scroll has no ink.

### The depth sweep, run

The most likely explanation for a false negative was surface offset: the models
are sensitive to where in the stack they look, and only the default window had
been tried. Segment 025628 — the most structured of the three — was re-run at
three non-overlapping depth windows of the 31 available slices (0-16, 7-23,
14-30), two seeds, both directions:

| depth window | Spearman (seeds 42/43) | best score s42 | best score s43 |
|---|---|---|---|
| low (0-16) | 0.336 | 0.384 | 0.569 |
| mid (7-23, default) | 0.370 | 0.480 | 0.633 |
| high (14-30) | 0.325 | **0.697** | 0.432 |

Concordance is flat across depth, and every window sits far below the 0.78 of a
text-bearing segment. Global statistics were equally flat (nonzero fraction
0.649 at every depth, p99 between 140 and 157). **Surface offset does not
explain the negative.**

The `high` window is worth naming: seed 42 produced 0.697, the highest score
PHerc1447 has ever given and well above the 0.6 threshold, while seed 43 on the
same slices gave 0.432 and the two agree at 0.325. Without the concordance check
this would have looked like the strongest candidate of the project.

What would change the answer now, in order of expected value: the remaining
twelve segments, which need rendering and therefore VC3D; a scroll-specific
model, since the Challenge lists cross-scroll ink generalisation as open; and
pseudo-labelling from any region that shows structure under a finer render.

## Reproducibility

Notebook steps as listed above; scoring with this repository at tag `v0.7`:

```bash
scrollrank score data/1447_pred/mean_025628.tif --pixel-size-um 8.64 \
  --window-mm 10 --stride-mm 2 --no-auto-mask --out out/pred1447_025628
scrollrank concordance data/1447_pred/p025628_s42.tif data/1447_pred/p025628_s43.tif \
  --pixel-size-um 8.64 --no-auto-mask
```

Predictions are derived from CC BY-NC 4.0 data and are not committed to this
repository; the twelve TIFFs can be regenerated from the steps above in about
ten minutes of free GPU.
