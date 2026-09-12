# Validation: real ink predictions on a known-text segment (ScrollRank v0.3)

*The raw-render baseline (`baseline_pherc1447.md`) establishes what the scorer
does on papyrus without ink. This document establishes what it does on real
model output where text is known to exist. Together they define the operating
thresholds.*

## Data

Segment **w035 of PHerc. 0139**, the reference segment of the official ink
detection tutorial. Predictions come from the public cross-scroll models
`scrollprize/ink_9um` (hybrid 3D→2D, two independent seeds, checkpoint
step-020000), as published on the tutorial page, rescaled for display:

- https://scrollprize.org/img/tutorials/ink-9um-w035-seed42-20k.webp
- https://scrollprize.org/img/tutorials/ink-9um-w035-seed43-20k.webp

Pixel size was recovered by comparing the image dimensions (1600 × 1441) with
the full-resolution label of the same segment (5820 × 5240 at 9.362 µm, from
the `ink_9um` labels dataset, `native9` folder): factor 3.64 in both axes →
**34.0 µm/px**.

No training, no fitting: the scorer was frozen before these images were used.

## Commands

```bash
scrollrank score data/w035_pred/seed42.tif --pixel-size-um 34.0 \
  --window-mm 10 --stride-mm 2 --no-auto-mask --out out/w035_pred42 --top-k 8
scrollrank score data/w035_pred/seed43.tif --pixel-size-um 34.0 \
  --window-mm 10 --stride-mm 2 --no-auto-mask --out out/w035_pred43 --top-k 8
```

`--no-auto-mask` because a prediction's zeros mean "no ink", not "outside the
mesh".

## Results

| | seed 42 | seed 43 |
|---|---|---|
| Windows scored | 440 | 440 |
| Best score | **0.933** | **0.924** |
| Windows > 0.6 | 117 | 88 |
| Windows > 0.5 | 173 | 151 |
| `period` (best window) | 1.00 | 1.00 |
| Detected line pitch (top 8) | 3.9–4.8 mm | 4.4–4.9 mm |
| Best window (px) | (600,1080)–(894,1374) | (600,1080)–(894,1374) |

Both seeds rank the same window first, and 6 of the top 8 windows coincide.
The heatmap is bright on the left and lower part of the segment, where the
lines are clean, and dark in the upper right, where the prediction is noisy —
the ranking follows legibility, not just ink density.

## Comparison with the negatives

| Input | Best score | Windows > 0.6 |
|---|---|---|
| Real prediction, known text (w035) | 0.92–0.93 | 88–117 / 440 |
| Raw render, PHerc1447 (3 segments, 12 projections) | 0.50 | 0 |
| Synthetic blobby noise | 0.44 | 0 |
| Synthetic stripes | 0.04 | 0 |

## Operating thresholds (v0.3)

- **> 0.6** on a prediction: candidate — look at it, render deeper, run more
  checkpoints.
- **0.5–0.6**: ambiguous; check `period` and whether other seeds agree.
- **< 0.5**: background.

Line pitch on PHerc. 0139 is **~4.7 mm**. Horizontal fibre bundles in PHerc1447
renders produced spurious periods at 3.2–3.4 mm, so for scrolls with similar
handwriting `--pitch-min 4.0` is recommended. The pitch must be re-checked per
scroll; the sparse `ink_9um` labels for PHerc. Paris 4 suggest a similar value
but were too sparse to confirm it.

## Known limitation revealed by the labels

On the hand-annotated `ink_9um` *labels* (sparse: only clearly visible strokes
are marked, a few letters per window) the scorer gives only 0.22–0.53, because
`line_periodicity` needs several consecutive lines inside a window. Dense model
predictions do not have this problem, but partially recovered text might. A
future version should let `stroke_shape` carry more weight when `period` is
undefined rather than penalising sparse but well-shaped strokes.

## Reproducibility

Code: this repository at tag `v0.4`. Results were reproduced independently on
two machines with identical numbers (deterministic pipeline, no random state).
