# Feature selection, measured (ScrollRank v0.6 → v0.8)

*Until v0.5 the score was a weighted geometric mean of four sub-scores with
weights chosen by hand (1 / 4 / 1 / 1). This document reports the experiments
that replaced two of those weights with zero, and why the change is defensible
on the evidence available rather than on intuition.*

## What was wrong with the previous evidence

The v0.3 validation reported a maximum score of 0.93 on real predictions of a
known-text segment against 0.50 on raw papyrus. A maximum is not a measure of a
ranking: it says nothing about how often the tool puts text in front of a human
who can only look at a few windows. The v0.4 benchmark replaced it with Average
Precision, ablations and baselines, and immediately showed the problem: on
PHerc0139 w035 the full score reached AP 0.643 (seed 42) and 0.631 (seed 43),
while `line_periodicity` alone reached 0.663 and 0.601. One feature matched
four, and the order flipped between two runs of the same model.

## Experiment 1 — saturation

The benchmark reports, per sub-score, the fraction of windows saturated at 1.00
and the number of distinct values. On the w035 predictions:

| sub-score | at 1.00 | distinct values (440 windows) |
|---|---|---|
| `ink_fraction` | 1.00 | **1** |
| `line_periodicity` | 0.09 | 384 |
| `stroke_shape` | 0.00 | 264 |
| `anisotropy` | 0.00 | 403 |

`ink_fraction` took a single value across the whole segment. A term that is
constant cannot order anything: the `solo ink_fraction` row of the ablation was
reporting the scan order, and its AP — identical to the full score to three
decimals — was an artefact, not a result.

The same measurement on 28 hand-annotated label images gives 0.14 saturation for
the same feature, so it is not intrinsically constant. It saturates on *dense
model predictions*, where nearly every window falls inside the plausible band.
The feature is a sanity filter that had been given a vote.

## Experiment 2 — separation between text and papyrus

The question the prizes actually ask is not "rank windows inside one segment"
but "does this area deserve attention at all". So: two segments known to contain
text (the tutorial's `ink_9um` predictions of PHerc0139 w035, seeds 42 and 43)
against three raw renders of PHerc1447 with no visible ink. Same settings, 95th
percentile of each sub-score over all windows.

| segment | `line_periodicity` | `stroke_shape` | `anisotropy` |
|---|---|---|---|
| w035 seed 42 — TEXT | **1.000** | 0.491 | 0.890 |
| w035 seed 43 — TEXT | **1.000** | 0.488 | 0.891 |
| PHerc1447 235910 | 0.282 | 0.919 | 0.613 |
| PHerc1447 025628 | 0.263 | 0.707 | — |
| PHerc1447 034159 | 0.218 | 0.878 | 0.613 |

Three readings:

1. **`line_periodicity` separates the two classes with no overlap** and a large
   margin: 1.000 against a maximum of 0.282. Any threshold between 0.3 and 0.9
   classifies all five segments correctly.
2. **`stroke_shape` is inverted on real data.** Papyrus scores *higher* than
   text on all three segments. In the combined score it was being read as
   evidence for text while pointing at fibres.
3. **`anisotropy` separates in the right direction** but weakly (0.89 vs 0.61).

## Why `stroke_shape` was zeroed and not inverted

Because its sign is not stable. On synthetic data it points the right way (text
1.00 vs blobby noise 0.24); on real data it inverts. The mechanism is plausible:
a fibre lattice survives the top-hat as thin fragmented structures with high
perimeter-to-area ratio, while a blurred model prediction of a letter does not.
But three papyrus segments against two text segments is thin evidence for
flipping a sign, and a flipped feature that is wrong on some future scroll is
worse than an absent one. It is computed, reported, and given no vote.

## Result

Weights are now `line_periodicity` 4, `anisotropy` 1. `ink_fraction` became a
multiplicative gate — it vetoes empty windows and smears without ranking
anything — and its decay was made asymmetric, because with the previous
symmetric decay a completely empty window still scored 0.87 and the gate waved
through exactly what it exists to stop.

Separation of the final score, 95th percentile over windows:

| input | v0.3 (four features) | v0.6 (two + gate) |
|---|---|---|
| w035 seed 42 — text | 0.829 | **0.972** |
| w035 seed 43 — text | 0.791 | **0.973** |
| PHerc1447 034159 — papyrus | 0.256 | **0.190** |
| synthetic text | 0.99 | 0.971 |
| synthetic noise | 0.44 | 0.284 |
| synthetic stripes | 0.04 | 0.014 |

## The trade, stated plainly

Dropping `stroke_shape` **costs** performance on synthetic data, where its sign
happens to be correct, and buys it back on real data, where it is inverted. The
synthetic regression test was relaxed from 2x to 1.5x over random accordingly.
That is the honest shape of the trade, and it is recorded in the test itself
rather than quietly absorbed.

## Addendum (v0.8): `anisotropy` did not survive six segments

The section above kept `anisotropy` on a 0.89-vs-0.61 gap measured on two text
segments. Run on six annotated segments of PHerc. 139 with the published
`canon_2um` predictions ([`benchmark_pherc0139.md`](benchmark_pherc0139.md)),
removing it changes Average Precision by +0.001 (4 wins, 2 losses, p = 0.44).
It has no measurable effect. **v0.8 drops it**; the score is `line_periodicity`
times the ink gate, and separation on the papyrus side improves from 0.190 to
0.149 while text stays at 1.000. The prediction made above — that one more
scroll could erase the gap — was correct, and it took only one.

## The limitation that matters most

The two text segments are model predictions at 34 µm; the three papyrus segments
are raw renders at 8.64 µm. The classes differ in *image type*, not only in the
presence of text, so the separation may partly measure "prediction vs render".
The control that settles it is running `ink_9um` on the three PHerc1447 segments
and repeating the table with predictions on both sides. That is the first thing
the next GPU session should produce, and it now has a precise question to answer
rather than a hope to confirm.

Secondary limitations: five real segments is a small sample; `anisotropy` earns
its place on a 0.89-vs-0.61 gap that one more scroll could erase; and no scroll
outside PHerc0139 / PHerc1447 has been examined this way.

## Reproducibility

Code at tag `v0.6`. Saturation figures come from `scrollrank benchmark`;
cross-segment statistics from `scrollrank benchmark-suite`; the separation
table from a five-line script over `score_image`, included as
`docs/experiments/separation.py`.
