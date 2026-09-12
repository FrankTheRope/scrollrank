# Baseline: PHerc. 1447 raw surface renders (ScrollRank)

*Measured 2026-09 on the three public `auto_grown` segments of PHerc1447 that
ship with pre-rendered surface volumes (8.64 µm, 31 slices). No ink model was
run: these are raw CT renders, so the expected result is "no text". The point of
this baseline is to know what the scorer does on real papyrus **without** ink,
so that a real signal can be recognised later. The complementary document,
[`validation_w035.md`](validation_w035.md), measures what it does on real model
predictions where text is known to exist.*

## Setup

```bash
scrollrank project data/1447/<ID>_tifs --out data/1447/<ID>
scrollrank score  data/1447/<ID>_<proj>.tif --pixel-size-um 8.64 \
                   --window-mm 10 --stride-mm 2 --top-k 3
```

Projections written by `project`: `mid` (central slice), `max` (max over depth),
`avgc` (mean of the 9 central slices), `minc` (min of the 9 central slices).

Data: `s3://vesuvius-challenge-open-data/PHerc1447/segments/<ID>/surface-volumes/8.64um-1.2m-116keV-volume-20250521151220.tifs/`
for `20250702235910-auto_grown_20250702235910292`,
`20250703025628-auto_grown_20250703025628283`,
`20250703034159-auto_grown_20250703034159599` (~543 MB each).

## Results (best window per image)

| Segment | Size (px) | Valid | Windows | avgc | max | mid | minc | period max |
|---|---|---|---|---|---|---|---|---|
| 20250703034159 | 3620 x 5220 | 50 % | 86 | 0.445 | - | - | - | 0.29 |
| 20250702235910 | 2980 x 3240 | 52 % | 36 | 0.393 | 0.195 | 0.175 | 0.500 | 0.39 |
| 20250703025628 | 4100 x 4260 | 49 % | 72 | 0.294 | 0.337 | 0.505 | 0.407 | 0.42 |

Raw papyrus tops out at **0.505**, and `line_periodicity` never exceeds 0.42
(real text gives ~1.0). No window reaches 0.6. No projection wins consistently -
`minc` on one segment, `mid` on another, `avgc` on the third.

Reference values at the same settings: synthetic text 0.988, synthetic blobby
noise 0.441, pure stripes 0.044; real `ink_9um` predictions of a known-text
segment 0.92-0.93 (see `validation_w035.md`).

## Visual inspection

The renders show well-defined horizontal fibre bundles crossed by vertical
fibres, with no abrupt discontinuities - the meshes follow a single sheet.
Black regions are missing papyrus; dark lines are cracks. No ink or crackle is
visible by eye, which is the expected outcome: directly visible ink in a raw
render is the exception, not the rule.

## False-positive modes identified

1. **Global threshold (v0.1-v0.2).** Otsu over the whole image selected 55-65 %
   of the pixels of a raw render - it split papyrus texture into a light and a
   dark half rather than finding marks. A conspicuous artefact followed: the
   darker `minc` projection fell by chance inside the plausible ink-fraction
   band and won everywhere with 0.45-0.52. Fixed in v0.3 with a local white
   top-hat; after the fix `minc` no longer wins systematically, which is how we
   know the earlier result was an artefact and not a signal.
2. **Fibre crosshatch.** Bright vertical fibre bundles raise `stroke_shape` and
   `ink_fraction` while `period` stays low. Visual check of the top windows of
   segment 025628 confirmed the boxes sat on vertical bright streaks on a
   regular fibre lattice - writing runs horizontally and consists of separate
   marks. Mitigated by giving `line_periodicity` the dominant weight (4 of 7).
   A fifth "discreteness" sub-score (maximum run length of foreground pixels)
   was implemented and discarded: on real data it was 1.00 everywhere,
   discriminating nothing while inflating the total.
3. **Fibre-bundle periodicity.** In segment 025628 the top three windows overlap
   in the same region and all report a 3.2-3.4 mm period at +3..+10 degrees. A
   pitch that regular at ~3 mm is most likely the spacing of horizontal fibre
   bundles, not text lines. Since PHerc. 0139 handwriting has a measured line
   pitch of ~4.7 mm, `--pitch-min 4.0` is recommended for scrolls with similar
   hands. Not yet mitigated in the code.
4. **Out-of-mesh area.** ~50 % of each rendered rectangle is exact zero. Before
   v0.2 these pixels entered the threshold and the projections. Now they are
   masked automatically (`--no-auto-mask` disables this) and windows with < 80 %
   valid coverage are skipped.

## What this baseline is for

It fixes the noise floor. On a raw render, anything below 0.6 is texture. The
decisive sub-score is `period`. When model predictions become available for
these segments, a window above 0.6 will be a candidate precisely because we
measured where the background sits - and a negative result will itself be a
documented answer for three segments nobody has run a model on.

## Reproducibility

Code: this repository at tag `v0.4`. The pipeline has no random state: the same
images give the same numbers, reproduced on two machines. Vesuvius Challenge
data is CC BY-NC 4.0 and is not included in this repository; `.gitignore`
excludes `data/`, `out/` and `ink_9um/`.
