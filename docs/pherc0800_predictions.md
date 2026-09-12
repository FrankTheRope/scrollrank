# PHerc. 0800: first ink predictions, negative

Six public `auto_grown` segments (0.4-2.3 cm², 15-27 mm across) rendered locally
with `vc_render_tifxyz` (see RENDERING.md), `ink_9um` seeds 42/43 both
directions on free Kaggle GPU (24 predictions, 11 min), scored with ScrollRank
v0.8 at 0°, +45° and -45° because the papyrus fibres run diagonally in these
renders. Segment 010146 (7 mm) is too small for a 10 mm window and was skipped.

| segment | Spearman s42/s43 | p95 @0° | @+45° | @-45° | best |
|---|---|---|---|---|---|
| 213516 | 0.499 | 0.280 | 0.126 | 0.165 | 0.28 |
| 220042 | 0.306 | 0.253 | 0.101 | 0.098 | 0.25 |
| 220955 | 0.500 | 0.259 | 0.072 | 0.109 | 0.26 |
| 222030 | 0.349 | 0.312 | 0.258 | 0.112 | 0.31 |
| 225813 | 0.591 | 0.250 | 0.242 | 0.085 | 0.25 |

References: text-bearing w035 scores 1.00 and its seeds agree at 0.78; raw
PHerc1447 papyrus scores 0.11-0.26. No orientation raises any segment above
0.31; the model responds over the entire valid area; no rows of separated marks
are visible. **No detectable text.**

Seed agreement (0.31-0.59) is higher than on PHerc1447 (0.16-0.37) and should
not be read as signal: these segments hold only a few dozen windows, and the
renders show a centre-bright vignetting shared by both seeds. A per-segment
shuffled control with the same window count is the right comparison and is not
yet implemented.

This negative is weaker than the PHerc1447 one. The segments are small,
unrefined, and at the default depth window only; and nothing establishes that
they lie on the recto — the diagonal fibre direction does not tell horizontal
(written side) from vertical (verso) fibres. A negative on a verso says nothing
about ink. Larger, recto-verified segments are the prerequisite for a stronger
statement.
