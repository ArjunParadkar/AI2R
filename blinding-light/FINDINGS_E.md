# FINDINGS E — task-relevant cohomology and invariance as quotient

**In plain English.** The unsupervised naming test fails outright (E1: 0 of 5 on three seeds). The invariance-as-quotient test
gives the run's one clean confirmation: the Z₂ quotient found in run 1 replicates on two models that had never been examined
(E3, Z113). The rot-MNIST half of E3 fails by the letter, as predicted when it was registered.

**Contradicting the brief's framing, first.** (i) Output sensitivity does not pick meaningful classes out of Rips debris — the
brief's premise that the five circles are among the persistent classes is false at this sampling: they are never formed by the
filtration. (ii) "Pullback collapses debris" is wrong on Z113 (12 → 26 classes above the band). (iii) Raw Rips cannot certify the
interval either: the fibers' β₀ is 1 but 1–13 H¹ classes clear the noise band in every fiber; the quotient is established by
the axis-ratio witness and the commutativity measurement, not by homology.

| cell | pre-registered | measured | met? |
|---|---|---|---|
| E1 primary (clean_s1) | ≥ 4 of top-5 by s_c named at r > 0.9 with a key frequency | 0/5; best r among all 30 candidates 0.448 (k=3 of a+b, ranked 5th) | **NOT MET** |
| E1 other seeds | reported | clean_s0 0/5 (best r 0.065); cleanctrl_s2 0/5 (0.065) | — |
| E1 null (feature shuffle) | 0 of top-5 named | 0/5, best r 0.065 | met |
| E2 (no criterion) | prediction: fewer classes above band under pullback | Z113 12 → 26 ✗; ring 2 → 1 ✓ (top bar 5.5× band); GPT-2 22 → 0 ✓ | 2 of 3 |
| E3 Z113, RE-SCORE (clean_s0, clean_s1, cleanctrl_s2) | fiber axis ratio < 0.1, slice > 0.9, swap < 0.02 | 0.025/0.996/0.0075 · 0.017/0.998/0.0039 · 0.020/0.999/0.0034 | met (seen in run 1) |
| **E3 Z113, UNSEEN (decoy_s0, decoy_s1)** | same | **0.019 / 0.997 / 0.0051 · 0.044 / 0.997 / 0.0081** | **MET** |
| E3 rot-MNIST, RE-SCORE | a layer where the trained net has no circle (median r ≤ 0.9) and the untrained net has one | both lose it at conv3_gap (0.57 vs 0.79); fc1 0.29 vs 0.57 | **NOT MET** |
| **E3 overall** | both halves | | **FAIL** (Z113 half met, including on unseen models) |

E3 detail. On all five grokked models — three seeds, two training-set fractions, with and without decoy hints — the answer-fiber's
image in each of the top-4 frequency planes is a segment (axis ratio 0.017–0.044) while the b = 0 orbit in the same planes is a
circle (0.996–0.999), and swapping operands moves the representation by 0.3–0.8% of the fiber radius. The learned invariance is
the Z₂ action (a, b) ↦ (b, a); the fiber is S¹/Z₂, an interval. rot-MNIST, graded: the trained network's orbit coordinate
degrades faster than the untrained one's at every layer after conv2 (gap 0.22 at conv3_gap, 0.28 at fc1), but by the
certification rule the "invariance depth" is the same layer for both.

**Claimable.** The Z₂ quotient of Z113 answer-fibers, on five models, two of them unseen when the criterion was written.
Not claimable: any unsupervised discovery; any layer-resolved invariance depth on rot-MNIST.

Timings: E1 7–11 s per model; E2 < 1 min per system; E3 ~40 s. Files: `results/run2/blockE12.json`, `results/run2/blockE3.json`.
