# FINDINGS (Run 5) P6 — the definition of understanding, registered and scored on models that did not exist when it was written

**Plain English.** Three of the five instruments pass by the letter on every model; two fail by one model each, both on the memoriser
side, and the block therefore fails. Every one of the twelve clean-side clauses passed. The Run-4 thresholds are what they were said to be —
midpoints of a gap seen on four pairs — and on seven new pairs the gap on two instruments turned out to be narrower on the memoriser side
than four pairs had shown. That is the result: P6 stays "found", not "registered". Neighbouring work: LieGG (Moskalev, Sepliarskaia, Sosnovik,
Smeulders, NeurIPS 2022); shift consistency: Zhang 2019; Azulay & Weiss 2019.

| instrument | registered (SPEC5) | generalisers (clean 9, 10, 11; ResNet+aug 3, 4) | memorisers (Z113 9, 10; ResNet 3, 4) | null | verdict |
|---|---|---|---|---|---|
| P6.a q(swap) at resid_mid | clean < 0.10; mem > 0.30 | 0.004, 0.006, 0.004 | 0.465, 0.676 | same model untrained 0.010–0.016 (clean side, as SPEC5 §3 said) | **PASS** 5/5 |
| P6.b shift consistency ±2 px | clean ≥ 0.75; mem ≤ 0.35 | 0.988, 0.984 | 0.202, 0.180 | untrained 0.989, 0.853 (clean side) | **PASS** 4/4 |
| P6.c depth slope of the fibre connection scale | clean < −0.35; mem > −0.30 | Z −0.526, −0.562, −0.572; R −0.536, −0.536 | Z +0.018, −0.006; **R −0.470**, −0.138 | label permutation: Z clean −0.11…−0.14, Z mem +0.05…+0.08; **R clean −0.43, −0.41; R mem −0.30, +0.06** | **FAIL** 8/9 (ResNet memoriser s3) |
| P6.d slot shares (a, b, hint) | clean: a, b ≥ 0.40, hint ≤ 0.05; mem: max ≤ 0.42 | (0.476, 0.482, 0.042), (0.489, 0.491, 0.020), (0.487, 0.490, 0.023) | (0.340, 0.308, 0.352); **(0.324, 0.231, 0.445)** | untrained (0.33, 0.33, 0.33) | **FAIL** 4/5 (memoriser s10, max slot 0.445) |
| P6.e full 113-cycle | found on clean; none on mem | found 3/3 | none 2/2 | untrained: none 5/5 | **PASS** 5/5 |
| **block** | all five pass | | | | **FAIL** (3 of 5) |

Reported, no criterion — concentration C = 1 − PR/PR_untrained on ḡ: Z113 clean 0.57, 0.60, 0.59; Z113 memorisers **0.76, 0.80**; ResNet+aug 0.64, 0.55;
ResNet memorisers 0.63, 0.64. Run 4's reason for removing it is confirmed on seven new models: memorisers concentrate the metric at least as much.

## What the two failures are
1. **P6.d, memoriser s10.** Its three slot shares are (0.32, 0.23, 0.45): the memoriser leans on the hint slot for 45% of its input geometry
   while a 0.42 ceiling was registered from two memorisers that read 0.38–0.39. A memoriser is not obliged to spread evenly; it is obliged
   not to *collapse*, and it did not — the clean signature (two slots ≈ 0.49, hint ≈ 0.02) is nowhere near it. The registered number was the
   wrong shape of clause (a ceiling on the maximum) for what the instrument shows (no collapse onto the task slots and no near-zero hint).
   Post-hoc and not a verdict: "hint share > 0.10 or max(a, b) < 0.40" would read 5/5. It is not registered and it upgrades nothing.
2. **P6.c, ResNet memoriser s3.** Slope −0.47 where > −0.30 was required. Its per-layer profile (1.11, 1.01, 0.96, 0.76, 0.65) tightens with
   depth almost like a clean net's (1.03, 0.91, 0.89, 0.65, 0.49); Run 4's two memorisers read 1.16→0.94 and 1.04→0.95. The label-permutation
   null explains it: on this model **random labels give slope −0.30**, and on the augmented clean nets the null is −0.43 against a measured
   −0.54. On ResNets most of the "fibre tightening" is contraction of the whole cloud with depth, not of the class fibres — the instrument
   measured 0.6–0.8 of its clean-side value with the labels destroyed. On Z113 the null is small (−0.11 to −0.14 vs −0.53 to −0.57 measured;
   memorisers +0.05 to +0.08 vs ≈ 0), and the instrument reads what it claims to. **The depth-slope instrument is sound on the transformer and
   is mostly a whole-cloud effect on the CNN**; the ResNet clause should not have been registered with a label-blind null unexamined.

## What holds
- Twelve of twelve clean-side clauses, on five generalisers none of which existed when the numbers were fixed.
- Instruments a, b, e separate every pair with a gap of at least 0.45 (a), 0.78 (b) and a categorical answer (e).
- The untrained nulls came out exactly as SPEC5 §3 predicted: an untrained network sits on the clean side of the two symmetry instruments
  (it *has* its architecture's symmetries) and on the memoriser side of slot collapse and the group. Clause (ii) of the definition is
  "preserves", and an untrained network preserves; the definition needs (i), (iii), (iv) to say "understands", and an untrained network fails all three.
- P6.b with augmentation: PAUSE5_1 records that augmentation *trains* shift consistency, so the with-aug vs no-aug pair is not like for like.
  Like for like (Run 4, no aug on either side): 0.68 vs 0.18. Memorisation costs ≈ 0.5 of shift consistency at equal recipe; augmentation
  supplies the last ≈ 0.3 on the clean side.

**Status of the definition.** Four instruments, one definition — measured on seven fresh matched pairs, 17 of 19 clauses hold. It is not
registered, because two memorisers crossed thresholds set from two. Files: `results/run5/z113_finals.json`, `resnet.json`, `score.json`,
`figures/run5_instruments.png`.
