# FINDINGS (Run 4) P1 — the group, discovered, registered in advance

**Plain English. This one passes.** With the rule fixed before the models existed — accept σ iff its mean defect is under a quarter of a random
relabelling's, and ask whether a full 113-cycle is among the accepted candidates — a generator is found on **4 of 4 fresh generalisers and on
neither fresh memoriser**, and on 4 of 4 the Fourier transform of the token embedding along the discovered generator recovers at least four of
the model's five frequencies at r > 0.9 under one unit. Nothing was supplied: no group parameter, no ordering of the tokens, no labels.
Neighbouring work: LieGG (Moskalev et al., NeurIPS 2022) extracts Lie generators from trained networks; this is the finite, single-metric
counterpart, from outputs alone. Corrections to the brief's citations (A-10, not A-12; no "δ = 0.06") are in SPEC4.

| cell | pre-registered | measured | verdict |
|---|---|---|---|
| P1.1 generator | full 113-cycle on 4/4 fresh clean; none on 2 fresh memorisers | 4/4; 0/2 | **PASS** |
| P1.1 representation | on ≥ 3/4: ≥ 4 of 5 frequencies at r > 0.9 under one unit | 4, 4, 5, 5 of 5 (r = 0.993–0.999; two weak fifths at 0.45 and 0.96); the unit is unique (1 of 56) and equals the one the generator implies (1, 33, 1, 20) on 4/4 | **PASS** |
| P1.2 definition on planted truth | noisy Fourier embedding → cycle; noise cloud → none; untrained Z113 → none | all three; operating range measured: entrywise isometry is lost by noise 0.05, the mean rule holds to 0.08 (acceptance level 0.25·δ_rand ≈ 0.065 × median) | **PASS** |
| P1.3 tower (structural) | decoys: hint-change null at or before translation; memorisers: nothing null | translation never reaches q < 0.05 (0.22–0.35 at the logits); hint-change reads "null" at the embedding for EVERY model incl. memorisers (0.001–0.005) | **FAIL** |
| P1.4 orbit explanation | ≥ 0.99 on 4 clean; < 0.60 on memorisers | 1.000, 1.000, 1.000, 0.990; memorisers 0.000, 0.000 | **PASS** |

**Said plainly about the pass.** Under the output-pullback metric the statistic is a property of the *function* on all 12,769 inputs (stated in
SPEC3): a model that is right everywhere is translation-equivariant in its predictions by arithmetic. What is not arithmetic, and is what
passed: that the generator is recoverable from the logit geometry with nothing supplied, that it is absent in true memorisers, and that the
network's own embedding, ordered by that generator, decomposes into the frequencies the symmetry-informed route had needed the group for.
Two caveats: both planted-shortcut models also carry the generator (they compute a+b off the leaked residues), and the accepted-set size
(8–4,097) is not a group order — only the full cycle is meaningful.

**Why the tower criterion failed.** Two artefacts of the linearised pullback, both the "infinitesimal misses finite" problem of run 3: at the
embedding the derivative with respect to a NULL hint is flat for every model, memorisers included, although their hint-change q is 0.38–0.41 one
layer later; and translation leaves the *prediction* exactly invariant but moves the logit vector by 0.22–0.35 of a typical distance, so it is
never "null" at q < 0.05. What the tower does show: swap becomes null at resid_mid in every generaliser (0.002–0.005) and never in a memoriser
(0.56–0.85); hint-change is null throughout a grokked decoy (≤ 0.007) and grows with depth in a memoriser (0.38 → 0.66–0.72) and in a
shortcut model (1.07 → 1.23).
Files: `results/run4/z113_finals.json`, `score.json`, `tests/test_run4_delta_symmetry.py`, `figures/run4_tower.png`.
