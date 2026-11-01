# FINDINGS (Run 5) T — the finite tower

**Plain English.** With finite transformed inputs and the finite g-length of the segment between activations (no Jacobian), all three
registered clauses pass on both decoys and both memorisers. The two Run-4 artefacts are gone: a memoriser's hint change is visible at the
embedding (0.43–0.48, where the linearisation read 0.00–0.01), and translation stays non-null at every layer. Neighbouring work: LieGG.

Planted check (before any model, `tests/test_run5_finite_length.py`): linear ψ → equals the secant to 1e-9; unit circle → arc length to < 1%,
never shorter than the chord; a null direction → length 0.

| cell | registered | decoy 9 | decoy 10 | memoriser 9 | memoriser 10 | verdict |
|---|---|---|---|---|---|---|
| T.1 hint change null from resid_mid on (decoys) | q < 0.05 at mid, post, logits | 0.006, 0.009, 0.009 | 0.007, 0.010, 0.010 | | | **PASS** |
| T.2 translation by u non-null everywhere (decoys) | q > 0.20 at all four layers | 0.465, 0.470, 0.382, 0.382 | 0.790, 0.828, 0.281, 0.281 | | | **PASS** |
| T.3 hint change non-null everywhere (memorisers) | q > 0.15 at all four layers | | | 0.477, 0.496, 0.711, 0.711 | 0.432, 0.422, 0.572, 0.572 | **PASS** |
| **block** | | | | | | **PASS** |

Finite vs linearised, side by side (embed → resid_mid → resid_post → logits):
- memoriser 9, hint change: finite 0.48 / 0.50 / 0.71 / 0.71 vs linearised 0.01 / 0.42 / 0.71 / 0.71. The embedding *does* carry the hint;
  the linearisation said it did not because a token change is not a small perturbation.
- clean 10, translation: finite 0.46 / 0.41 / 0.18 / 0.18 (clean 9) and linearised 1.57 / 1.29 / 0.19 / 0.19 (clean 10): the linearised
  numbers above 1 were the second artefact — a Jacobian applied to a finite step.
- swap on generalisers: finite 0.57–0.62 at embed, 0.00–0.01 from resid_mid on; on memorisers 0.51–0.72 at resid_mid and 0.80–0.89 after.
  The quotient by the swap is taken at resid_mid in every generaliser and in no memoriser — the same statement as P6.a, now made by finite inputs.
- reported: orbit translation (a, b) → (a+u, b), which changes the answer, reads 0.44–0.95 at embed rising to 0.65–1.35 at the logits on
  generalisers: a non-symmetry moves the representation *more* with depth, a symmetry less. The tower orders transformations by how much of
  them the network keeps.
- ResNet (Euclidean, reported): 2-px shift moves the clean+aug nets by 0.29 → 0.10 across layers and the memorisers by 0.22 → 0.74.

Files: `results/run5/z113_finals.json` (`tower_finite`, `tower_linearised`), `figures/run5_tower.png`, `whiteboxtopo/finite_tower.py`.
