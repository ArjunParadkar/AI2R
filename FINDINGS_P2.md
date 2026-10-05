# FINDINGS P2 — the orbit explanation

**Plain English.** The orbit explanation is exact. On all five grokked models, every one of 2,000 random inputs gives the same prediction
across its whole 113-element translation orbit {(σⁿa, σ⁻ⁿb)} under the discovered generator σ and across its swap coset {(a,b),(b,a)}:
1.000 on 5/5 (criterion ≥ 0.99). On the memoriser, with the true translation supplied because nothing was discovered, 0.000 of inputs are
orbit-constant (criterion < 0.60). A generaliser respects its own symmetry on every input; a memoriser on none. P2.1 is MET.
The boundary prediction recomputed with the discovered characters is NOT MET.

Caveat stated first: for the three run-1 seeds the generator used is the POST-HOC one (A-10) because the letter found none; for the two
decoy models it is the by-the-letter generator. And a model that is 100% accurate on all 12,769 inputs is orbit-constant under the true
translation by arithmetic; what the cell adds is that the generator was found from the outputs, and that the memoriser scores zero.

| cell | pre-registered | measured | verdict |
|---|---|---|---|
| P2.1 generalisers | ≥ 99% of inputs orbit- and coset-constant on 5/5 | 1.000, 1.000, 1.000, 1.000, 1.000 | met |
| P2.1 memoriser | < 60% | 0.000 (orbit-constant 0.000; swap-coset-constant 0.028) | met |
| P2.2 boundary from discovered characters (RE-SCORE) | top-3 predicted shifts hold ≥ 53.6% of geodesic destinations on 3 seeds | 49% / 22% / 14% (9×, 4×, 2.6× chance); generator post-hoc on all three | **NOT MET** |

P2.2 is weaker than run 1's version (43% / 52% / 25%) because the characters' weights are taken from the token-embedding power along the
discovered coordinate instead of from the variance of the opened layer; the predicted shifts differ (s1: {9, 4, 2} here vs {38, 4, 40}).
Example explanation string (demo): "predicted 14; input (5, 9) lies on translation-orbit {(5+s, 9−s)} and swap-coset {(5,9),(9,5)}, all mapped
to the residue-14 fiber" — verified on the model: (9,5) → 14, (6,8) → 14, (7,7) → 14; on the memoriser the same four inputs give 93, 12, 42, 59.
Neighbouring work: LieGG (Moskalev et al., NeurIPS 2022). Files: `results/run3/blockZ_finals.json`, `blockP22.json`, `logs/demo_run3_transcript.txt`.
