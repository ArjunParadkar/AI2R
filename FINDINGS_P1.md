# FINDINGS P1 — what it learned: the group, discovered?

**First, as the brief requires: by the letter the group is NOT discoverable on Z113 without being supplied.** Max-norm isometries of the
output-pullback metric on the 113 tokens: orders 1, 1, 2, 13, 5 on the five grokked models (needed ≥ 113 with a 113-cycle on ≥ 4/5). Ring RNN:
order 1 on 3/3. rot-MNIST: rotation is not Killing at conv1/conv2 in either network (FINDINGS_IMG.md). Full narrative: `PAUSE3_1.md`.
Neighbouring work: LieGG (Moskalev, Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022) extracts Lie generators from trained networks; what is
different here is the finite, metric formulation (isometries of one pulled-back distance matrix) and the quotient-by-depth reading.

| cell | pre-registered | measured | verdict |
|---|---|---|---|
| P1.1 group | order ≥ 113 + 113-cycle on ≥ 4/5 | 1, 1, 2, 13, 5; cycle on 2/5 | **NOT MET** |
| P1.1 frequencies (from the letter generator) | ≥ 4 of 5 at r > 0.9 on ≥ 4/5 | possible on 2 models: 4 and 3 | **NOT MET** |
| P1.2 memoriser | order ≤ 2 | 1 | met |
| P1.3 ring | cyclic order ≥ 36 on 3/3; null ≤ 2 | 1, 1, 1; null 1, 1, 1 | **NOT MET** |
| P1.4 (i) rotation Killing at input, conv1, conv2 in both nets | r < 0.1 | 0.025, 0.26, 0.29 (trained); 0.025, 0.30, 0.34 (untrained) | **NOT MET** |
| P1.4 (ii) rotation null at depth in the trained net only | < 0.05 trained somewhere, untrained nowhere | trained 0.0006 after pooling; untrained also < 0.05 everywhere (threshold below the isotropic expectation) | **NOT MET** |
| P1.5 swap quotient | q < 0.05 at resid_mid on 5/5 (RE-SCORE) | 0.0019–0.0050; invariance depth = resid_mid on 5/5 | met |

**Why the letter fails (post-hoc diagnostic).** A true translation changes D by 3% of the median on average on a grokked model (memoriser 22%,
random relabelling 21–33%) and by 14–42% in its worst entry. The symmetry of a trained network is statistical, not entrywise. On the raw
embedding rows nothing is found at all (orders 1, 1, 1, 1, 3): the embedding is not an isometric copy of Z_113.

**POST-HOC (A-10; cannot upgrade a verdict).** Among candidates whose mean defect is under a quarter of a random relabelling's, a full
113-cycle exists on **5/5 grokked models and not on the memoriser**. The DFT of the token embedding along that discovered generator's
coordinate recovers 5, 4, 4, 4, 4 of each model's five inventory frequencies, r > 0.9 for 5, 4, 4, 4, 3 (r = 0.990–0.999; one 0.35), under one
unit that is unique (1 of 56) and equal to the unit the generator itself implies (1, 1, 1, 43, 42). The group *order* is not recoverable this
way (232, 1006, 244, 1027, 1909 "elements": the mean-norm rule admits sparse permutations; 3,406 on the memoriser).

**The swap quotient is not simply learned.** q(swap) at resid_mid is 0.012 at initialisation (attention pooling at the read-off position is
permutation-invariant up to positional embeddings), rises to 0.29–0.46 during memorisation, is 0.58 in the memoriser, and returns to
0.002–0.005 at generalisation (trajectory in FINDINGS_P7.md). Generalisers keep a symmetry the architecture offers; memorisers break it.

**rot-MNIST (agent IMG).** The "trained" CNN is a weak model (60% on rotated digits, 30% on the 20 probe digits). Graded and post-hoc: the
rotation direction is ~2.2× more visible than a random direction at conv1/conv2 and 0.42–0.48× after global pooling in the trained net
(untrained: 0.76–0.83× then 0.61–0.65×); but translation and shear fall just as far, and scale alone stays visible, so the instrument does
not isolate rotation. q(30°) never drops below 0.29; invariance depth: none, in either network.

**Claimable.** P1.5 and the memoriser control. Not claimable: a discovered group. Worth pre-registering on fresh seeds: generator + characters
from outputs alone under the mean-norm full-cycle rule.
Files: `results/run3/blockZ_finals.json`, `blockRing.json`, `blockIMG.json`; `figures/run3_discovered_group.png`, `run3_rotation_invariance_depth.png`.
