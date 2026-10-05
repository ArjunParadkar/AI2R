# PAUSE 1 — does unsupervised naming work?  (G, E1, E2) · 2026-09-20 09:40

**Plain English.** No. Ranking persistent H¹ classes by how much the output depends on them does not recover the frequency
circles of a grokked Z113 model: 0 of the top-5 classes are named on any of three seeds (criterion: ≥4). Across all 30 candidate
classes per seed the best correlation with *any* frequency of a, b, a+b or a−b is 0.45 on one seed and 0.07 on the other two.
The instrument still certifies and does not discover. The manifold-score replacement also fails its pre-registration.

**Contradicting the brief's framing, stated first.**
1. E2's prediction ("pullback collapses geometric debris") is wrong on Z113: **more** classes clear the noise band under the
   pullback-geodesic metric (26) than under the Euclidean one (12). It holds on the ring (2 → 1, the one true circle) and on
   GPT-2 (22 → 0). On Z113 nothing that clears the band is nameable under either metric, and the longest bar is only 1.2–1.4×
   the band — these are marginal classes, not structure.
2. Output sensitivity s_c does not single out meaningful classes. On the primary seed the top-5 by s_c are all unnamed
   (r ≈ 0.06); the one class with a glimmer of meaning (k = 3 of a+b, the model's dominant frequency, r = 0.45) ranks 5th.
3. Block G: the ResNet penultimate layer has no dimension plateau at any scale (ID 14.9 → 2.4 as k goes 5 → 160); m_plateau = 0
   there, exactly where the brief expected it to be nonzero. GPT-2 scores as high as Z113.

| cell | pre-registered threshold | measured | met? |
|---|---|---|---|
| E1 (primary clean_s1) | ≥4 of top-5 by s_c named at r>0.9 with a key frequency, no group parameter | 0 / 5 (s0: 0/5, ctrl: 0/5) | **NOT MET** |
| E1 null (feature shuffle) | 0 of top-5 named | 0 / 5 | met (vacuously) |
| E2 | none (prediction: fewer classes under pullback) | Z113 12 → 26 ✗ · ring 2 → 1 ✓ · GPT-2 22 → 0 ✓ | 2 of 3 |
| G: CV < 0.15 everywhere | | Z113 0.00, ring 0.29, GPT-2 0.24, ResNet undefined | **NOT MET** |
| G: ResNet penultimate m_plateau > 0 | | 0.000 | **NOT MET** |
| G: ordering Z113>ring>ResNet>GPT-2 | | 0.60, 0.40, 0.00, 0.60 | **NOT MET** |

**Why E1 fails (the run-1 diagnosis, now tested a second way).** The representation is the image of (a, b) — a 2-torus wound
at five frequencies. 1,200 samples of that surface do not put neighbouring residues next to each other in any metric, so
Rips sees thousands of short bars (2,089 H¹ classes; top bar 1.4× the noise band) rather than five long ones. Sensitivity-ranking
re-orders debris; it cannot create classes the filtration never formed. The symmetry-informed route works because it
projects onto one frequency plane at a time, where the circle *is* densely sampled.

**Kill tests.** Feature-shuffle null: 0/5 named, best r 0.065 — the pipeline does not hallucinate names.

**What is now claimable.** (i) Unsupervised discovery of Z113's circles fails under two independent rankings (lifetime in
run 1, output sensitivity here) and two metrics; the symmetry-informed inventory remains the only route, and must be stated as
such. (ii) On systems where the manifold is a single densely sampled circle (ring), the pullback metric reduces the
above-band classes to exactly the true one. (iii) Neither scalar manifold score is fit for purpose; the ID(k) profile
(plateau / decay / growth) is the informative object (FINDINGS_G.md).

Timings: E1 7–11 s per model (1,200 points, 30 coordinates); E2 < 1 min per system; G 1–7 s per cloud.
Files: `results/run2/blockE12.json`, `results/run2/blockG.json`, `FINDINGS_G.md`.

Proceeding to D1/D2 (PAUSE 2).
