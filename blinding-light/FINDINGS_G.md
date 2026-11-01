# FINDINGS G — manifold score replacement (m_plateau)

**In plain English.** The replacement fails its own pre-registration on all three counts, and the way it fails is
informative. m_plateau is not stable under bootstrap on two of four systems, it is exactly zero on the ResNet penultimate
layer, and it ranks GPT-2 level with Z113. It should not replace the old m. What *is* worth keeping is the raw ID(k) curve it
is computed from: that curve distinguishes three regimes cleanly, and a single thresholded number throws that away.

**Contradicts the brief's framing:** the brief expected "m_plateau nonzero on ResNet layers" and the ordering
Z113 > ring > ResNet > GPT-2. The ResNet penultimate cloud has *no* dimension plateau at any scale — its estimated dimension
falls monotonically 14.9 → 12.2 → 10.2 → 8.1 → 5.7 → 2.4 as k goes 5 → 160. That is what ten tight class clusters look like,
not a manifold with a dimension. The old m = 0.00 there was not a miscalibration; by a second instrument the penultimate layer
of a trained classifier is not manifold-like at one scale.

| criterion (SPEC2 §G) | measured | verdict |
|---|---|---|
| bootstrap CV < 0.15 on every primary cloud | Z113 0.00 · ring 0.29 · ResNet penultimate undefined (mean 0) · GPT-2 0.24 | NOT MET |
| m_plateau > 0 on ResNet penultimate | 0.000 | NOT MET |
| m_plateau > 0 on ≥ 3 of ResNet layer1–4 | 4 of 4 (0.4, 0.6, 0.8, 0.2) | met |
| ordering Z113 > ring > ResNet > GPT-2 | 0.60, 0.40, 0.00, 0.60 | NOT MET |
| **G overall** | | **FAIL** |

ID(k) at k = 5, 10, 20, 40, 80, 160 (general-rank MLE, ranks (k//2, k)):

| cloud | ID(k) | m_plateau | boot CV | old m (default / k=15) |
|---|---|---|---|---|
| Z113 clean_s1 resid_mid | 10.5 10.0 7.7 7.2 7.3 6.9 | 0.60 | 0.00 | 0.96 / 0.91 |
| Z113 clean_s0 | 11.2 10.2 7.9 7.5 7.7 7.2 | 0.60 | 0.00 | 0.48 / 0.86 |
| Z113 cleanctrl_s2 | 10.7 8.6 7.4 7.5 7.4 6.8 | 0.60 | 0.00 | 0.65 / 0.91 |
| Z113 scrambled_s0 | 9.9 8.5 9.2 11.8 13.4 13.4 | 0.40 | 0.00 | 0.00 / 0.12 |
| ring s0 / s1 / s2 | 4.6 4.2 4.0 3.7 3.4 3.0 (s1) | 0.40 | 0.11 / 0.29 / 0.16 | 0.60–0.71 |
| ResNet clean layer1…4 | 25→16, 25→19, 25→21, 19→6 | 0.4 0.6 0.8 0.2 | 0.33 0.17 0.13 0.00 | 0.00–0.01 |
| ResNet clean penultimate | 14.9 12.2 10.2 8.1 5.7 2.4 | 0.00 | — | 0.00 |
| ResNet scrambled penultimate | 22.2 18.4 15.7 13.3 10.9 9.3 | 0.00 | — | 0.00 |
| GPT-2 L6 | 13.9 15.5 17.2 16.4 17.7 20.9 | 0.60 | 0.24 | 0.06 |

What the curves say (descriptive, POST-HOC reading, no verdict attached):
* **Plateau** — grokked Z113: ID settles at 7–7.5 from k = 20 to 160 on all three seeds (note the true manifold is the image of the pairs (a, b) and therefore 2-dimensional; a plateau at 7 is what a
  2-torus wound at five frequencies looks like at 1,200 samples — the plateau is real, its height is not the intrinsic
  dimension). The memorising Z113 model has
  no plateau and its ID *rises* with scale (9.9 → 13.4).
* **Decay** — ring (4.6 → 3.0) and every ResNet layer: dimension falls with scale, fastest at the penultimate layer. Decay is
  the signature of cluster structure; the clean ResNet decays faster than the scrambled one (2.4 vs 9.3 at k = 160).
* **Growth** — GPT-2 (13.9 → 20.9) and the Z113 memoriser: dimension rises with scale, the signature of a cloud that fills more
  directions the further you look.

Why the statistic itself is poor: with six k values m_plateau can only take the values 0, 0.2, …, 1.0, so its bootstrap CV is
either 0 or large; and a 15% band is crossed by a slow monotone drift (ring) as easily as by a real change of regime.

**Claimable:** the spectral-gap m and m_plateau both fail as a scalar honesty score. The ID(k) *profile* (plateau / decay /
growth) separates generaliser from memoriser on Z113 and is cheap (1–7 s per cloud). Nothing here is a validated manifold score.

Timings: 1–7 s per cloud including 20 bootstraps. Files: `analysis/run2_blockG.py`, `results/run2/blockG.json`,
`logs/run2_blockG.log`.
