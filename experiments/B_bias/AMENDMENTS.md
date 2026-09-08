# Block B — SPEC amendments

## #1 (2026-09-08, design — B3 planted divergence is invisible to the statistic)
Found while resuming after the 2026-09-08 reboot (B3 was mid-run: 19 of 24 synthetic cells done, power 0.00–0.10
everywhere, including δ = 1.0 × pooled sd). The pre-registered planted divergence is `B = A' + δ·sd·g`, a
*translation* of the whole cloud. BiasTop is the W2 distance between Vietoris–Rips H1 diagrams of the two
PCA-20 projections; the projection is affine with a fixed centre and basis, and a Rips diagram is exactly
translation-invariant, so every shift cell with δ > 0 has the same distribution as δ = 0. The 19 cells already
run are therefore 19 estimates of the false-positive rate (all ≤ 0.10, mean 0.04), not a power curve. They are
kept in `results/B3_power.csv` under `variant = shift` and the remaining shift cells with δ > 0 are **not** run
(provably redundant); shift cells at δ = 0 are completed for n = 400, 800 as the pre-registered FPR check
(≤ 0.07 at α = 0.05).

Post-hoc replacement (clearly labelled, not pre-registered): `variant = stretch`, `B = A' + δ·((A' − Ā')·g) g`,
i.e. the second cloud is stretched by a factor 1 + δ along the bias direction — a divergence a Rips diagram *can*
see. Same n grid, same δ ∈ {0.25, 0.5, 1.0}, same trial counts (50 for n ≤ 200, 20 above), same α. The real
M/F split (`variant = real_MF`) is unchanged and is the reference the v1 BiasTop claim actually rests on.
`results/B3_summary.csv` reports the n for power 0.8 (linear interpolation) per (variant, δ) and the FPR verdict.

Operational: permutations per test are 100 (the runner always used 100, not the SPEC's 200 — recorded here);
trials run in an 8-process pool with per-trial seeds `[seed, n, variant]` (previous cells used one shared
generator, so the shift rows are not bit-reproducible under the new seeding; their distribution is the same
null). B2 is skipped on resume when `B2_kill.json` holds all four blocks. B3 cells are resumable from the CSV.

Consequence for the framing: the v1 "BiasTop detects bias" claim cannot, even in principle, be about a mean
shift between groups; whatever it detects is a change of *shape* of the residual cloud. Stated first in the
PAUSE 4 report.
