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

## #2 (2026-09-08, design — the v1 probe corpus has only 60 distinct sentences per group)
`build_corpus` (copied from v1) samples with replacement from 5 frames × 12 occupations, so *every* pool it
produces, at any n, is a multiset supported on 60 distinct sentences per group; the residual clouds behind
B3 (and v1's BiasTop) were therefore 60 distinct points with multiplicities, and a Rips diagram of a
multiset is the diagram of its support (duplicates only add zero-length bars). The pre-registered n grid
up to 800 per group was never attainable, and the resumed run failed at n = 800 (`Cannot take a larger
sample than population`) because 2n = 1600 exceeded even the duplicated pool of 800.

Post-hoc fix (not pre-registered): `blib.EXTRA_FRAMES` adds 35 neutral frames with the same {occ}/{pronoun}
slots → `build_corpus_unique` = all 40 × 12 = 480 distinct sentences per group, no duplicates. B3 pools are
now these 480 + 480 points; synthetic cells (shift at δ = 0, stretch) use n ∈ {25, 50, 100, 200} (2n ≤ 480),
the real M/F reference uses n ∈ {25, 50, 100, 200, 400}. n = 800 is dropped as infeasible. The 19 rows
computed on 2026-09-07 stay in `B3_power.csv` as `variant = shift_v1corpus` (they are FPR estimates on the
degenerate pool, see #1) and are not used for the summary. B2 (2026-09-07) used the duplicated 100 + 100
corpus; its direction is a frequency-weighted mean over ≤ 60 distinct sentences per group — left as run,
flagged in the PAUSE 4 report; the kill outcome (disjoint-halves cosine 0.65–0.79) is if anything
flattered by the duplication and still fails.
