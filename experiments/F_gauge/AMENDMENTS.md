# Block F — amendments (append-only)

**#0 (2026-09-07, before the full F2 run; after a 2-specimen smoke test, no
results/ files written).** The smoke test (XOR3_d2_w6_s0, THR2_d2_w10_s0)
showed 0 of 28 minimal circuits carry a handbook (bit-at-0) interpretation:
with bits fixed at pre-activation 0, a second-layer unit's bit is generally
not a function of its parents' bits because it depends on their real-valued
activations. Decision: promote the Méloux threshold definition (E1's) to
co-primary, analysed in full alongside the handbook definition; specimens
with N = 0 under a definition are excluded from that definition's ratio
statistics and counted. No threshold or hypothesis changed.

**#1 (2026-09-07, before any kill-test result is recorded).** The F4 kill networks (k
algorithms × 2 gauge copies, each copy with its own constant-1 bias chain so that a copy
alone yields ±0.5 and an ablated copy contributes 0 under both criteria) have 14 / 28 / 42
hidden units for k = 1 / 2 / 3. Exhaustive enumeration is run for k = 1 (CPU) and k = 2
(GPU, 2^28); for k = 3 (2^42) the kill is evaluated on the planted circuits directly:
each of the 2k planted circuits must be valid under both criteria, minimal (checked by
enumerating all 2^|C| subsets of each planted circuit), and the 2k explanations must
partition into exactly k orbits. The first version of the planted construction (copies
averaged into a shared output bias) was wrong — a single copy could not cross the output
threshold under zero ablation — and was replaced before use; its output was discarded.

**#2 (2026-09-07, before kill-test results are recorded).** With the sign-of-logit validity
criterion and complement ablation, a parallel composition of k algorithms always admits
*recombined* circuits (e.g. algorithm A's positive-side units with algorithm B's bias
chain, or A's "p∧¬x3" unit with B's "x3∧¬p" unit) — these are genuinely valid circuits of
the planted network, not artefacts, so the literal expectation N = 2K cannot hold for any
construction we found (three constructions tried on k = 1, all showed recombinations).
The F4 kill is therefore evaluated as: (i) every planted copy is a valid circuit under both
criteria; (ii) the two gauge copies of the same algorithm receive the same orbit key and
copies of different algorithms different keys, so K over planted circuits = k for
k = 1, 2, 3; (iii) the exhaustive N and K (k ≤ 2) are reported with the number of recombined
circuits; and (iv) **F4-kill-b (new, stronger)**: for 100 trained specimens (d·w ≤ 16) a
random g ∈ G is applied to the whole network and the multiset of orbit keys of its
explanations must be identical to the original's under both criteria — the direct test that
gauge-related explanations are identified.

**#3 (2026-09-07, before F8–F10 run).** F10 audit (100 renamings × 3 kinds × ~16 candidates,
persistence on 256-point probe clouds) costs ~1 min per specimen; it is run on seeds 0–4 of
every (function, depth, width) cell (760 specimens) rather than all 3 143 converged specimens.
F8 and F9 run on every converged specimen. Nothing else changes.

**#4 (2026-09-07, during the F2 run; affects only the Méloux co-primary column).** Exact Méloux
threshold-interpretation counting is exponential in circuit size; at d·w = 20 single circuits were
taking > 30 min and all 12 workers stalled on THR3/PARITY4 d2w10. A per-circuit wall cap of 60 s is
imposed on `interpretation_count`; capped circuits get `n_meloux = -1` and are excluded from the
Méloux N/K (their number is reported per cell in FINDINGS). The handbook (bit-at-0) definition,
the circuit enumeration, the canonical keys and everything else are unaffected. Specimens
already completed before the cap keep their exact counts.

**#5 (2026-09-07, during F7; 11 of 80 models done).** Two-layer toy transformers (24 nodes, 2^24
subsets) take ~40 min each under GPU sharing. Enumeration switched to float32 with 32 768-mask
chunks (validity is a sign with margin ≥ 0.5, unaffected), and the 2-layer cells are run for seeds
0–4 instead of 0–9 (4 functions × 5 seeds = 20 two-layer models; the 1-layer cells keep 10 seeds).
Completed models are kept.

**#6 (2026-09-07, during F11; no F11 row written yet).** Persistence on the full 2 000-token
clouds (five clouds per renaming, 100th-percentile threshold) cost ~45 min per renaming; the F11
persistence candidates are computed on a fixed 800-token subsample (same tokens for every
renaming) at the v1 95th-percentile threshold. All non-persistence candidates keep the 2 000
tokens. n = 30 per kind unchanged.

**#7 (2026-09-07, F3 accounting; measurement unchanged).** The first F3 pass identified only
dead / duplicate / constant units and left a median excess of 22 flat directions "unclassified" on
1 000 specimens. The identification in `symmetry.hidden` was replaced by an exact first-order
accounting: null directions of the realisation Jacobian confined to a single parameter block
(W_l, b_l) — outgoing weights of a rank-deficient activation layer, incoming weights of units whose
masked inputs are dependent (dead units, units sharing an activation pattern, always-active groups)
— together with the scaling generators span the *explained* fiber; the remainder is *unexplained*.
`corank`, `known_dim` and `excess` are unchanged; only the split of `excess` into explained /
unexplained changed. `symmetry.csv` is recomputed for every specimen by `run_F3.py` (the F2 job's
first-pass values are superseded; the file is rewritten).

**#8 (2026-09-07, DESIGN amendment, principal review — not a threshold change).** The F4/F5 test as
specified cannot return anything but K = N, for two structural reasons:
1. *Within a network*: gauge acts on the whole parameter vector. Two distinct unit subsets of one
   fixed θ are gauge-equivalent only if a symmetry maps θ to itself while carrying one subset onto
   the other — i.e. only if θ has an automorphism (neurons with identical weights). A generically
   trained network has none, so K_within = N always, regardless of anything about explanations.
2. *Pooled across seeds* (the SPEC's primary statistic): equivalence was decided by exact
   coincidence of canonical forms at relative tolerance 1e-6. Two independent trainings coincide
   to 1e-6 with probability zero, so K_pooled = N always as well. E1 (2026-08-30) had both defects,
   which is why the sprint "replicated" it.
The failed planted N = 2K kill (#2) was the symptom: the one test that could have shown the orbit
counter has power was replaced by a consistency check, leaving a test with no demonstrated power
that returns the null everywhere.
**Consequences.** (a) The F5 verdicts computed so far (`results/F5_verdicts.json`, H-F5a/b "fail") are
**FROZEN and withdrawn** — not reported as a result about nature. (b) K_within = N is kept as a
*documented finding about redundancy, not gauge*: Méloux-style within-network multiplicity is
structural redundancy — the planted construction showed duplicated sub-circuits generate
combinatorially many recombined valid circuits, not 2 orbits. Gauge explains multiplicity across
parameterisations, not among subsets of one network; the SPEC conflated the two.
(c) Three reformulations replace F4/F5, in order of importance, each with its own pre-registered
threshold and a working planted kill (`SPEC_F4_reformulated.md`):
   1. **Cross-seed orbit test (primary).** For networks trained from different seeds on the same
      function: is a minimal circuit of seed A gauge-equivalent to one of seed B — does some g ∈ G
      map A's circuit weights onto B's within tolerance, after unit matching and the functional-
      dimension correction? Novelty FLAGGED: adjacent to convergent learning (Li et al. 2015) and
      permutation-matching / re-basin (Ainsworth et al. 2022) — search before write-up.
   2. **Approximate stabilisers within a network.** ε-symmetries: units whose scale-normalised
      incoming weights match within ε; K_within(ε)/N reported as a curve.
   3. **Interpretation-level gauge.** Whether interpretive multiplicity (Méloux thresholds) collapses
      under per-unit sign-flip / negation equivalence.
Nothing in F1–F3, F7–F11 changes; `explanations.csv` is the input to all three.
