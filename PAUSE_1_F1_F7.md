# PAUSE 1 — Block F, the spine: F1–F7 (headline decision) — 2026-09-09 FINAL on the full enumeration

Written as a report, not a stop (standing instruction). Status of the inputs: F1 final; F2 2 692/2 694 CPU
specimens + 424/424 d3w8 GPU specimens (the last two d2w10 specimens are on an 8 h wall budget, AMENDMENTS #11;
d3w10 deliberately last in the GPU queue); F3 recomputed in full 2026-09-09 01:11–03:42 (3 143 specimens, totals aggregated 09:09); **F4a/F4b/F4c FINAL** (full reruns 03:42–09:02; F4a 133 cells per criterion,
106 392 / 68 558 pooled circuits under zero / mean ablation; F4b all 2 692 specimens; F4c 3 000 circuits). F5/F6 as pre-registered are **withdrawn** (AMENDMENTS #8). F7 final.

## 1. Plain English
The headline the handbook was built around — that the many "valid explanations" of a small network are mostly the
same explanation written in different coordinates, so that counting gauge orbits collapses the multiplicity —
is dead, twice. First, inside one network it cannot even be tested: a renaming acts on the whole parameter
vector, so two different unit-subsets of the same trained network are gauge-equivalent only if the network has an
automorphism, which a generically trained network never has. K = N is a theorem, not a finding; E1's 2026-08-30
"replication" and this sprint's first F4/F5 pass were both that theorem (AMENDMENTS #8, principal review
2026-09-07). Second, across seeds — the version of the question that can be asked — trained circuits for the same
Boolean function are **not** renamings of each other: with a residual calibrated so that r tracks injected weight
noise (r/δ = 0.54–0.56) and an exact-relabel positive control at 3e-14, the median cell merges essentially nothing
at tolerance 0.1 (median cell K(0.1)/N = 1.000 under both ablation criteria; 1.000 at every τ ≤ 0.03 in every
cell) and even at the very loose 0.5 the median cell merges only 17 % (zero) / 1 % (mean) of its circuits. Same-function cross-seed pairs are closer than different-function pairs (median residual 1.027 vs
1.164, KS D = 0.27, p underflows to 0 with 5.74 M vs 165 pairs), so the test has signal; the signal is just small. What *is* shared
across seeds is the algorithm, not the weights: under zero-ablation the number of distinct gate-composition
strings is well under half the number of circuits (K_alg/N median 0.39; AND4 0.24, OR4 0.28, THR3 0.30 —
monotone functions converge on fewer algorithms; EXACT2 and AND_OR3 1.0), so "same algorithm, different weights" is the
accurate description of multiplicity across seeds. What survives from F1–F7 as a headline candidate is the other
thing Block F measured on the way: **excess flat directions are universal but almost entirely identifiable** —
99.0 % of the 3 143 converged specimens have more flat directions than the permutation × scaling group predicts
(median excess 25; up to 182 at d3w10), and 95.6 % of that excess is accounted for by dead units, constant units
and shared-activation-pattern groups (AMENDMENTS #7 identification); the unexplained remainder has median 0
directions, is positive in 43 % of specimens, and averages 0.25–3.3 directions per specimen, growing with depth
and width. So the "hidden symmetry" of trained small MLPs is degeneracy of the specimen, not a new symmetry group;
what remains open is a small, depth-growing residual. Within one network, approximate gauge copies are real but rare: at 1 % weight tolerance only 3–12 % of specimens have
any pair of circuits that are near-renamings of each other (F4b passes its ≥ 0.95 threshold with a median of 1.0), so
the Méloux-style multiplicity inside a network is structural redundancy, not approximate symmetry; and negating
units never merges interpretations (F4c: 3 000 circuits, zero collapse), so it is not an orientation convention
either. The transformer extension (F7: 60/60 two-layer attention models
converged, median 14 / 24 minimal circuits per model under zero / mean ablation) inherits the withdrawal: its
pooled orbit counts (K/N = 1.0 in all 16 cells) are the same tautology and are not a result.

## 2. Pre-registered thresholds and outcomes
| Item | Threshold | Outcome |
|---|---|---|
| F1 specimens | ≥ 20 seeds per (function, depth, width); drop non-converged | **MET** — 3 648 trained (24 seeds × 152 cells), 3 143 converged (86.2 %); non-converged logged in `specimens.csv` |
| F2 enumeration | all subsets, both ablation criteria; report |mean∩zero|/|mean∪zero| | **MET** for 2 692 + 424 specimens (2 pending on wall budget). Median minimal circuits: 9 (zero), 4 (mean); 15 % / 28 % of specimens have a unique minimal circuit |
| F2 kill | hand-built XOR: exactly one circuit under both criteria | **PASS** (circuit 31 = the planted set, 1 interpretation, both criteria) |
| F3 | record known_dim vs fiber_dim; extract and identify excess | **MET** (final, n = 3 143): G_incomplete 99.0 %; excess median 25; 95.6 % of excess identified (dead / constant units, shared-pattern groups); unexplained median 0, > 0 in 43 % of specimens (mean 0.25 at d2w10 … 3.3 at d3w10) |
| F4 kill (original) | planted 1/2/3 algorithms → K = 1/2/3 | **PASS as amended** (AMENDMENTS #2/#4): K over the *planted* circuits = 1/2/3 and 100/100 gauge-consistency; exhaustive enumeration finds additional recombination circuits (K_exhaustive = 3, 47 for k = 1, 2) that are real, not a counter bug |
| F5a/F5b | median K/N ≤ 0.5; width slopes | **WITHDRAWN** (AMENDMENTS #8) — tautological; `F5_verdicts.json` frozen |
| F4a-1 (primary, reformulated) | median cell K(0.1)/N ≤ 0.5 | **FAIL** — 1.000 (zero), 1.000 (mean); K(τ)/N = 1.000 for every τ ≤ 0.03; 0.828 / 0.993 at τ = 0.5 (q10 across cells 0.445 / 0.816) |
| F4a-2 | K_alg/N ≤ K(0.1)/N in the median cell | **PASS** under zero (0.39 ≤ 1.00); **vacuous** under mean (1.00 ≤ 1.00 — the mean criterion leaves ≤ 1 interpretable circuit per seed in most cells) |
| F4b | median K_within(1e-2)/N ≥ 0.95 | **PASS** — median 1.00 under both criteria (2 692 specimens). Approximate stabilisers exist but are rare at tight ε: at ε = 1e-2, 3 % (mean) / 12 % (zero) of specimens have any collapse (mean ratio 0.997 / 0.985); at ε = 0.1 it is 35 % / 59 % (mean ratio 0.92 / 0.81, zero median 0.90). Within-network redundancy is mostly *not* approximate gauge |
| F4c | ratio 1 expected (no negation collapse) | **AS EXPECTED** — 3 000 sampled circuits (1 < n_Méloux ≤ 20 000): ratio median = mean = 1.000, 0 circuits with any collapse, interpretation counts reproduce n_Méloux in 100 % of circuits (kill pass) |
| F7 | repeat F2–F5 on 1–2-layer attention models | enumeration **MET** (60/60 converged, extended G verified to ≤ 4e-15); orbit verdicts **withdrawn** with F5 |

## 3. Kill-test outcomes
* F2 kill: PASS (above). F4 original kill: PASS as amended (the planted circuits are recovered as one orbit
  per algorithm; the exhaustive count is larger because duplicated sub-circuits recombine — that observation is
  what led to AMENDMENTS #8).
* F4a K1 exact relabel: PASS, r_max = 0.0 over 412 circuit pairs on the final run (3.2e-14 on the partial run;
  three bugs found and fixed by this kill before it first passed: Hungarian on unnormalised vectors, scale gauge
  not fixed, shared-pattern collapse done in the wrong frame). K2 noise tracking: PASS, r/δ = 0.62, 0.61, 0.60 at
  δ = 1e-3, 1e-2, 3e-2, so τ has a meaning (τ = 0.1 ≈ 16 % relative weight noise). K3 planted cross-algorithm: PASS, r(A,B) = 0.86 > 0.3 (A–C and B–C
  have different layer sizes → r = ∞ by definition; only one finite cross-algorithm pair, stated as such).
  K4 null: same-function pairs are closer than different-function pairs (1.027 vs 1.164, KS D = 0.27, p ≈ 0) — the
  test has signal.
* F4b kills: PASS — planted two gauge copies r = 7e-17; duplicated-unit specimen: both-copies circuit collapses to
  the original at 3.8e-15 (the half-weight single-copy minimal circuits differ by a real r = 0.21, as they should).
* F4c kill: PASS (complement pair → one class; different patterns → two classes).

## 4. What contradicts the plan's framing (stated first)
1. The primary hypothesis of the block (F5a) could not have failed or passed — the test was tautological within
   networks and at 1e-6 exact coincidence across seeds. This is the sprint's most important finding about its own
   plan and is why F5/F6 are withdrawn rather than reported as "H-F5a fails".
2. The reformulated cross-seed test, which can fail, does fail in the direction opposite to the headline: circuits
   for the same function from different seeds are essentially as far apart in gauge-residual terms as circuits for
   *different* functions (1.027 vs 1.164 on a [0, 2] scale). Convergent learning at the circuit-weight level does
   not happen in these networks; only algorithmic (gate-string) convergence does, and only under zero-ablation.
3. F6 as specified ("one representative per orbit, name the algorithm") was executed under the withdrawn test, so
   its table (`F6_algorithms.csv`, 5 rows, n_orbits = n_explanations) is not an orbit autopsy and most entries are
   "unnamed" / "partial"; it is superseded by F4a's K_alg count.
4. F3's result is stronger than planned in one direction and weaker in another: the handbook expected occasional
   G_incomplete specimens; it is 99.0 %. But on the full data 95.6 % of the excess is ordinary degeneracy
   (dead/constant units, shared-pattern groups) — the 1 000-specimen draft's "41 % unclassified" was the
   pre-Amendment-7 identifier and is withdrawn. The residual is small (median 0) and is an open item, not a discovery.
5. H-F4a-2 is not informative under mean-ablation (both sides 1.0).

## 5. Novelty searches (experiments/F_gauge/NOVELTY.md)
* Orbit partition of explanations: **OPEN** — closest 2606.04754 (neuron identifiability beyond
  permutation/scaling), 2506.13018 (parameter-symmetry survey), Méloux et al. ICLR 2025 (multiplicity, no gauge),
  2608.13754 / 2606.06267 (multiplicity under analytic variation). Nobody computes orbits of explanations — and per
  §4 the within-network version is empty, which is itself unclaimed.
* Cross-seed test (F4a): **FLAGGED** — the tools are convergent learning (Li et al. 2015) and re-basin /
  permutation weight matching (Ainsworth et al. 2022; 2402.04051; 2310.13397); closest results 2602.22600
  "Transformers Converge to Invariant Algorithmic Cores" and 2602.22968 "Certified Circuits". Position as
  gauge-orbit counting of *enumerated circuits* across seeds, calibrated by planted controls, with a negative result.
* Hidden-symmetry excess (F3): 2112.12982 (identifiability of deep ReLU nets — G complete outside degenerate cases)
  is the theoretical basis and, on the full data, the empirical picture agrees with it: the excess is the degenerate
  cases. The rate (99 % of trained small nets are degenerate in this sense) is not in the literature found.

## 6. Claimable / not claimable
**Claimable now.**
* Explanation multiplicity in small trained MLPs is not a coordinate artifact: structurally within a network
  (theorem-level), empirically across seeds (K(0.1)/N ≈ 1 with a residual whose meaning is calibrated by planted
  noise and an exact-relabel control).
* Across seeds the shared object is the algorithm (gate-composition string), not the weights: K_alg/N ≈ 0.39 under
  zero-ablation (133 cells), lowest for monotone functions.
* Nearly every trained small ReLU MLP has flat directions beyond permutation × scaling (99.0 %), and they are
  identifiable degeneracies (95.6 %): dead units, constant units, shared-pattern groups. A depth-growing residual
  (median 0, mean ≤ 3.3 directions) is unexplained.
* The gauge machinery itself (S1) and the enumeration (F2) are validated by kills to 1e-14.
**Not claimable.**
* Any K/N orbit ratio from F5 or F7 pooled orbits; the F6 algorithm table; anything about K vs width (F5b).
* Any claim of a *new* symmetry group in these networks — the unexplained residual is too small and unclassified.
* Any transformer-level orbit statement (F7 enumeration is fine; its orbit counts are the withdrawn test).
**Headline decision (for the principal).** The paper's headline cannot be "gauge explains multiplicity". The
defensible spine is: (i) the invariance audit (PAUSE 2 — F10/F11 as the mechanistic extension of Implementation
Invariance), (ii) the negative cross-seed orbit result with calibrated controls, and (iii) the degeneracy census of trained small
nets (99 % G_incomplete, 96 % identified). All three are final on the full data.
