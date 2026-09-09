# F4/F5 reformulated (pre-registration, 2026-09-07, after AMENDMENTS #8; before any run of this code)

_Naming: every run in this document is a horse from the Tracen roster (`experiments/HORSES.md`); the handbook codes below are the pre-registration labels and appear here only as such._

Inputs: `results/explanations.csv` (minimal circuits per specimen × criterion, handbook-interpretable
flag, Méloux count, functional key) and the trained specimens. Frozen: F1–F3, F7–F11.

## F4a — Cross-seed orbit test (PRIMARY)
**Object.** For a cell (function, depth, width) and criterion, the set of minimal circuits of every
converged seed. For two circuits C_A (seed A) and C_B (seed B) with the same per-layer unit counts,
the *gauge residual* r(C_A, C_B) = min over g ∈ G of the relative Frobenius distance between the
scale-normalised sub-implementations (each unit's incoming (w, b) unit-normalised with the norm
pushed downstream; output row unit-normalised), with the permutation part found by layer-wise
Hungarian matching on the normalised incoming vectors (inputs in fixed order; deeper layers in the
matched order of their parents). r ∈ [0, 2]; r = 0 iff C_B = g·C_A exactly. Circuits with different
layer sizes get r = ∞.
**Functional-dimension correction.** Dead units never enter minimal circuits (they are ablated for
free). Units of a circuit that share one activation pattern on the truth table (F3's shared-pattern
groups) are jointly linear there: before matching, each such group is replaced by its effective map
M = Σ_u w_out[u] (w_u, b_u) (one "effective unit" per group, rank-normalised), so two circuits that
differ only by how a linear group is split are not counted as distinct.
**Statistics.** Per cell and criterion: (i) the distribution of r over all cross-seed circuit pairs;
(ii) K(τ)/N for τ ∈ {1e-6, 1e-3, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5} by union-find over pairs with r < τ,
N = circuits pooled over seeds; (iii) the *algorithmic* count K_alg = number of distinct handbook
gate-composition strings (unit labels stripped, per-layer gate multisets) among interpretable
circuits, and K_fkey = distinct functional keys (E1's binarised-truth-table multiset) among all
minimal circuits — both are exact gauge invariants that ignore weights.
**Pre-registered hypotheses.**
* H-F4a-1 (mechanism unique up to renaming): at τ = 0.1 the median cell has K(τ)/N ≤ 0.5.
* H-F4a-2 (weights vs algorithm): K_alg/N ≤ K(0.1)/N in the median cell (algorithms are fewer than
  weight-orbits: multiplicity across seeds is mostly *the same algorithm in different weights*).
* Reported regardless: the full r distribution against two controls (below) and K(τ)/N curves.
**Kills (must pass before any verdict).**
* K1 (positive control, exact): seed A vs g·A for random g (100 specimens): the matched circuit
  pair must have r < 1e-9 and be joined at every τ.
* K2 (positive control, approximate): seed A vs g·A + Gaussian weight noise of relative size δ ∈
  {1e-3, 1e-2, 3e-2} (circuits re-enumerated on the noisy net; only pairs whose circuit unit sets
  correspond): r must track δ (r ≈ δ within a factor 3), so τ has a known meaning.
* K3 (negative control): circuits of the three planted XOR3 algorithms (A, B, C in
  `flib/planted.py`) pairwise: r > 0.3 for every cross-algorithm pair.
* K4 (null): r between circuits of *different target functions* with the same layer sizes must not
  be smaller than the same-function cross-seed r (KS test, p reported); if same-function pairs are
  not closer than different-function pairs, the cross-seed test has no signal and says so.
**Figures.** `fig/F4a_residual_hist.png` (same-function vs different-function vs planted controls),
`fig/F4a_KN_vs_tau.png` (per function), `fig/F4a_Kalg_vs_width.png`.

## F4b — Approximate stabilisers within a network
*(Definition refined 2026-09-07 before the first F4b run, after the first kill attempt: a per-unit
clustering key cannot identify gauge copies because copies read from different parent units; the
equivalence must be hierarchical, i.e. the same gauge residual as F4a applied within one network.)*
Two minimal circuits of the SAME specimen are ε-equivalent iff their gauge residual r (F4a's
definition, with the functional-dimension collapse) is < ε; K_within(ε) = number of union-find
classes for ε ∈ {1e-6, 1e-3, 1e-2, 0.05, 0.1, 0.2}. Report K_within(ε)/N per specimen and its
median per (cell, criterion); this is the ε-stabiliser structure of the network (Hydra-style
approximate duplication shows up as K_within(ε) < N at small ε).
Pre-registered: at ε = 1e-2 the median K_within(ε)/N ≥ 0.95 (trained small nets have essentially no
approximate stabilisers; if this FAILS, within-network redundancy is partly approximate gauge).
**Kills.** (i) The planted k = 1 network (two gauge copies of one algorithm): its two planted
circuits must have r < 1e-9 (one class at every ε). (ii) A trained specimen with a deliberately
duplicated unit (unit copied into a spare slot, outgoing weight split): its minimal circuit's
effective sub-implementation must match the original specimen's minimal circuit with r < 1e-9
(the duplication is invisible after the functional-dimension correction).

## F4c — Interpretation-level gauge
For circuits with 1 < n_meloux ≤ 20 000, enumerate the Méloux interpretations' bit matrices and
count classes under per-unit negation (bits complemented, gate tables re-derived). Report
n_classes / n_meloux per circuit and its median. Pre-registered expectation stated honestly: with
Méloux's one-sided thresholds (1 = above), negating a unit is the same threshold read the other
way, so we expect **no collapse** (ratio 1); a ratio < 1 would mean interpretations differ only by
orientation conventions. Kill: a hand-built circuit whose unit has two interpretations that are
complements must give n_classes = 1.
