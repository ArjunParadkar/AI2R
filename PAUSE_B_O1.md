# PAUSE B — T1, all seven cells (Block O, FINAL_SPEC §9 day 4) — and PAUSE C, opened / not (day 5)

_Written 2026-09-14 22:30, not stopped at. Sources: `experiments/O_opener/results/O1_2_summary.json` (aggregate), `results/grid.md` /
`grid.csv` (O5 Symboli Kris S, the headline grid for T1), `results/open/*.json` (every cell with criterion, null, provenance),
`nishino_flower_detections.csv`, `fine_motion_P4.csv` / `_P5.csv`, `AMENDMENTS.md` (consolidated ledger). PAUSE_A_O1.md holds the
amendment story and the O1.1 specimen table._

## 1. The pre-registered O1.2 verdict (SPEC_O §O1.2)

> Pass: all seven cells certified on ≥ 4/5 clean seeds; R1/R2/R5 pass; γ_cov < 0.05.

**Not met** (`O1_2_prereg_pass = false`). Per cell over the 5 clean seeds (certified / 5): P0 5, P1 5, P2 2, P3 0 (no bias to
detect on clean specimens; on the planted specimens 0/5 under the pre-registered detectors, 1/5 under probe v3 with a false positive
on the inert control), P4 0, P5 0, P6 5, P7 0; R1 2, R4 0, R5 2, R6 0, R7 5; γ_cov median **0.899** (range 0.883–0.941). Ring (9
seeds): gauge 9, P1 0 (one dominant H1 bar on every seed, coordinate agreement 0.73–0.98 < 0.99), P2 8, R2 0.

## 2. What is certified on T1 — the positive results

* **Gauge.** Every exact symmetry of the toy transformer (head permutations, per-head GL(d_h), MLP permutations and scalings, LN
  rescalings) and of the ring RNN (unit permutations × positive scalings) verified to float32 round-off (max relative deviation
  ≤ 9e-7) on every specimen.
* **P0** certified 15/15 grok (attribution-graph treewidth bound ≤ 8; Q below the rewiring null — the graphs are sparse stars).
* **P1 — what it learned** certified 10/10 grokked specimens: the function-level irrep inventory (κ_f 0.94–0.97; 16–24 frequencies
  above the off-diagonal noise floor; K identical on all 300 gauge samples; CV < 1e-3; row-shuffle null κ_f < 0.3), and **not**
  certified on 5/5 memorisers (empty inventory, κ_f ≤ 1e-4) — K-O-3 satisfied. The pre-registered embedding-level estimator (κ_emb
  > 0.9) fails on every grokked specimen (0.00–0.59) and is reported in every cell.
* **P6 — memorising vs learning** AUC **1.00** (κ_f) over 15 specimens, gap 0.94; the pre-registered κ_emb gives AUC 0.85.
* **P2 — why this decision** invariant on every specimen (CV ≤ 6e-5 except one memoriser at 0.048, an artifact by the K-O-2 rule);
  the D1 anchor holds on the constant-pattern region (local IG r 0.95–0.99) on 9/15 grok and 8/9 rings, and never with a zero baseline
  (global IG r 0.04–0.18): certified 2/5 clean, 4/5 planted, 8/9 ring — clean seeds miss 0.95 by 0.004–0.03.
* **R1 — rebuild from irreps** (no weights copied; K and the diagonal amplitudes only): argmax accuracy **1.000 on all 10** grokked
  specimens; logit r 0.966–0.986; passes the > 0.98 band on 3/10 (clean s2 0.984, s3 0.981, planted s0 0.986). The scale-free K-O-1
  kill passes on all 10 (certified r above the 99th percentile of 100 random same-size sets, 0.43–0.59).
* **R7 — negative control** certified 10/10: the uncertified frequencies rebuild to r 0.01–0.11; certified subsets degrade monotonically
  (Spearman 1.0).
* **Cross-seed count** (function level): one algorithm class ("Fourier/character") on 5/5 clean seeds, five distinct frequency sets
  (|π₀| of named algorithms = 1; the frequency set is seed-specific).
* **Glue** sanity gate passed (H1 = 0) on 3/5 clean and 5/5 memorisers; **fails on 2/5 clean and 4/5 planted** (H1 2–3) — the Čech
  instrument at these atoms/τ is not validated on a single-mechanism task (§2.12 gate), recorded as such.
* **Ring P1 object**: exactly one dominant H1 bar on the zero-velocity attractor sample of every seed (ratio ∞ on 7, 43 and 279 on the
  other two), nulls ≈ 1, gauge-identical diagrams.

## 3. What is not — the negatives, each with its diagnosis

* **P7** 0/15: irrep appearance tracks the accuracy jump (Δ 0.04 on every grokked seed) but the Laplacian spectral gap never "opens"
  (flat 0.24–0.31, n_small = 1 throughout); the frequency-ordering rule (Spearman of appearance vs amplitude) is inconsistent in sign
  across seeds.
* **R4 / γ_cov** 0/10: after projecting the MLP cloud on the degree-2 trigonometric basis of the certified irreps (r² 0.76–0.89), the
  residue is almost entirely structured (γ_cov 0.88–0.98; both nulls far below). The MLP is not a linear function of its certified
  features; one residue-guided pass (R8) moves γ by ≤ 0.06. Degree of opening 1 − γ_cov = **0.03–0.12** by the pre-registered metric.
* **R6** 0/10: removing one Fourier component from the token embedding is rescaled per token by LayerNorm, so the model-side ablation
  is not the rebuild-side one (r 0.04–0.84).
* **R5** follows R1 (2/5 clean); the perturbation/OOD blocks are not applicable on the finite integer domain.
* **P3** not certified (O1.3): zigzag false-detects on every null, eigenflow locks onto the initialisation transient, the pre-registered
  probe guard never fires; probe v3 lands within 5 % on planted_s2 (0.03) and 0.06 on s0, but false-detects on the inert-shortcut
  control (p 0.028). The planted shortcut itself is adopted on only 2/5 seeds (s3 never, s4 in a 3-checkpoint flicker), so the
  specimen design gives 3 of 5 planted seeds no ground truth.
* **P4** 0/27 (O1.4): diffusion OOD distance at chance on grokked specimens (AUC 0.41–0.64) while max-softmax sits at 0.99–1.00; on
  memorisers d_M beats softmax but loses to its random-projection null; `spectral.out_of_manifold` inverts on 7/9 rings (Nyström
  image collapses on far points — failure boundary realised, guard needed).
* **P5** 0/27: with the failure radius fully resolved by bisection (v2), ρ = ‖Q_h‖·Lip carries no fragility information (|ρ| ≤ 0.10 on
  14/15 grok; ring sign unstable across seeds), while the softmax-margin baseline it was to beat clears the SPEC threshold on all 5
  memorisers (−0.60 to −0.68).
* **Ring P1 agreement** 0/9 at the 0.99 band (0.73–0.98): the de Silva coordinate on 1500 cues is the limiting estimator (the
  network's own read-out agrees with the cue at 0.99–1.00). **R2** 0/9: the memoryless integrator fitted to on-ring pulse responses
  integrates (error 0.28–1.47 rad vs the model's 0.05–0.08) but is 5–22× worse than the network — the reduced model θ ← θ + γ(θ)ω
  does not capture the network's integration precision (three constructions tried, all pre-registered, all recorded).

## 4. PAUSE C — opened or not

By FINAL_SPEC §1.1 and Constitution §12.9: T1 is **not completely opened**. It is opened to degree 1 − γ_cov ∈ [0.03, 0.12] on the
pre-registered residue metric, with P0, P1, P6 and R7 certified on every grokked seed, P2 and R1 certified on a minority, and P3, P4,
P5, P7, R4, R6 failed under every pre-registered and every pre-registered-amended estimator. The strongest positive statement the
evidence supports: *the function computed by every grokked pre-LN Z113 transformer lies to 94–97 % of its logit variance in the span of
the characters χ_k(a+b−c), its certified irrep inventory rebuilds the argmax exactly and the logits to r ≥ 0.97, the inventory is
gauge-invariant, and the memorising controls certify nothing* — a certified-and-rebuilt P1/P6/R1 result. The neuron-level residue,
the timing of representation change, bias onset, OOD distance and fragility are **not reached** by the instruments as built.

The three failures that are most likely instrument defects rather than facts about the models, and would be the first targets of a
next round (not run here — the thresholds and estimators stand as pre-registered): the Čech atoms/τ (sanity gate fails on 6/15), the
Nyström collapse in `out_of_manifold`, and the de Silva coordinate resolution on the ring.

## 5. Bookkeeping

Commits today: O1.1 trainer + diagnosis, O1.2/O1.3/O1.4 runners and results, VALIDATION.md (+38 rows), AMENDMENTS.md
(consolidated), PAUSE_A_O1.md, this file, CLAIMS.md rows for the Block O modules. Still owed: O2 (GPT-2, day 6), O3/O4 (HPCC —
runbook and Azure launcher ready, blocked on access), the F2 d3w10 tail (235 specimens left, local), push (remote to be named).
