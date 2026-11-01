# E2 — Gauge-invariant observables: FINDINGS

**Status: COMPLETE (2026-08-30).** 687 converged E1 networks (every `.npz` present
in `E1/results/networks` when the run started; E1 `_part2` was still writing —
35 of 36 cells are represented — RAND4 d2 w10 absent, RAND4/THR2_4 d3 and w10 cells partial),
3 sampling groups × 100 orbit representatives each, plus 840 untrained
same-shape networks as the null. Runtime 33 min (trained) + 79 min (untrained)
on 14 CPU workers. All numbers below are from `results/e2_invariance_table.csv`,
`e2_cells.csv`, `e2_scaling_verdicts.csv`, `e2_invariance_table_untrained.csv`.

## Verdicts on the pre-registered hypotheses

| Hypothesis | Criterion | Result |
|---|---|---|
| **H-E2a** perm-invariance of the candidate list | median CV < 1e-6 under `perm` for I1–I8 and every sorted reading | **PASSES.** Every one of them has median CV ≤ 1.5e-15 (β and TC(H1) on the truth table exactly 0). Float floor of the pipeline (I1 logits): 1.3e-15, max over networks 2.2e-15. |
| **H-E2b** TC scaling behaviour | per-unit `scale` CV > 0.05 for TC/β/PE; uniform-sweep exponent 2 for TC, 0 for β/PE; I8 exact | **PASSES** on TC (per-unit CV 1.4–3.0; fitted exponent 2.000 ± 1e-9 in 1586/1586 (layer, network) fits with all-positive TC, R² > 0.999) and on β/PE (exponent 0 / constant along the sweep in > 99.3 % of fits). I8 (per-unit standardised cloud) is exactly invariant in all three groups (CV 1.3e-15). **Partial exception:** PE(H0) and β0 of the box cloud have per-unit CV 0.015 and 0.048 — below the 0.05 "broken" line, so by the SPEC they are "approximately invariant", not broken. They are still not invariant (floor is 1e-15, and > 0.05 in the worst cells). |
| **H-E2c** spectra | `perm` exact, `scale` CV > 0.05 | **PASSES.** sv_layer: perm 1.3e-15, scale 1.0 (per-cell medians 0.81–1.44). |
| **H-E2d** sanity | I1, I2, I3 exact in all groups | **PASSES** (1.2–1.3e-15). |
| **H-E2e** indictment, raw-index reading | N1–N4 CV > 0.05 under `combined`; N5 and N6 raw agreement < 0.5 | **PASSES** for every item. CVs 0.5–2.6. Top-neuron raw agreement 0.10 (magnitude) / 0.17 (ablation); "gate of neuron i" raw agreement 0.28 (0.34 at w=4 → 0.22 at w=10 — it is ≈ 1/w, the chance of the same slot). `perm` breaks every raw-index quantity; `scale` additionally breaks N1, N4 and the magnitude-based top neuron, exactly as pre-stated. |
| **H-E2f** honest prior on attribution | grad×act, ablation effect, gates scaling-invariant per unit; sorted readings fully invariant | **PASSES.** Under `scale`: gradxact 1.3e-15, ablation 1.3e-15, gates agreement 1.00, separation counts agreement 1.00, ablation-based top neuron agreement 1.00. Sorted readings exact in all groups. |

## The table (core artifact)

CV = relative dispersion across 100 orbit representatives, median over 687
networks (bootstrap 95 % CIs over networks are in the CSV; they are narrower
than the printed precision everywhere). "Between-net CV" is the same
dispersion of the base value across the networks of a cell (median over
cells) — the vacuity check: every row has between-net CV ≥ 0.035, so nothing
in the invariant column is invariant merely by being constant.
"Broken by" lists which sampled symmetry alone drives the CV above 0.05;
"combined-only" means neither alone crosses 0.05 (b0_box, pe0_box are
approximate under `scale`: 0.048, 0.015).

### Invariant (median orbit CV < 1e-6 in perm, scale and combined)

| id | quantity | CV perm | CV scale | CV combined | between-net CV | broken by |
|---|---|---|---|---|---|---|
| I1 | `logits_tt` | 1.3e-15 | 1.3e-15 | 1.3e-15 | 6.0e-01 | none |
| I2 | `input_grad_tt` | 1.3e-15 | 1.3e-15 | 1.3e-15 | 8.1e-01 | none |
| I3 | `W_product` | 1.3e-15 | 1.2e-15 | 1.3e-15 | 6.0e-01 | none |
| I8 | `tc0_box_std` | 1.3e-15 | 1.3e-15 | 1.3e-15 | 3.9e-01 | none |
| I8 | `tc1_box_std` | 1.2e-15 | 1.2e-15 | 1.2e-15 | 5.2e-01 | none |
| N2 | `gradxact_sorted` | 1.3e-15 | 1.3e-15 | 1.3e-15 | 7.4e-01 | none |
| N3 | `ablation_sorted` | 1.3e-15 | 1.3e-15 | 1.3e-15 | 5.3e-01 | none |
| N6 | `gates_sorted` | 0.0e+00 | 0.0e+00 | 0.0e+00 | 7.0e-01 | none |

### Not invariant

| id | quantity | CV perm | CV scale | CV combined | between-net CV | broken by |
|---|---|---|---|---|---|---|
| I4 | `sv_layer` | 1.3e-15 | 1.0e+00 | 1.0e+00 | 2.1e-01 | scale |
| I5 | `tc0_tt` | 1.3e-15 | 2.0e+00 | 2.0e+00 | 6.6e-01 | scale |
| I5 | `tc1_tt` | 0.0e+00 | 3.0e+00 | 3.1e+00 | 1.6e+00 | scale |
| I5 | `tc0_box` | 1.3e-15 | 1.4e+00 | 1.4e+00 | 6.0e-01 | scale |
| I5 | `tc1_box` | 1.3e-15 | 1.4e+00 | 1.4e+00 | 7.7e-01 | scale |
| I6 | `b0_tt` | 0.0e+00 | 1.1e-01 | 1.1e-01 | 1.7e-01 | scale |
| I6 | `b1_tt` | 0.0e+00 | 1.1e+00 | 1.1e+00 | 1.0e+00 | scale |
| I6 | `b0_box` | 0.0e+00 | 4.8e-02 | 4.8e-02 | 1.1e-01 | combined-only |
| I6 | `b1_box` | 0.0e+00 | 4.8e-01 | 4.7e-01 | 5.3e-01 | scale |
| I7 | `pe0_box` | 1.5e-15 | 1.5e-02 | 1.5e-02 | 3.5e-02 | combined-only |
| I7 | `pe1_box` | 1.2e-15 | 1.1e-01 | 1.1e-01 | 2.0e-01 | scale |
| N1 | `act_mag` | 9.1e-01 | 1.2e+00 | 1.9e+00 | 1.0e+00 | perm+scale |
| N1 | `act_mag_sorted` | 1.1e-15 | 9.0e-01 | 9.1e-01 | 4.5e-01 | scale |
| N2 | `gradxact` | 2.3e+00 | 1.3e-15 | 2.3e+00 | 2.6e+00 | perm |
| N3 | `ablation` | 9.6e-01 | 1.3e-15 | 9.6e-01 | 1.1e+00 | perm |
| N4 | `in_norm` | 3.6e-01 | 1.6e+00 | 1.7e+00 | 3.9e-01 | perm+scale |
| N4 | `in_norm_sorted` | 1.4e-15 | 1.1e+00 | 1.1e+00 | 2.1e-01 | scale |
| N4 | `W_abs` | 6.8e-01 | 2.0e+00 | 2.6e+00 | 7.0e-01 | perm+scale |
| N5 | `top_neuron_mag` | 2.0e-01 | 4.9e-01 | 5.1e-01 | 3.7e-01 | perm+scale |
| N5 | `top_neuron_abl` | 6.3e-01 | 0.0e+00 | 6.3e-01 | 7.8e-01 | perm |
| N6 | `gates` | 8.5e-01 | 0.0e+00 | 8.5e-01 | 9.5e-01 | perm |
| N6 | `n_separations` | 6.8e-01 | 0.0e+00 | 6.8e-01 | 7.2e-01 | perm |
| N6 | `gates_index_agreement` | n/a | n/a | n/a | n/a | combined-only |

Reading the indictment rows: the *sorted* readings of grad×act, ablation and
gates are exactly invariant, i.e. the **multiset** of attribution values a
network produces is a well-defined property of the function, while the
**assignment of those values to neuron indices** is not. Raw activation
magnitude and raw weight norms are broken even after sorting (scale CV 0.9 /
1.1): they carry no function-level meaning at all.

## What this says (the honest framing)

1. **Permutation is never the problem for a summary statistic.** Everything
   that is computed from the activation *cloud* or the *spectrum* is exactly
   permutation-invariant in practice, at the 1e-15 floor, on 687 nets and 840
   random ones. Permutation only breaks *indexed* claims ("neuron 3").
2. **Scaling is the problem for topology.** TC of a raw activation cloud is
   not a gauge invariant: under the per-unit positive scalings that ReLU
   admits, its orbit CV is 1.4–3.0 — larger than its variation *between
   independently trained networks* (0.6–1.6). A TC value, or a TC ratio
   between two checkpoints, on raw hidden activations of a ReLU layer is
   therefore not a statement about the function the layer computes. This
   applies to the v1 library's TC instrument as used on raw activations, and
   to any persistence-based statistic (β, PE, RTD) computed in the raw
   coordinate system. Under *uniform* scaling TC is cleanly covariant of
   degree 2 (squared lifetimes), β and PE are invariant — but uniform scaling
   is a measure-zero subgroup of what the network's symmetries are.
3. **The repair is cheap and exact.** Standardising each unit over the probe
   inputs (I8) makes TC exactly invariant (1.3e-15) while keeping its
   between-network variation (0.4–0.5). This is the form Stage 2's
   `whitebox.invariants` should ship; the raw-cloud TC should not appear
   there without this normalisation. Caveat: standardisation is a
   *different* metric — it removes per-unit scale information along with the
   gauge, and whether that costs signal (e.g. for the v1 memorisation score,
   which lives on non-ReLU pooled features and transformer residuals with
   their own symmetry groups) is a Stage-2 question, not answered here.
4. **Singular value spectra of individual ReLU layers are not invariants**
   (scale CV ≈ 1) — the "spectrum of W_ℓ" is a coordinate-dependent
   quantity; only the product / the per-region end-to-end Jacobian (I2, I3)
   is.
5. **Attribution scores are better than their reputation, labels are worse.**
   Gradient × activation and single-unit ablation effects are exactly
   scaling-invariant per unit (the α cancels analytically, and the
   measurement confirms it to 1e-15). Their non-invariance is entirely the
   neuron *labelling*: the top-neuron-by-ablation identity is preserved under
   scaling (agreement 1.00) and destroyed under permutation (0.17). The claim
   "the most important neuron is neuron 7" is not well-defined; the claim
   "the largest ablation effect is 2.3 and there are three units above 1" is.
   Magnitude-based "top neuron" (the |activation| argmax) is not even
   scaling-stable (0.30).
6. **Per-neuron gate assignments** (E1's binarisation, and Méloux's
   separation counts) are scaling-invariant (agreement 1.00) and only
   relabelled by permutation; the per-layer multiset of gates is exactly
   invariant. "Neuron i computes AND" survives at the chance rate 1/w; "this
   layer contains an AND, two ORs and a constant" is a gauge invariant.

## Kill attempts and controls

* **Parameter distance moved (SPEC control 1).** Relative L2 distance of
  each representative from the base: median 1.35 (`perm`), 2.56 (`scale`),
  3.15 (`combined`); 5th percentile 1.22 / 1.84 / 2.48; minimum over all
  68 700 representatives of `scale`: 0.21. The orbit samples are far from the
  base network — the invariants are invariant across genuinely different
  parameter vectors, not across near-copies.
* **Vacuity (control 2).** Between-network CV ≥ 0.035 for every quantity
  (≥ 0.39 for every quantity in the invariant column). No "constant" flags.
* **Untrained null (control 3).** 840 random-init networks of the 36 shapes,
  same audit: **0 of 93 (quantity × group) verdicts differ** from the trained
  networks (`e2_invariance_table_untrained.csv`). Invariance here is a
  property of the group action, not of anything training did — as it should
  be; had any verdict differed the corresponding row would have been a
  statement about trained geometry, not about gauge.
* **Planted metrics (control 4)** and **broken-group control (5)**:
  `tests/test_e2.py` — logits pass (CV < 1e-10) and W0[0,0] fails in all
  three groups; a layer-0 permutation that is *not* propagated to layer 1
  (not a symmetry) is flagged by the audit (logit CV > 0.05). 13/13 tests
  green.
* **Hand-built networks (control 6).** E1's hand XOR3: I1–I3 exact; a
  duplicated ×3-rescaled sub-circuit changes TC(H0) of the truth-table cloud
  while leaving f unchanged (TC sees the extra coordinates; f does not) —
  TC is sensitive to the thing it is supposed to see.
* **Float floor (control 7).** I1 under `scale` (α over e^±3): median
  1.3e-15, max 2.2e-15. Every "exact" verdict sits at this floor.

## Negatives and limitations

* **Hidden symmetries not sampled.** E1's Jacobian-rank deficits (92 % of
  nets) are infinitesimal flat directions; a finite step along one changes
  f at second order unless it integrates to a true symmetry, which E1 did
  not characterise. Every verdict above is scoped to the known group G
  (permutations × positive scalings). A quantity invariant under G may still
  vary along hidden flat directions; that would only move rows from the
  invariant column to the non-invariant one, never the reverse.
* **TC(H1) on the truth-table cloud** is zero for 56 % of (net, layer)
  pairs (8 or 16 points rarely form a loop); its statistics are dominated
  by the minority with loops. The box cloud (200 points) has H1 in 97 % of
  fits and tells the same story.
* **The "approximate" rows** (β0 and PE(H0) of the box cloud, scale CV 0.048
  and 0.015) are the only quantities in the grey zone of the SPEC thresholds.
  They are *not* invariants (floor 1e-15; per-cell medians up to 0.03–0.07);
  PE is closer to invariant than TC because it is scale-free under uniform
  scaling and per-unit scaling perturbs it only through the changed
  lifetime *distribution*. Not claimed either way.
* **Scope.** Small ReLU MLPs on Boolean functions, with the truth-table and
  a uniform box as probe distributions. The group for transformers (with
  LayerNorm, attention-head permutations, residual-stream rotations) is
  different and larger; the tool (`check_invariance`) is general, the
  table is not.
* The agreement rate is exact equality, so it is only meaningful for
  integer-valued quantities (amendment 1 of the SPEC); CV is the statistic
  for everything else.

## Tool shipped

`e2lib.check_invariance(metric_fn, model, n_reps=100, group='combined')`
audits any `metric_fn(model) -> scalar | ndarray` by orbit sampling and
returns CV, agreement, verdict (`exact` / `approximate` / `not_invariant`
at the SPEC thresholds), max absolute deviation and the parameter distance
moved; `audit_metrics` does the same for a dict of metrics in one pass;
`sample_orbit` exposes the sampler. Stage 2's `whitebox.invariants` wraps
this. Tests: `E2/tests/test_e2.py` (planted invariant / non-invariant
metrics, broken group, hand-built nets, exponent-2 law, gate agreement, CV
definition).

## Literature

Symmetry facts: 2025 survey "Symmetry in Neural Network Parameter Spaces";
Grigsby, Lindsey, Meyerhoff, Wu 2022 (functional dimension; hidden
symmetries). Explanation multiplicity: Méloux et al. 2025 (the gate /
separation interpretations audited in N6). Persistence instruments: v1
whitebox (Ripser, Bauer 2021; TC as squared total persistence; persistent
entropy, Rucco et al. 2016; RTD, Barannikov et al. 2022 — computed in raw
coordinates and so subject to the same scaling caveat). The framing "an
interpretability claim is well-defined only if gauge-invariant" is the
programme's (PROGRAM.md); the sheaf / gauge-theoretic frameworks
(arXiv 2603.17141; "A Gauge Theory of Superposition") are the conceptual
neighbours and are not what is measured here.

Files: `results/e2_invariance_table.csv` (+ `_untrained`), `e2_cells.csv`,
`e2_per_network.csv` (+ `_untrained`), `e2_scaling.csv`,
`e2_scaling_verdicts.csv`, `e2_sweep_raw.csv`, `fig_cv_per_quantity.png`,
`fig_tc_scale_exponent.png`, `fig_agreement.png` (all 300 dpi).
