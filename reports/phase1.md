# Phase 1 report (2026-10-04) — estimators and theory skeleton
Authors: Arjun Paradkar and Reva Paradkar.

## Result: gate NOT met. 28 of 32 numerics cells pass; 4 fail on the frozen tolerances.
Tolerances were frozen in `numerics/TOLERANCES.md` (sha256 checked at every run) before any estimator existed. They were not changed. All numbers below are attempt 1 (`numerics/results/*.json`, each with seed + config hash); full rows in `docs/record.md`.

| block | cells | pass | failed cells |
|---|---|---|---|
| N1 Jacobian / Hutchinson / sketches / gradient-free / Fisher-Rao / rewriting-invariance | 14 | 14 | - |
| N2 graph geodesics | 4 | 3 | T-N2-01 |
| N3 delta-symmetry search | 4 | 4 | - |
| N4 Killing solver | 5 | 3 | T-N4-02, T-N4-03 |
| N5 persistent homology | 3 | 2 | T-N5-02 |
| N6 reproducibility | 3 | 3 | - |

## The four failures
1. **T-N2-01** mean relative geodesic error 0.0326 (ambient) / 0.0328 (chart) vs limit 0.03; p95 0.061 passes. Direction quantisation at k=12 (documented in `geodesic.py`) is the expected cause; the plateau in n was stated up front. Marginal.
2. **T-N4-02** sphere: solver finds a 1-dim Killing space, not 3 (gap 41.9, subspace error 3.7e-3). The rotation about the axis is found; the two fields involving cot(theta) are not resolved at n=600. Measured convergence is algebraic (order 3.1), not spectral.
3. **T-N4-03** rotated image manifold: null dim 0, subspace error 0.061. The reference fields are exactly Killing for the test metric (residual 4e-11), so this is solver resolution (3D chart, n=500). Post hoc: a wider RBF shape parameter gets the subspace error to 1.6e-4, which suggests the span is recovered and the failure is in the absolute null-dimension rule; not claimed as a pass.
4. **T-N5-02** certified Betti numbers: circle and two-circles pass; S^2 (n=300) and Clifford torus (n=500) are not certified because the certificate needs bar life > 4 d_H and d_H at these sizes is about 0.4.

Consequence: **the Killing solver and the certified-homology step may not be used on real models** until they pass a gate. Phase 3 (Killing recovery on rotated MNIST) is blocked on N4. Options for Arjun: (a) fix the solver (more centres, non-stationary shape, relative null-dimension rule) and re-run as a labelled attempt 2 under a new, openly post-hoc tolerance file; (b) accept N4 as exploratory only and drop the Killing-dimension claim from the paper. N1, N3, N6 and the discrete-symmetry/N2 machinery that Phase 2 actually uses are unaffected.

## Theory
`theory/core.pdf` (5 pp): M0-M5 with proofs where I have one, cited results marked as cited, conjectures stated with the experiment that would refute them, and a status table. Nothing is called a theorem and no proof has been checked line by line by anyone but the author. Proved here: Fisher-Rao invariances (M0.1), the counterexample and an exact condition for metric-vs-output equivariance (M1.4, sufficiency open), approximate closure with words of length k within k*delta (M2.1), explicit Ulam-type stability for Z_n (generator within pi*eps/2, action within (n/2)(1+pi/2)*eps; n-dependence is from the elementary proof, not shown necessary), Taylor remainder (M2.3), exact heavy-ball drift laws (M4.1), the equivariance gap inequality R(f) >= R(Qf) + (mu/2) Gamma(f) (M5.1). M2.4 is a conditional reduction to Xu-Mannor robustness, not an independent proof. M3.2-3.4 are not formalised. Marsden-Weinstein reduction (M4.3) is recorded as ornamental unless H-symp passes. `proofs/check_m0..m5.py` all pass as sanity checks.

## Process
The earlier terminal deaths were the numerics suite itself: T-N4-03 used reverse-mode Jacobians over 2809 pixels (~60 GB intermediates). Fixed with forward mode. Long jobs now run through `scripts/wbrun` (systemd unit, MemoryHigh 12G / MemoryMax 16G). Full suite now takes 5m46s.

## Compute
CPU only, about 25 minutes total across attempts; no GPU. Phase 2 estimate and budget approval request to follow before any launch.

## Not done
Bootstrap CIs/Holm (planned, Phase 2), v1-table re-run with the Fisher-Rao metric (Phase 2), Gromov-Hausdorff resampling test of M2.2 (Phase 2), proof checking by a second person (needed before any word "theorem").
