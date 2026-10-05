# PAUSE 0 (Run 3) — instruments on planted truth · 2026-09-20 15:16

**Plain English.** All seven instruments were verified on planted ground truth before any network was analysed, and the verification did
its job: it caught two faults in the finite-group search (A-8) before they could touch a result. Suite: **202 passed, 1 documented xfail**
(run 1: 143, run 2: +31, run 3: +28). Criterion "200+ tests green": met.

| instrument | planted truth | result |
|---|---|---|
| S1 input pullback `metric.pullback_input`, `gram_spectrum` | analytic Jacobian; Gram spectrum = nonzero spectrum of g | exact |
| S2 `symmetry.null_space` | f = sin‖x‖² on R⁸: null = 7-dim sphere tangent, ν = 7/8 | projector equal to 1e-9 |
| S3 `symmetry_finite.automorphisms` | regular 12-, 31-, 113-gons → dihedral order 2p; noisy 3-frequency Z113 embedding in scrambled order → order 226, 113-cycle, planted frequencies up to one unit; random cloud → 1; simplex → degenerate | pass, after A-8 (resolution guard + all-anchor refinement) |
| S4 `symmetry.killing` | radial and equivariant maps on R² → exactly one Killing generator, the rotation; identity → rotation + 2 translations, scaling r = 1/√2 | pass |
| S5 `symmetry_breaking.breaking_score` | output ignores a block → 0; depends only on it → 1; linear map → exact ratio | pass |
| S6 `symmetry.quotient_depth` | MLP with a swap-symmetric layer 2 → depth 2; generic T → none | pass |
| S7 `id_profile`, `connection_scale` | flat 3-plane → plateau; tight clusters → decay; two blobs of one class → the gap | pass |

Stated in SPEC3 before computing, all borne out: a symmetric distance matrix always has the reflection (no "Z_p only" plant exists); the
discovered generator is defined up to a unit; ν ≥ 0.9967 for any 10-logit image network; the secant pullback is layer-independent.
Faults in my own spec found later by the agents (recorded in FINDINGS_IMG.md, FINDINGS_P3.md): generators normalised by ‖A‖_F make the 0.1
Killing tolerance dimension-dependent and brightness incommensurate with the warps; the 0.05 null-fraction threshold is below the isotropic
expectation of a 32-row sketch; b_z = 1/3 at random initialisation lies above its own threshold.
Neighbouring work cited throughout: LieGG (Moskalev, Sepliarskaia, Sosnovik, Smeulders, NeurIPS 2022).
