# Phase 6 report — biological circuits, level D (2026-10-05)
Authors: Arjun Paradkar and Reva Paradkar.  Pre-registration `prereg/PHASE_6.md` (tag prereg-phase-6).  Full cells in `docs/record.md`.  The Fiete-lab collaboration, the lab's own RNN configuration and any lab data are not available on this machine; Arjun owns that coordination.

## Level-D validation on planted ground truth (N7; Phase 1 gate)
Response route (symmetry profiles of ratemaps -> module clustering -> lattice -> hexagonal characters): 3 planted modules recovered with spacing <= 1.7%, orientation <= 1.2 deg, hex R2 ~0.85 on five fresh seeds (PASS).  Connectivity route (spectral module count + Rips H1 of a weight-derived metric): module count exact on 10/10 fresh-seed networks, but the torus (b1 = 2) in only 13/15 modules on each of two fresh-seed sets (FAIL by the all-modules criterion, 87%); random wiring gives no bars, a ring gives one.  A shuffle-null bar rule was tried and rejected.
## Own path-integrating RNN (H6.1-H6.4): NOT TESTABLE
Three attempts (up to 512 units, 40,000 steps, difference-of-Gaussians place code, with and without a conformal-isometry loss): path integration is learned (0.06-0.08 m error on a 2.2 m arena) but no grid-like units emerge in either condition (0 of ~506; boundary/band cells).  By the registered rule the confirmatory runs were not launched.  Consequently the circularity controls (no-conformal, memorizer, connectivity route on a trained network, formation timing) have no RNN evidence.  This is a statement about our RNN and ~1 GPU-hour budget.
## Real rodent recordings, Gardner et al. 2022 (H6.5; no loss imposed by anyone)
| cell | result | verdict |
|---|---|---|
| H6.5a blind module recovery from ratemaps (>= 2 of 3 files) | rat Q 2/2 modules (ARI 0.95) PASS; rat R day 1 1/3 (ARI 0) FAIL; rat S 1/1 PASS (trivial) | PASS (2/3, one trivial) |
| H6.5b torus per module (>= 4 of 6) | called in 5/6: rat R modules 1-3 and rat S clearly (top H1 0.8-1.1 vs null max ~0.4, H2 above null); rat Q module 1 marginal (0.387/0.360 vs 0.350), rat Q module 2 not called | PASS (5/6) |
| H6.5c ring (head direction) | not tested (no small head-direction dataset located) | - |
| H6.6 kappa on recorded populations | not tested | - |
Caveats: the shuffle-null criterion is lenient (67-93 H1 bars exceed the null maximum in the clear modules; lifetimes do not drop after the second bar), so these are not clean b1 = 2 certificates; coarse-spacing modules cannot be lattice-fitted in a 1.5 m arena, so the torus calls are not tied to a recovered lattice group; the two rat Q modules are small.
## Not done
Insect head-direction connectome in v2 (data outside this repo); rodent head-direction ring; formation of structure in a trained grid network.
## Compute
~2 GPU-hours (RNN pilots), ~1.5 CPU-hours (rodent topology), no cloud spend.
