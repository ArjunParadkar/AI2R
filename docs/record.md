# record.md — every pre-registered cell (hypothesis | threshold | result | pass/fail | deciding null)

| cell | hypothesis | threshold | result | verdict | null |
|---|---|---|---|---|---|
| P0.E1 separation (Runs 4-6) | complete separation of 10 generalizers vs 7 memorizers, one instrument | no overlap | no overlap; one-sided exact permutation p = 1/C(17,7) = 1/19,448 = 5.1e-5 (replaces "p < 1e-5 under exchangeability"; not combined across instruments) | PASS (correct p) | permutation of labels |
| P0.E2 formation (P7.a, Runs 5-6) | generator accepted at/before jump | by run, not checkpoint | 5/6 generalizer seeds at/before (leads 0,0,+1,+4,0), 1/6 one checkpoint after (-1); decoys 0,0; 0 of 4 swept memorizer runs (138 checkpoints, descriptive). Binomial "0/102, p<1e-30" RETRACTED (pseudoreplication) | PASS as coincident marker; p-value retracted | memorizer sweeps |
| P0 denominators | reconcile 102 vs 204 | - | 102 = Run 5 only (2x51). Spec-compliant swept memorizer checkpoints on disk: Run 4 36 + Run 5 102 = 138, 0 accepted. 204 NOT reproducible from results/ | corrected to 138 | - |
| P0 P5 natural failure caveat (Run 7) | held-out test acc window 0.70-0.90 | kappa AUROC > softmax 3/3 | test acc 0.09 / 0.785 / 0.880 (v1 text said 0.51-0.88: wrong; 0.51 is acc at generator acceptance in Run 5); generator found at scored ckpt on 1/3; AUROC 0.983/1.000/1.000 vs 0.563/0.852/0.900; random 0.45/0.44/0.49; n=3 seeds, no CI | PASS by letter of SPEC7, claim narrowed | random transformations |
| P1 N1 (T-N1-01..14) | Jacobian estimators vs manufactured solutions | TOLERANCES.md (frozen, sha256 checked each run) | 14/14 pass (attempt 1) | PASS | analytic J |
| P1 N2 T-N2-01 | graph geodesic on S^2, n=4000, k=12: mean<=0.03 and p95<=0.08 | frozen | ambient mean 0.0326 / p95 0.0612; chart mean 0.0328 / p95 0.0613. p95 passes, mean misses by 0.0026 | **FAIL by the letter** (threshold stays 0.03) | exact great-circle distance |
| P1 N2 T-N2-02..04 | scaling in k,n; chart/ambient consistency; metric axioms | frozen | pass | PASS | - |
| P1 N3 (T-N3-01..04) | delta-symmetry search incl. non-abelian, calibration | frozen | 4/4 pass | PASS | random-relabelling null |
| P1 N4 T-N4-01, 04, 05 | torus Killing space; negative control; self-consistency | frozen | pass | PASS | - |
| P1 N4 T-N4-02 | sphere band, n=600: null dim 3, gap>=100, err<=1e-3 | frozen | null dim 1, gap 41.9, subspace err 3.7e-3; observed convergence order 3.10 (err 7.4e-2 -> 2.8e-3 for n=100..800) | **FAIL** (dim, gap, err) | exact so(3) |
| P1 N4 T-N4-03 | rotated/translated image manifold, n=500: null dim 3, gap>=20, err<=1e-2 | frozen | null dim 0, subspace err 6.1e-2 (exact fields satisfy L_v g = 0 to 4e-11, so the test metric is right; the solver is under-resolved in 3D at n=500) | **FAIL** (dim, err) | exact Killing fields |
| P1 N5 T-N5-01, 03 | bottleneck stability; subsampling bound | frozen | pass (100% of trials) | PASS | - |
| P1 N5 T-N5-02 | certified Betti counts: circle, two circles, S^2, Clifford torus | frozen | circle b1=1 PASS; two circles b0=2,b1=2 PASS; S^2 b2: 0 (needs 1) FAIL; torus b1: 0 (needs 2) FAIL; eta_c ~ 0.79 at n=300/500 so the certificate (life > 4 d_H) cannot certify at these sample sizes | **FAIL** (sphere, torus) | dense reference sample |
| P1 N6 (T-N6-01..03) | bitwise reproducibility, result records, deterministic flags | frozen | 3/3 pass | PASS | - |
| P1 theory checks (proofs/check_m0..m5) | numerical sanity of M0-M5 statements | stated in each script | all pass; these are sanity checks, not proofs | PASS (sanity) | - |

Post hoc (exploratory, not verdict-changing; logs/sweep.log): shape-multiplier sweep for the N4 solver. Images n=500: multiplier 6 gives subspace err 1.6e-4 but only 2 singular values below the absolute 1e-3 cutoff (third is 2e-3), so null-dim still fails under the frozen rule. Sphere: multipliers 2-8 all give dim 1; the two off-axis fields stay unresolved (sigma 0.03-0.14). Process note: attempt 1 of T-N4-03 used jacrev for the 2809-pixel Jacobian and needed ~60 GB (the cause of the OOM kills on 2026-10-04); switched to jacfwd (identical exact Jacobian), result above is from that.

## Phase 1 attempt 2 (POST HOC, 2026-10-04; does not overwrite attempt 1)
Protocol (numerics/n4_attempt2.py): tune (RBF shape multiplier, centre fraction) only on three geometries that are not the failed cells (flat cylinder, hyperbolic half-plane patch, flat T^3); chosen shape 6.0, centres = n. Apply once to the frozen cells with the same error thresholds and gap thresholds but reading the null dimension by a gap rule (largest sigma ratio among the first 8, accepted if ratio >= gap_min and sigma <= 0.05) instead of the absolute cut 1e-3.
| cell | result | verdict |
|---|---|---|
| T-N4-03b images, n=500 | dim 3, gap 81.7 (>=20), subspace err 9.0e-5 (<=1e-2) | PASS (post hoc; attempt 1 stays FAIL) |
| T-N4-01b torus | dim 2, err 0 | PASS |
| T-N4-04b negative control | dim 0, sigma_min 0.23 | PASS |
| T-N4-02b sphere band, n=600 | dim 1 (needs 3), gap 2041, subspace err 8.5e-3 (needs <=1e-3); the two fields with cot(theta) coefficients stay at sigma 0.07 | **FAIL** (also post hoc) |
Consequence: the Killing solver is validated for constant-structure / rotation-like algebras on image manifolds (what Phase 3 needs: rotated MNIST) and NOT for curved-band fields with non-constant coefficients. N2-01 (mean 0.0326 vs 0.03) and N5-02 (S^2/torus certification at n=300/500) are unchanged failures; Betti certificates are used only where the certificate bound is met.

## Phase 6 numerics gate N7 (level D routes validated on planted populations with exact ground truth; 2026-10-04)
| cell | statement (thresholds fixed in numerics/test_n7_bio.py before the fresh-seed run) | result | verdict |
|---|---|---|---|
| T-N7-01 response route | 3 planted hexagonal modules (spacing 0.45/0.65/0.9, orientations free), fresh seeds 10-14: module count exact, spacing rel err <= 0.05, orientation err <= 5 deg, hex-template R2 >= 0.7 | all five seeds pass (spacings within 1.7%, orientations within 1.2 deg, R2 ~0.85, silhouette 0.98) | PASS |
| T-N7-02 connectivity route (attempt 1, ratio rule 2.0, seeds 10-14) | k modules exact; every torus module b1 = 2; random wiring b1 = 0; ring b1 = 1 | k 5/5; torus 13/15 modules (one gave 3 bars, one gave 0); random 0 PASS; ring 1 PASS | **FAIL** (87% modules) |
| T-N7-02 attempt 2 (shuffle-null threshold, POST HOC, seeds 20-24) | same | torus 0/15 (shuffled-weight nulls produce long bars too); ring split into 3 modules | **FAIL, rule rejected** |
| T-N7-02 attempt 3 (ratio rule 1.5, POST HOC, fresh seeds 30-34) | same | k 5/5; torus 13/15 modules (seed 30: [0,5,2]); random 0 PASS; ring 1 PASS | **FAIL by the all-modules criterion; 87% module-level accuracy twice** |
Conclusion carried into Phase 6: the connectivity route recovers the number of modules reliably and the torus in ~87% of modules on planted wiring; any H6.2 claim must be stated at module level with this error rate, not as "recovered".

## Phase 2 pre-registration freeze (2026-10-04)
Commit 3d448ff1622cc8072cd8fc6babcde66bfe23efd4, annotated tag prereg-phase-2 -> d831a06f1199ce0bae27b238b0723b816f93ff00 (unsigned: no GPG key). prereg/PHASE_2.md lists sha256 of the grid (960 configs), zoo, worker, stats and analysis code. Confirmatory run launched after this entry.

## Phase 4 part 1 — blind circular structure in language models (prereg-phase-4, commit 9c2db1a; run 2026-10-04/05; results/phase4/circ_*.json)
Unit = one pretrained model (6 models; no seed replication).  Headline = max-over-layers adjacency, p = max-over-layers permutation null.
| cell | rule | result | verdict |
|---|---|---|---|
| H4.1a days | score >= 0.85 and p < 0.01 in >= 4/6 models | 6/6 models: score 1.0, p 0.0027-0.009 (gpt2, gpt2-medium, pythia-70m, pythia-410m, Llama-3.2-1B, Qwen2.5-1.5B-Instruct) | PASS |
| H4.1b months | same | 5/6: score 1.0, p = 0 ; pythia-70m 0.67 (p 0.0014, below the 0.85 bar) | PASS (5/6) |
| H4.1c controls | numbers/colours/animals: fraction of models with p < 0.01 <= 1/6 | numbers 0/6 (Llama score 1.0 but p = 0.014; others 0.57-0.71), colours 0/6, animals 0/6 | PASS by the registered rule |
| H4.1d Holm (30 tests) | reported next to raw | Holm-adjusted: months 6/6 < 0.05 (0.001 x5, pythia-70m 0.036); days 0/6 < 0.05 (adjusted 0.066-0.17: a 7-item set has only 360 distinct cyclic orders so its permutation p-value is floored near 0.003 before correction); controls: smallest adjusted p 0.254 | days: PASS on the registered raw rule, NOT significant after Holm; months PASS both |
Caveats (all disclosed): the best layer is layer 0 (the input embedding) in 10 of the 12 passing cases, so the cyclic order is present in the static token embeddings, not computed in context.  The adjacency score cannot separate a line from a circle (an optimal tour closes the endpoints).  POST HOC check (bench/llm/circular_posthoc.py, never upgrades a verdict): closure ratio ||z_last - z_first|| / mean adjacent edge, at layers with adjacency >= 0.85: days 1.07-1.19, months 0.91-1.22 (circle-like, ~1), the single number-word layer that reached 0.85 (Llama layer 1) 1.48 (higher than every day/month value).  So the structure recovered for days/months is closed, and the one control hit is less closed; this is a post hoc separation, not a pre-registered one.
Not done in Phase 4: formation across Pythia checkpoints, hallucination/knowledge-boundary vs semantic entropy/SelfCheckGPT, LLM memorization detection, extractor lambda on LLMs, transforms library (see below once run).

## Phase 4B H4.2 — formation of cyclic structure across Pythia training (prereg-phase-4b; descriptive; ONE training run per size, no across-run test possible; results/phase4/pythia_*.json)
Rule: first checkpoint with adjacency >= 0.85 and max-perm p < 0.05.
| model | days | months | numbers (control) |
|---|---|---|---|
| pythia-70m | first = step 512 (score 1.0, p 0.015) but it is NOT stable: 0.43 at steps 256 and 1000, 1.0 from step 2000 on | first = step 4000 (0.83 at step 1000/2000 is below the bar) | never >= 0.85; max 0.71 (p 0.12) |
| pythia-160m | first = step 2000 (1.0, p 0.011), stable | first = step 2000 (1.0, p = 0), stable | never >= 0.85; max 0.71 (p 0.23) |
Reading: before step ~512 neither days nor months are recovered (adjacency at chance 0.17-0.57, p >= 0.5); the structure is in place by step 2000-4000 of 143000 (1.4-2.8% of training) and persists to the end, apart from pythia-70m's final months score (0.67).  Single runs; the numbers control is clean at every checkpoint.  No claim about what drives the transition.
H4.3 (hallucination, prereg-phase-4b): the first launch FAILED on CUDA OOM (GPU full of sibling jobs) for both models before producing any result; it is re-run, no result yet.
