# TOLERANCES.md — Phase 1 numerics (N1–N6)

Authors: Arjun Paradkar and Reva Paradkar.  Written 2026-10-04, BEFORE any numerics test was run
(no estimator in `whiteboxtopo/numerics/` had been written when this file was frozen; its sha256 is
stored in `numerics/TOLERANCES.sha256` and every test run re-checks it).  Thresholds never change after this.
A failed cell is a result: it goes to `docs/record.md` and the report, the threshold stays.

Conventions: float64, CPU only, seeds fixed in each test (the seed is part of the cell).  "rel" = divided by the
max-abs entry (or Frobenius norm where stated) of the exact reference.  Test-IDs are `T-Nk-xx`.

## N1  Jacobian estimators
| ID | statement (manufactured solution) | tolerance |
|---|---|---|
| T-N1-01 | exact autograd Jacobian vs hand-derived analytic J (affine, polynomial/trig map, hand-backpropped tanh MLP) | max abs err <= 1e-12 * (1+max abs J) |
| T-N1-02 | exact JVP and VJP vs analytic J v, J^T u | rel err <= 1e-12 |
| T-N1-03 | VJP-sketch estimate g_hat = (SJ)^T(SJ), S Gaussian k x m, E[S^T S]=I: mean over 400 sketches (k=8) vs exact g (unbiasedness) | Frobenius rel err of the mean <= 0.15 ; and median single-sketch error at k=64 < median at k=4 (20 trials) |
| T-N1-04 | Hutchinson tr(g), Gaussian probes, probe count N from the Laurent–Massart/Bernstein formula (eps=0.1, delta=0.05, true a=||g||_F/tr g, b=||g||_2/tr g): empirical failure rate P(\|T_hat - tr g\| > eps tr g) over 400 repeats | <= delta = 0.05 |
| T-N1-05 | slot shares tr(P_s g P_s)/tr g, slots partition the input coords, 4000 probes, vs exact | abs err per slot <= 0.03 ; shares (normalised by the common estimate) sum to 1 within 1e-12 |
| T-N1-06 | tr(P_U g P_U) for a random 3-dim subspace U, 4000 probes, vs exact tr(U^T g U) | rel err <= 0.05 |
| T-N1-07 | randomized SVD (rank 10, oversample 10, 2 power iters) of a 60x80 matrix with sigma_i = 2^-i, via matvec/rmatvec only | rel err of top-5 singular values <= 1e-6 ; spectral truncation error <= 2 sigma_11 |
| T-N1-08 | gradient-free central finite differences, h = eps^(1/3)*\|\|x\|\| (float64), analytic map: plain central and 1-level Richardson | rel err <= 1e-8 each |
| T-N1-09 | same, deliberately large step h = 1e-2*\|\|x\|\|: Richardson improves on plain central | err_plain / err_richardson >= 100 |
| T-N1-10 | query-cost accounting: counted f-evaluations equal 2*n*L (L = Richardson levels incl. base) exactly; the error at L=1,2,3 (h=1e-1\|\|x\|\|) is non-increasing in L | counts exact; err(L+1) <= err(L) |
| T-N1-11 | float32 vs float64 drift: exact-autograd J on a 3-layer tanh MLP; FD in float32 with h = eps32^(1/3)\|\|x\|\| | autograd drift rel <= 1e-5 ; FD(float32) rel err <= 5e-2 (the ratio FD/autograd drift is reported) |
| T-N1-12 | Fisher–Rao pullback g = J^T(diag p - p p^T)J for logits f(x)=Wx+b, vs analytic | rel err <= 1e-12 ; the sqrt(p) embedding form 4 J_s^T J_s agrees within 1e-12 |
| T-N1-13 | logit-shift invariance: f -> f + c(x)1 with nonconstant smooth c leaves g_FR unchanged; negative control: Euclidean pullback J^TJ is changed | g_FR max rel change <= 1e-12 ; Euclidean pullback change >= 1e-2 |
| T-N1-14 | rewriting group (hidden-unit permutation + positive rescaling, scales log-uniform [0.1,10], ReLU MLP 3 hidden layers, 20 random rewritings x 50 points): input-space g (Euclidean and Fisher–Rao pullbacks) invariant; hidden-layer metric covariant g_h' = M^-T g_h M^-1 | input-space max abs diff <= 1e-10 * max abs g ; hidden covariance <= 1e-10 rel |

## N2  graph geodesic distances on the g-weighted kNN graph
Manufactured manifold: S^2 (true distance = great-circle angle).  (a) ambient points, g = identity of R^3;
(b) spherical chart (theta,phi), g = J^T J from exact autograd Jacobian of the embedding (N1 feeds N2).
Evaluated on core pairs with true distance >= 0.3; sampled in a margin larger than the core so paths may bulge.
| ID | statement | tolerance |
|---|---|---|
| T-N2-01 | n=4000, k=12, both (a) and (b): relative error of graph distance | mean <= 0.03 and 95th percentile <= 0.08 |
| T-N2-02 | error vs k (n=4000, k in {6,12,24}) is non-increasing within 10% slack; error vs n (k=12, n in {500,1000,2000,4000}) does not worsen: err(4000) <= 1.1*err(500). (Expected plateau in n at fixed k — direction quantisation — stated up front.) | as stated |
| T-N2-03 | consistency of the g-weighted chart graph (b) with the ambient graph (a) on the same geometry | mean abs relative difference of distances <= 0.02 |
| T-N2-04 | structural: symmetric, zero diagonal, triangle inequality holds exactly, and no under-shoot of the truth | symmetry err <= 1e-12 ; triangle violation <= 1e-9 ; min(d_graph/d_true) >= 0.995 |

## N3  delta-symmetry search
Defect of a transformation T for a map F: delta(T) = RMS_x\|F(Tx)-F(x)\| / RMS_x\|F(x)-mean F\|.
Lie-algebra search: Gram Q_ij = E<dF(A_i x), dF(A_j x)> over a candidate basis {A_i}; subalgebra = near-null space; Lie closure by commutators.
| ID | statement | tolerance |
|---|---|---|
| T-N3-01 | F=\|\|x\|\| on R^3, candidate basis gl(3) (9 dims), 2000 points: symmetry algebra = so(3) | dim = 3 ; lambda_4/lambda_3 >= 1e6 ; sine of largest principal angle to so(3) <= 1e-8 |
| T-N3-02 | non-abelian detection: (i) F=\|\|x\|\| -> so(3) non-abelian, closure residual of the found basis under brackets <= 1e-8; (ii) F=(x1,x2) on gl(3): symmetry = matrices with only third row nonzero, dim 3, non-abelian; (iii) F=(x1^2+x2^2, x3): dim 1, abelian; (iv) F=x: dim 0 | dims exactly right ; abelian flag right ; closure residual <= 1e-8 |
| T-N3-03 | finite non-abelian group: F=(sum x, sum x^2, sum x^3) on R^3, S3 permutation matrices + 10 random orthogonal decoys; closure of generators (12),(123) | closure size = 6, flagged non-abelian ; delta(true) <= 1e-12 ; delta(every decoy) >= 0.1 |
| T-N3-04 | calibration (dataset form, kNN lookup, n=3000 gaussian points in R^2, F=\|\|x\|\|, random-relabelling null with 400 shuffles, alpha=0.05, DKW 95% band): true rotations (angles 0.3,1,2,pi) accepted (delta <= delta*_alpha) ; scaling by 2 and translation by 3e_1 rejected ; false-positive rate of 400 fresh held-out shuffles at the calibrated threshold | accept/reject exactly as stated ; FPR <= 0.10 |

## N4  meshfree least-squares Killing solver (Gaussian RBF collocation, orthonormalised value basis)
sigma_i = singular values of the collocation operator on L2-unit fields; null dim = #{sigma_i < 1e-3}, additionally required gap sigma_(K+1)/sigma_K >= 100 (>=20 for T-N4-03).
| ID | statement | tolerance |
|---|---|---|
| T-N4-01 | flat torus T^2, g=I (periodic basis), n=400: true Killing space = span(d_1,d_2) | null dim = 2 ; gap >= 100 ; subspace error (sine of largest principal angle to exact span, on independent test points) <= 1e-4 |
| T-N4-02 | sphere S^2 band chart theta in [0.3,pi-0.3], g=diag(1,sin^2 theta) (phi periodic), n=600: true space = so(3), dim 3 | null dim = 3 ; gap >= 100 ; subspace error <= 1e-3. Convergence: n in {100,200,400,800}: err(800) <= 0.1*err(100) OR err(800) <= 1e-8; observed order (slope log err vs log h) reported |
| T-N4-03 | synthetic rotated/translated-image case: manifold of images I0(R_-a (x-t)) (3 asymmetric Gaussian blobs, 53x53 grid), chart (t1,t2,a), pixel-L2 pullback metric from exact autograd J, n=500; true Killing fields (1,0,0),(0,1,0),(-t2,t1,1) | null dim = 3 ; gap >= 20 ; subspace error <= 1e-2 |
| T-N4-04 | negative control: g = exp(2u)I with strongly varying u on [0,2]^2 (no Killing fields) | null dim = 0, i.e. sigma_min >= 1e-3 |
| T-N4-05 | self-consistency: recovered fields from T-N4-02 evaluated on independent points: \|\|L_v g\|\|/(\|\|v\|\| scale) | <= 1e-3 |

## N5  persistent homology
ripser (float32 internally) -> every bound below carries slack 1e-6.  Stability facts used: Rips filtration values are edge weights, so a
sup-norm perturbation eta of the distance matrix moves the diagram by bottleneck <= eta; for S subset X, d_B(dgm S, dgm X) <= 2 d_H(S,X).
| ID | statement | tolerance |
|---|---|---|
| T-N5-01 | stability theorem, empirically: circle (n=150), symmetric perturbation of D with sup-norm eta in {0.01,0.05,0.1}, 20 trials, H0 and H1 | d_B <= eta + 1e-6 in 100% of trials |
| T-N5-02 | certified Betti counts (bars with life > 2*eta_c, eta_c = 2 d_H(sample, manifold) bounded by a dense reference, alive at scale r with margin): circle n=200 r=1.0 -> b1=1 ; two circles (centres 6 apart) n=2x150 r=1.0 -> b0 certified alive = 2 and b1 = 2 ; sphere S^2 n=300 r=1.0 -> b2=1 (b1=0) ; Clifford torus n=500 r=0.9 H1 -> b1=2 (H2 reported only) | certified count == truth exactly ; no further bar with life > 2*eta_c |
| T-N5-03 | subsampling: noisy torus cloud X (n=1500, maxdim 1), random and maxmin subsets m in {50,100,200,400}, 5 seeds: bottleneck(dgm X, dgm S) vs the certified bound 2*cov_radius(S;X) | bound holds in 100% of cases (+1e-6) ; mean error at m=400 < mean at m=50 (maxmin) |

## N6  reproducibility
| ID | statement | tolerance |
|---|---|---|
| T-N6-01 | the same seed + config gives a bitwise-identical saved payload (sha256) for a full N1 run; a different seed does not | equal / different |
| T-N6-02 | every saved result has seed, config, config_hash, library versions, determinism flags; config_hash invariant under key order, changes with any value | exact |
| T-N6-03 | torch deterministic algorithms enabled; SDPA restricted to MATH (flash and mem-efficient off) | flags exactly as stated |

## Property test (spec-mandated)
Rewriting-group invariance of g to 1e-10 in float64 is T-N1-14.
