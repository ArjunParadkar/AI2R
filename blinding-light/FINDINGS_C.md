# FINDINGS — Run 2, Block C (Morse–Smale complex of the max-logit function, d_sep, ‖II∂‖, ρ′)

Written 2026-09-20. Pre-registration: `SPEC2.md` §C. Code: `whiteboxtopo/morse/__init__.py`, `analysis/run2_blockC.py`, `analysis/fig_run2_morse.py`, `tests/test_run2_morse.py`. Numbers: `results/run2/blockC.json`, per-point arrays `results/run2/blockC_points.npz`, synthetic-test record `results/run2/blockC_tests.json`. Figure: `figures/run2_morse_smale.png`.

## In plain English

Block C asked whether "how far is this point, along the network's own manifold, from a ridge of the confidence landscape where the predicted class changes" (d_sep), discounted by how sharply the decision boundary bends there (ρ′), predicts how small a perturbation flips the prediction. **It does not beat, or match, the plain softmax margin on any system, and Pass (C) is NOT MET.** On ResNet-18 ρ′ reaches Spearman +0.780 with the PGD failure radius (criterion: > 0.90); the softmax margin reaches +0.932 and run 1's simpler manifold margin +0.842. On the three Z113 seeds ρ′ is +0.19 / +0.18 / +0.16 (criterion: > 0.5 on all three; softmax margin +0.21 / +0.30 / +0.29). On GPT-2 (exploratory) it is +0.005 (p = 0.92), i.e. nothing, against +0.72 for the softmax margin. The instrument itself works — on planted landscapes it finds the two basins, puts the separatrix on the known boundary, measures distance to it to ~2 %, and recovers 1/R for a circular boundary to ~1 % — so this is a negative about the quantity, not a bug in the code.

## What contradicts the brief's framing (stated first)

1. **On Z113 and GPT-2 the Morse–Smale complex is not a partition of the probe into class regions with thin separatrices between them; almost the whole graph is separatrix.** Z113 has 113 classes and the 1200-point probe has ~10.6 points per class, so 94–96 % of kNN edges already join different predicted labels. 67–73 % of ALL graph edges are separatrix edges (clean_s0 10417/14333, clean_s1 9988/14845, cleanctrl_s2 10594/14624) and 99.6–99.9 % of nodes are an endpoint of one. d_sep therefore collapses to "half the length of my shortest cross-label, cross-basin edge" — a local kNN edge length, not a geodesic distance. GPT-2: 5576/7982 edges, 96.2 % of nodes. Only ResNet (10 classes, 100 points per class) has a sparse separatrix: 1177/10150 edges, 44.7 % of nodes on one.
2. **Basins of max-logit are not classes.** Z113: 40 / 25 / 39 maxima for 113 predicted classes; the label of a node's terminal maximum equals the node's own label for only 7.6 % / 4.4 % / 7.2 % of nodes. GPT-2: 41 maxima for 289 predicted tokens, 10 % agreement. ResNet: 12 maxima for 10 classes, 95.3 % agreement, no basin exactly label-pure. The label-change half of the separatrix condition does nearly all the work on ResNet too: 1449 edges change label, 1177 of them also change basin.
3. **"Softmax margin is a geodesic separatrix distance in disguise" is NOT supported.** |Spearman(d_sep) − Spearman(softmax margin)| against the failure radius is 0.073 / 0.153 / 0.176 (Z113), 0.111 (ResNet), 0.717 (GPT-2) — not within 0.02 on any system, always with the softmax margin ahead. The two quantities themselves correlate at +0.27 / +0.31 / +0.21 (Z113), +0.90 (ResNet), +0.34 (GPT-2). On ResNet d_sep is essentially run 1's manifold margin again (Spearman between them +0.974), slightly worse (+0.821 vs +0.842).
4. **The curvature factor as pre-registered is not a measurement of curvature.** d = clip(round d_global, 1, 8) is 8 on every system (d_global 8.8 / 8.5 / 7.7 / 9.9 / 15.8), so the quadratic has 45 unknowns and the fit has 30 points: it interpolates (relative residual ~1e-15 everywhere) and ‖II∂‖ is a property of the minimum-norm solution. ‖II∂‖ at the nearest edge has no relation to the PGD radius on ResNet (+0.028, p = 0.38) and dividing by (1 + ‖II∂‖) LOWERS the ResNet correlation (d_sep +0.821 → ρ′ +0.780). On Z113 it raises it slightly (+0.14 → +0.19 etc.; ‖II∂‖ itself −0.12 to −0.14 with radius), still below the softmax margin.

## Pre-registered criteria, by the letter

Spearman with the primary failure radius (Z113: activation-space bisection, n = 1200, all finite; ResNet: PGD-direction bisection, n = 977 finite of 1000; GPT-2: 14-radius Gaussian, n = 348 finite of 600). All predictors on the same points.

| system | ρ′ | d_sep | softmax margin | softmax conf. | run-1 manifold margin | shuffled-ρ′ null (seed 0) | \|d_sep − softmax\| | Spearman(d_sep, softmax margin) |
|---|---|---|---|---|---|---|---|---|
| Z113 clean_s0 | +0.189 (p 5e-11) | +0.139 | +0.212 | +0.149 | +0.170 | −0.031 | 0.073 | +0.270 |
| Z113 clean_s1 | +0.175 (p 9e-10) | +0.147 | +0.300 | +0.236 | +0.199 | +0.010 | 0.153 | +0.309 |
| Z113 cleanctrl_s2 | +0.158 (p 4e-8) | +0.110 | +0.286 | +0.178 | +0.111 | +0.044 | 0.176 | +0.211 |
| ResNet-18 (PGD) | +0.780 (p 5e-201) | +0.821 | +0.932 | +0.929 | +0.842 | −0.072 | 0.111 | +0.900 |
| ResNet-18 (Gaussian-6, secondary) | +0.540 | +0.550 | +0.580 | +0.585 | +0.570 | −0.010 | 0.030 | — |
| GPT-2 L6 (exploratory) | +0.005 (p 0.92) | +0.004 | +0.721 | +0.452 | +0.071 | −0.032 | 0.717 | +0.339 |

| criterion | measured | verdict |
|---|---|---|
| ρ′ > 0.90 on ResNet, PGD radius | +0.780 | **NOT MET** |
| ρ′ > 0.5 on all three Z113 seeds | +0.189, +0.175, +0.158 | **NOT MET** |
| ρ′ positive on every system | +0.189, +0.175, +0.158, +0.780, +0.005 | MET by the letter (GPT-2's +0.005 is indistinguishable from 0, p = 0.92) |
| ρ′ beats softmax margin on ≥ 1 system | 0 of 5 | **NOT MET** |
| shuffled-ρ′ null \|Spearman\| < 0.05 | 0.031, 0.010, 0.044, **0.072**, 0.032 | **NOT MET** (ResNet). Over 200 further shuffles mean \|r\| is 0.021–0.024 (Z113), 0.024 (ResNet), 0.042 (GPT-2) and 90–94 % / 90 % / 67 % fall below 0.05: at n ≈ 1000 a single shuffle has s.d. ≈ 0.032, so 0.05 is a 1.6-σ bar and the miss is sampling noise of the null, not leakage. It is still a miss by the letter. |
| **Pass (C)** | | **NOT MET** |
| d_sep within 0.02 of softmax margin on every system | 0.073, 0.153, 0.176, 0.111, 0.717 | NO — not on any system |

RE-SCORE note honoured: run 1's ResNet numbers were reproduced exactly on the recomputed radius (softmax margin +0.9322, manifold margin +0.8416), so the comparison is like for like.

Structure of the complexes

| system | n | predicted classes | maxima = basins | separatrix edges / graph edges | finite d_sep | nodes on a sep. edge | ‖II∂‖ median [p05, p95] |
|---|---|---|---|---|---|---|---|
| Z113 clean_s0 | 1200 | 113 | 40 | 10417 / 14333 | 100 % | 99.9 % | 0.47 [0.26, 0.86] |
| Z113 clean_s1 | 1200 | 113 | 25 | 9988 / 14845 | 100 % | 99.6 % | 0.65 [0.38, 1.10] |
| Z113 cleanctrl_s2 | 1200 | 113 | 39 | 10594 / 14624 | 100 % | 99.6 % | 0.59 [0.33, 1.04] |
| ResNet-18 | 1000 | 10 | 12 | 1177 / 10150 | 100 % | 44.7 % | 0.92 [0.55, 1.49] |
| GPT-2 L6 | 600 | 289 | 41 | 5576 / 7982 | 100 % | 96.2 % | 0.67 [0.28, 2.14] |

‖II∂‖ is in units of 1/(activation length) while d_sep is in pullback (logit) units; ρ′ = d_sep/(1 + ‖II∂‖) as pre-registered is therefore not scale-free. No degenerate fits (‖b‖ = 0) occurred.

POST-HOC (labelled; does not touch any verdict): the same C3 fit with k = 90 neighbours (2 × 45 unknowns, so over-determined) gives ρ′ +0.223 / +0.243 / +0.237 on Z113, +0.823 on ResNet, +0.005 on GPT-2. That is above d_sep on Z113 and nominally above the softmax margin on clean_s0 only (+0.223 vs +0.212; a 0.011 gap, post-hoc, one seed of three — not claimable). On ResNet it is a wash with d_sep (+0.823 vs +0.821) and far below the softmax margin.

## Synthetic ground truth (`tests/test_run2_morse.py`, 13 tests, all pass, 1.2–2.0 s)

Uniform 2-D cloud, library k=15 graph. Two Gaussian bumps: exactly 2 maxima, within 0.043 of the bump centres; all 371 separatrix edges cross x = 0 with midpoints within 0.070 of it; d_sep vs true distance |x|: Pearson 0.999, median ratio 0.979, median abs. error 0.021; basin-of-maximum label agrees with the node label for 99.5 % of nodes (not 100 %: the discrete flow follows the boundary only to graph scale). Half-edge convention checked exactly on a 4-node path (d = [1.2, 0.2, 0.2, 2.2]). Two basins inside one class → no separatrix edge, d_sep = ∞ (the label condition is load-bearing). Straight boundary with a linear readout: ‖II∂‖ max 1.9e-14. Straight boundary with a non-linear g (the bumps): median 0.049. Circular boundary of radius R: median ‖II∂‖ = 2.026 (R = 0.5), 0.994 (R = 1), 0.664 (R = 1.5) against 2, 1, 0.667 — relative error 1.3 %, 0.6 %, 0.4 %; inter-quartile spread about ±5 %. The under-determined case (d = 8, k = 30) is flagged and tested to interpolate.

## What is claimable

* A tested, torch-free implementation of the discrete Morse–Smale complex of a node function on the pullback graph, the half-edge separatrix distance, and a boundary second-fundamental-form estimator that recovers 1/R to ~1 % on planted circles.
* A negative: on five systems, geodesic distance to the Morse–Smale separatrix of max-logit, with or without the curvature discount, predicts failure radius worse than the top-1 − top-2 logit gap, and on ResNet no better than run 1's manifold margin. The softmax margin is NOT a separatrix distance in disguise by the pre-registered 0.02 test.
* A structural observation: with ~10 probe points per class (Z113) or an open vocabulary (GPT-2), 67–73 % of kNN edges are separatrix edges and > 96 % of nodes sit on one; the construction only has room to say something when classes are few and densely sampled (ResNet), and there it restates the manifold margin (Spearman +0.974 between them).
* NOT claimable: any statement that ρ′ or ‖II∂‖ carries failure information beyond the softmax margin; the post-hoc k = 90 variant; anything on GPT-2 beyond "null".

## Timings (wall, 5 threads, shared GPU)

Z113: 20–23 s per seed (Jacobians 0.1–1.2 s, fit_manifold 0.8–0.9 s, failure radius < 0.2 s, Morse–Smale 3 ms, d_sep 3 ms, C3 curvature 3.3–3.6 s for ~10 k edges, manifold margin 0.03 s; the rest is the 200-shuffle null and imports); all three 65 s. ResNet: 97 s (PGD radius 34 s, Gaussian radius 49 s, fit 1.2 s, Morse–Smale 3 ms, d_sep 1 ms, C3 1.2 s). GPT-2: 822 s (14 × 6 perturbed forward passes 709 s, fit 2.5 s, C3 6.1 s). Tests 1.2–2.0 s.

## Deviations and literal readings (SPEC2.md and AMENDMENTS.md not edited)

1. **C3 is under-determined as written** (45 unknowns, 30 points, on every system). Literal reading taken: minimum-norm least squares (pseudo-inverse) in the monomial basis [1, u_i, u_i u_j (i ≤ j)]. The minimum-norm solution depends on that basis and on the scale of u; no rescaling was added. The over-determined k = 90 variant is reported as POST-HOC only.
2. "30 nearest probe points" = Euclidean nearest in the layer's activation space to the midpoint (X_u + X_v)/2. "Tangent coordinates" = top-d local PCA directions of those 30 points, origin at the midpoint. ψ_y − ψ_y′ is read from the cached logits of those neighbours (all systems); y, y′ = the argmax labels of the edge's two endpoints.
3. C2 is implemented as the literal minimum over all separatrix edges and both endpoints (virtual midpoint node per edge). A node on a separatrix edge can therefore have d_sep smaller than half its own shortest separatrix edge if a neighbour's is shorter; the parenthetical in C2 holds as an upper bound, not an identity.
4. Flow ties: a node moves only to a STRICTLY higher neighbour.
5. Z113 cleanctrl_s2: the probe is drawn from the model's own 50 % training split (SPEC2 common definitions and the brief). Run 1's `z113_static.py` built this model's probe with frac = 0.3, so its run-1 P5 row used different points; the run-1 softmax-margin figure for that seed was +0.303, here +0.286. clean_s0 / clean_s1 probes are identical to run 1 (softmax margin reproduces to every digit: +0.2118, +0.3005).
6. Shuffled null scored on the run-1 convention (one permutation, seed 0); the 200-permutation distribution is reported beside it and does not change the by-the-letter verdict.
7. Spearman is computed on points with finite radius and finite values of every predictor (Z113 1200/1200, ResNet 977/1000, GPT-2 348/600: 252 GPT-2 points never flipped at the largest radius, as in run 1). Own-mask values are in the JSON and are identical here because d_sep is finite everywhere.
8. GPT-2: the graph is built on the 600 held-out windows themselves (secant weights on the exact logit isometry), not on the 2000-window probe; the noise scale and RNG stream reproduce run 1 (348 finite radii, softmax margin +0.7206, both identical to `results/gpt2_explore.json`).
9. Figure: embedding is PCA (PC1 23 %, PC2 12 % of variance), not an isotypic plane. The brief asked for heavy black separatrix lines; with 9988 of them heavy lines render as a solid black disc, so they are drawn black at 0.6 pt with transparency. Basins are coded by 4 grey levels × 7 marker shapes (25 basins, so codes repeat across distant basins).
10. `results/test_timings.json` is rewritten by `tests/conftest.py` whenever any test file runs; it was backed up before and restored after running `test_run2_morse.py` alone, so the full-suite record is unchanged.
