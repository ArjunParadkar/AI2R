# PAUSE 3 — the three failed cells, re-scored  (A, B, C) · 2026-09-20 10:10

**Plain English.** All three re-scorings fail by the letter. None of the new objects — depth profile of fiber connectivity,
invisible (kernel) displacement with neighbourhood persistence, Morse–Smale separatrix distance — beats the baseline it was
designed to beat. In each case the instrument was verified on planted ground truth first, so these are negatives about the
quantities, not bugs. Full write-ups: `FINDINGS_A.md`, `FINDINGS_B.md`, `FINDINGS_C.md`.

**Contradicting the brief's framing, stated first.**
* **A (P6).** In the ambient metric, class fibers do not connect at the pre-registered scale at *any* layer of *either* network:
  distances concentrate, so at ε = ½ × (MST connectivity scale) the full cloud is 1,459–1,487 isolated points of 1,500. U = 0.000
  for clean and scrambled alike, on both systems; the kill test fires. Gromov–Wasserstein distortion to the input fiber does
  not track memorisation: ratio 0.97 on ResNet (needed > 2) and *reversed* on Z113 (the generaliser distorts more, 0.18–0.30).
* **B (P4).** On near-OOD images the displacement is proportionally *more* in the directions the head reads, not in its kernel:
  AUROC of the invisible fraction ‖δ⊥‖/‖δ‖ is 0.33 on both SVHN and CIFAR-100 — below chance. The visible part (0.70) beats ι
  (0.57 / 0.59). Only Gaussian noise fits the brief's picture (ι 0.95 vs softmax 0.71). On GPT-2 ι clears 0.5 (0.82 / 0.54),
  but it is Euclidean 1-NN under another name (rank correlation 0.993–0.999 — the kernel is 704 of 768 dimensions). κ has no
  consistent sign: it is a near-perfect detector *with the wrong sign* on Z113 repeated_token (0.006–0.018).
* **C (P5).** Softmax margin is NOT a separatrix distance in disguise (gaps 0.07–0.72, never within 0.02). On Z113 and GPT-2
  67–73% of all graph edges are separatrix edges and > 96% of nodes touch one, so d_sep degenerates to a local edge length; basins
  of max-logit are not classes (4–8% agreement on Z113). On ResNet d_sep is run 1's manifold margin again (ρ between them 0.974).
  The pre-registered curvature fit is under-determined (45 unknowns, 30 points) and lowers the ResNet correlation.

| cell | pre-registered | measured | met? |
|---|---|---|---|
| A: U_clean − U_scr > 0.4, ResNet / Z113 (Euclidean) | | 0.000 / 0.000 | **NOT MET** (kill fires: ambient fibers do not connect; under pullback also 0.000) |
| A: E[GW]_scr > 2 × E[GW]_clean, ResNet penultimate | | ratio 0.970 | **NOT MET** |
| A nulls: label-permuted U within 0.05; shuffled GW ratio in [0.8, 1.25] | | 0.000 (vacuous); 0.944 | met |
| B: same score beats softmax conf AND entropy on SVHN AND CIFAR-100 | softmax 0.831/0.819, entropy 0.837/0.827 | ι 0.566/0.590 · 1−κ 0.434/0.636 · ι(1−κ) 0.499/0.641 | **NOT MET** |
| B: GPT-2 ι > 0.5 on both sets | run 1 d_g: 0.40 / 0.38 | 0.823 / 0.541 | met (≈ Euclidean 1-NN: 0.817 / 0.536) |
| C: ρ′ > 0.90 on ResNet (PGD) | softmax margin +0.932 | +0.780 (d_sep +0.821; run-1 manifold margin +0.842) | **NOT MET** |
| C: ρ′ > 0.5 on three Z113 seeds | softmax margin +0.21/+0.30/+0.29 | +0.189 / +0.175 / +0.158 | **NOT MET** |
| C: beats softmax margin on ≥ 1 system | | 0 of 5 | **NOT MET** |
| C: shuffled null |ρ| < 0.05 | | 0.031, 0.010, 0.044, **0.072**, 0.032 | NOT MET on ResNet (single-shuffle noise; 200-shuffle mean 0.024) |

**Kill tests.** A: fires on both systems. C: planted landscapes pass (2 basins, separatrix on the known boundary, d_sep vs true
distance r = 0.999, circular boundary 1/R to ~1%). B: synthetic kernel tests pass; every part run twice with identical AUROCs.

**Descriptive, not scored (no null, seen after the fact).** The scale at which a class fiber becomes one piece, in units of
the layer's median distance, falls with depth in the generalisers and barely moves in the memorisers (clean ResNet class 0:
1.38 → 0.46; scrambled 1.13 → 0.92; Z113 clean 0.82 → 0.31; scrambled 0.92 → 0.92). At the last Z113 layer 88/113 clean fibers
are one piece under the pullback graph against 0/113 scrambled. This is the run-1 fiber-connectivity contrast again; it
survives, the scalar U built on it does not.

**What is now claimable.** Softmax confidence/entropy/margin remain unbeaten for OOD detection and failure prediction on every
system where they apply. The three new instruments are implemented, tested against analytic ground truth, and negative.

Timings: A 129 s end to end; B 501 s (GPT-2 Jacobians 362 s); C Z113 65 s, ResNet 97 s, GPT-2 822 s.
