# SPEC2 — Run 2 pre-registration (Morse-theoretic upgrade)

Written 2026-09-20 09:26 CDT, before any Run-2 computation. Budget: 7 h (deadline 16:24). Source: Arjun's "Run 2" brief
(saved verbatim as `docs/RUN2_BRIEF.md`). This file fixes every threshold and every operational choice the brief left
open. **Nothing below is edited after results exist**; changes go in `AMENDMENTS.md` with a timestamp and a reason.

Rules carried over: every positive gets its null; post-hoc variants are labelled and never upgrade a verdict; verdicts are by
the letter; timings reported; the word "complete" is not used. Nothing that exists is retrained.

**What has already been seen.** Run 1 looked at a great deal of this data. Where a Run-2 criterion is scored on data whose
relevant statistic was already seen, the cell is marked **RE-SCORE** (it cannot count as a prediction) and, where possible,
an **UNSEEN** confirmation set is named. Seen in run 1: Z113 fiber axis ratios and swap distance (3 clean seeds); frequency
birth checkpoints (3 clean seeds); rot-MNIST per-layer circle r (trained, untrained); fiber β0 under the secant graph on
ResNet (1.0–1.2 vs 7.9–9.1); manifold margin vs PGD radius on ResNet (+0.84). Never analysed: `z113_decoy_s0/s1` (clean-task
models, grokked to 100%) beyond the P3 scan — they are the UNSEEN set for E3 and F.

Common definitions
* Z113 layers (L=4): `embed` = concatenated token embeddings of a and b (positions 0,1; the read-off position's own
  embedding is constant and carries nothing); `attn` = attention output at the read-off position; `resid_mid`; `resid_post`.
* ResNet-18 layers (L=5): layer1–layer4 (adaptive-avg-pooled to 4×4, flattened; layers wider than 1024 reduced to 256 PCs as
  in run 1) and penultimate. rot-MNIST layers (L=4): conv1, conv2, conv3_gap, fc1.
* Probes: ResNet — 150 training images per class (labels = the labels the model was trained on). Z113 — all training pairs
  of the model's own split, all 113 classes, labels as trained. Same RNG seeds as run 1 (0).
* Primary Z113 models: clean_s1 (generaliser) vs scrambled_s0 (memoriser). ResNet: clean_s0 vs scrambled_s0.
* Spearman/AUROC as in run 1. "Null within x of 0" means |value| < x.

## G — manifold score replacement  (runs first)
G1. ID(k) = the library's general-rank MLE at ranks (k//2, k), k ∈ {5,10,20,40,80,160} (k capped at n−2).
`m_plateau` = (length in log k of the longest contiguous run of consecutive k's whose IDs satisfy max/min − 1 < 0.15) /
(log 160 − log 5). A single k is a run of length 0.
G2. 20 bootstrap subsamples (80% without replacement); CV = std/mean of m_plateau.
Clouds: Z113 clean_s1 resid_mid (1200-pt representative probe); ring s1 (1200 states); ResNet clean penultimate (1500) plus
layer1–4 reported; GPT-2 L6 natural probe (2000).
**Pass (G):** CV < 0.15 on every primary cloud; m_plateau > 0 on the ResNet penultimate and on ≥3 of layer1–4; ordering
m_plateau(Z113) > ring > ResNet-penultimate > GPT-2. If a bootstrap mean is 0 the CV is undefined and that cloud fails.

## E1/E2 — task-relevant cohomology
E1. Grokked Z113 (clean_s1 primary; clean_s0, cleanctrl_s2 reported), resid_mid, 1200-pt representative probe, NO group
parameter. Rips H¹ on the pullback-geodesic distance matrix. Candidate classes = the 30 longest-lived. For each: DSMVJ
coordinate θ_c; v_c = unit tangent of θ_c (library `_coordinate_tangent`); s_c = E_h ‖J v_c‖². Rank by s_c. Each of the top 5 is
named by the best circular correlation against k·a, k·b, k·(a+b), k·(a−b), k = 1..56.
**Pass (E1):** ≥4 of the top-5 classes have r > 0.9 with a k that is one of that model's five key frequencies (ground truth
from the DFT, as run 1). Null: feature-shuffled cloud, same pipeline — 0 of top-5 named.
E2. Number of H¹ classes above the noise band (1.5 × longest bar over two feature shuffles, run through the same metric) under
Euclidean vs pullback-geodesic, on Z113 clean_s1, ring s1, GPT-2 L6 (secant metric for GPT-2). No pass criterion; the brief's
prediction is "fewer under pullback".

## D — functional alignment
Probe (decorrelated): 2000 (a,b) pairs from the model's test split; hint drawn uniformly from Z113 ∪ {NULL}, independent of the
answer. Layer resid_mid. ŷ = argmax output.
D1. Mapper: lens values are residues (ŷ, or z with NULL as its own cover element); cover = 12 equal arcs of Z113 with 50%
overlap; clustering inside each cover element = single-linkage at threshold = median 5-NN distance of the whole cloud; a
point's node = (first cover element containing it, its cluster id). Alignment A = NMI(node under ŷ-lens, node under z-lens).
Direct floor: NMI(ŷ, z) on the same probe.
Threshold: mean + 3σ of A over all checkpoints of clean_s0 and clean_s1 (calibration). "Fires" = A above threshold.
Held-out nulls: cleanctrl_s2, decoy_s0, decoy_s1 — all checkpoints.
D2. The re-encoded model: see AMENDMENTS (it must be regenerated; it was never checkpointed). Evasion window = checkpoints
where the run-1 H¹ detector reports 0 spurious classes AND ablation reliance > 0.9.
**Pass (D2):** A fires on every checkpoint of the evasion window. Also reported: direct NMI on the same checkpoints.
D3. L_align = λ·E_{x, z≠z'} symKL(ψ(x,z) ‖ ψ(x,z')), z,z' uniform on Z113 ∪ {NULL}, x uniform random pairs (n=4096/step).
Trigger: D1 fires at a 500-step check. λ0 = 3; schedule R (run-1: ×10 while it still fires, max 3 rounds) and schedule G
(×3, max 5 rounds). Seeds 0–4, 6000 steps. **Pass (D3):** decorrelated accuracy > 0.98 at step 6000 on 5/5 seeds under
schedule R (schedule G reported).
D4. Continue each D3 run to 12000 steps, D1 every 500 steps. **Pass (D4):** A below threshold at every check in (6000, 12000]
on every seed that passed D3.
Nulls (D): zero firings over all checkpoints of cleanctrl_s2, decoy_s0, decoy_s1.
D5 (exploratory, no criterion): with z hidden, rank newborn H¹ classes (g-geodesic Rips on the 1200-pt probe of the planted
models at the onset checkpoint) by earliness × s_c × 1/local-ID; report the rank of the class whose coordinate best tracks z.
Stated in advance: D1 needs z, exactly as the KL penalty did. What it can buy over H¹ is robustness to re-encoding, not
unsupervised discovery.

## A — depth–scale bipersistence of class fibers
A1. β0(F_y^ℓ) = connected components of the Rips graph on the fiber's points at ε = 0.5 × (largest edge of the full cloud's
minimum spanning tree). (a) Euclidean — PRIMARY. (b) pullback: same construction with g-geodesic distances on the k=15 graph
(secant weights where no Jacobian: ResNet conv layers). d*(y) = first ℓ ∈ {1..L} with β0 = 1, L if never. U = 1 − E_y[d*/L].
Secondary, reported only: robust reach (ε = 0.5 × the scale at which 99% of points are in one component).
A2. Fibered barcode: β0(ℓ, ε) on a shared grid of 40 ε values in [0, 2] × (median pairwise distance of that layer's full
cloud), per class; saved as arrays; heatmaps. (RIVET only if `pip`-installable in < 15 min.)
A3. GW(F_y^input, F_y^ℓ): POT `gromov_wasserstein2`, square loss, uniform weights, both distance matrices divided by their
median, min(300, n) points per class. ResNet: all 10 classes, all 5 layers. Z113: 20 classes (every 6th), 4 layers; input
cloud = one-hot(a) ⊕ one-hot(b) — noted in advance that within a Z113 fiber this input cloud is a regular simplex.
**Pass (A):** U_clean − U_scrambled > 0.4 on ResNet-18 AND on Z113 (primary metric (a)). E[GW]_scr > 2 × E[GW]_clean at the
ResNet penultimate. Nulls: label-permuted U within 0.05 of 0 (both systems, clean model); under feature-shuffled activations
the ratio E[GW]_scr/E[GW]_clean ∈ [0.8, 1.25].
**Kill:** if d* = L for every class in both clean and scrambled under (a), report "ambient fibers do not connect", score
under (b) with the circularity caveat, verdict cannot exceed PARTIAL.
RE-SCORE note: fiber β0 under the secant graph was seen in run 1 on ResNet; (a) Euclidean at the MST reach scale was not.

## B — invisible displacement and neighbourhood persistence
B1. h̄ = nearest training activation (Euclidean); J = J_ℓ(h̄); δ⊥ = (I − J⁺J)δ with pinv rcond = 1e-3·σ_max.
ι = ‖δ⊥‖ / median in-distribution ‖δ⊥‖. Also ‖δ∥‖ and ‖δ⊥‖/‖δ‖. Layers: Z113 resid_mid; ring hidden state; ResNet
penultimate (J = W); GPT-2 L6 with J = Jacobian of the top-64 principal components of the logit image (the full 50257×768
Jacobian has trivial kernel; run 1 cached no Jacobian — recomputed).
B2. κ = mean Jaccard of the k=15 nearest-training-activation index sets between consecutive layers. Z113: 4 layers;
ResNet: 5; GPT-2: residual layers {2,4,6,8,10,12}; ring: single layer, κ undefined (ι only).
B3. Combined = ι·(1−κ).
OOD sets: Z113 random_tokens, no_equals, repeated_token, operand_is_EQ (run 1); ring radial_shrink, gaussian_state,
big_velocity; ResNet SVHN, Gaussian noise, CIFAR-100 test (new); GPT-2 random_tokens, shuffled_order.
Baselines: softmax confidence, entropy, Euclidean 1-NN, run-1 tubular distance. In-distribution = held-out test inputs.
**Pass (B):** on ResNet-18, AUROC of ι, of κ (as 1−κ), or of the combination exceeds BOTH softmax confidence and entropy on
SVHN AND on CIFAR-100 (the same score must win on both sets). On GPT-2: AUROC of that same score family > 0.5 on both
sets, for ι specifically (the successor of the below-chance d_g). All cells reported.

## C — Morse–Smale complex of the max-logit function
C1. f = max logit on the nodes of the pullback-weighted k=15 graph. Each node flows to its highest-f neighbour (itself if
none higher); basin = terminal maximum. Separatrix edge = graph edge whose endpoints are in different basins AND have
different argmax.
C2. d_sep(x) = min over separatrix edges e=(u,v) of [geodesic(x,u) + w(e)/2] (so a node on a separatrix edge has d = half
its shortest such edge).
C3. At each separatrix edge midpoint: 30 nearest probe points, tangent coordinates u (dimension = clip(round d_global),1,8),
least-squares fit of g = ψ_y − ψ_y' ≈ c + b·u + uᵀQu; ‖II∂‖ = ‖P(2Q)P‖_F / ‖b‖, P = projector orthogonal to b.
ρ'(x) = d_sep(x) / (1 + ‖II∂‖) at the nearest separatrix edge.
Probes: Z113 clean_s0, clean_s1, cleanctrl_s2 resid_mid representative probe (1200); ResNet clean: 1000 held-out test images
at the penultimate layer (run-1 depth design); GPT-2 L6 (exploratory, 600 points, secant weights, argmax next token).
Failure radius: run-1 procedures (Z113 activation-space bisection; ResNet PGD-direction bisection and six-radius Gaussian;
GPT-2 14-radius Gaussian) — recomputed where not on disk.
**Pass (C):** Spearman(ρ', radius) > 0.90 on ResNet (PGD radius) AND > 0.5 on Z113 (all three seeds); positive on every
system; beats softmax margin on ≥1 system; shuffled-ρ' null |Spearman| < 0.05. If d_sep alone is within 0.02 of softmax
margin on every system, that is reported as the finding.
RE-SCORE note: run 1 saw manifold margin +0.84 vs softmax +0.93 on ResNet.

## E3 — invariance as quotient
Z113: on each model, 10 full answer-fibers (c = 0,12,…,108): median axis ratio of the fiber's image in the top-4 frequency
planes, the same on the b=0 slice, and relative swap distance. **Pass:** fiber ratio < 0.1, slice ratio > 0.9, swap < 0.02 on
every model. RE-SCORE on clean_s0, clean_s1, cleanctrl_s2; **UNSEEN** confirmation on decoy_s0, decoy_s1 (hint = NULL).
Also H_*(F_y) by Rips with the two-shuffle band, reported against the orbit's.
rot-MNIST: circle present at a layer = median over digits of out-of-sample coordinate r > 0.9 (run-1 rule). Invariance depth
= first layer where the trained net has no circle and the untrained net has one. **Pass:** such a layer exists. RE-SCORE
(run 1 numbers: both lose the circle at conv3_gap by this rule, 0.57 vs 0.79 — expected verdict: fail by the letter; the
graded gap is reported).

## F — formation vineyard
Vines are the symmetry-informed frequency circles (stated as such: unsupervised Rips classes on Z113 are debris — run 1).
Per vine: onset = first checkpoint from which the circle is certified to the end; peak normalised lifetime; s_c at the
final checkpoint; survives-to-end. Birth point process = {(onset − grok checkpoint, peak lifetime / median over vines)} of
surviving vines. Grok checkpoint = largest single-checkpoint rise in test accuracy.
Seeds: clean_s0, clean_s1, cleanctrl_s2 (RE-SCORE: onsets seen) and decoy_s0, decoy_s1 (**UNSEEN**).
**Pass (F):** mode of onsets (ties → median of the tied values) within ±2 checkpoints of grokking on every seed; for every
pair of seeds, W2 between birth point processes < 5th percentile of W2 to 200 null processes (onsets resampled uniformly on
the checkpoint range, lifetimes kept); vine count and median peak lifetime each satisfy (max−min)/mean < 0.30 across seeds.

## Pauses
Pause reports are written to `PAUSE_n.md` in the run-1 format and relayed in the session; work continues after each unless
the result makes the next block meaningless.
