# SPEC7 — Run 7 pre-registration ("The Extractor, and the End")

Saved 2026-09-22 (mtime below is the clock start; hard stop five hours later; computing stops at +4 h 30). Brief: `docs/RUN7_BRIEF.md`.
Authors: Arjun Paradkar & Reva Paradkar (independent); run executed by Claude. Rules as runs 1–6; amendments continue at A-16; never "complete".
Cited in every findings file: LieGG (Moskalev et al. 2022); Bouchacourt, Ibrahim & Morcos 2021; Immer et al. 2022; Golan & El-Yaniv 2018;
Hendrycks & Gimpel 2017; Liang et al. 2018.

## Stated before anything is computed
1. **q for the CNN is Euclidean** (as Run 6's ResNet tower): q_ℓ(T) = mean_x ‖h_ℓ(Tx) − h_ℓ(x)‖ / mean_{x,x′} ‖h_ℓ(x) − h_ℓ(x′)‖ at layer1–4,
   penultimate, logits; 300 CIFAR-10 test images; 300 random pairs. For Z113 it is Run 5's finite g-length tower.
2. **λ_ℓ(family) = mean over the 5 magnitudes of (q_untr − q_tr)/q_untr**, with q_untr the mean over the three untrained ResNet seeds (init
   checkpoints of Run-6 memorisers 5, 6, 7, which are the same architecture at step 0). Reported per magnitude too. Invariance depth = first
   layer with λ ≥ 0.5. A family with q_untr < 0.05 at the layer is *built-in* and excluded (shift, flip are included in the sweep to show it).
3. **CIFAR-10-C is not on disk** and cannot be fetched inside the budget. P5′(b) uses the five corruptions **generated here with the reference
   code's parameters** (gaussian_noise σ ∈ {.04,.06,.08,.09,.10}; brightness +{.1,.2,.3,.4,.5} in HSV V; contrast ×{.75,.5,.4,.3,.15};
   gaussian_blur σ ∈ {.4,.6,.7,.8,1.0}; jpeg quality {80,65,58,50,40}) on 1,000 CIFAR-10 test images each — 25 cells. Said now.
4. **The Z113 "translation" family in X.4** is the orbit translation (a+u, b) — the answer moves with it (ρ ≠ id), so a network that has
   learned the group is *equivariant*, not invariant, and λ ≈ 0 is the prediction. The symmetric translation (a+u, b−u) is ρ = id and is
   reported alongside (its λ is expected high — it is an invariance the network learned). Swap is scored at resid_post: λ ≥ 0.5 on clean
   9–14, < 0.2 on memorisers 15–17. Untrained Z113 q comes from the same seed at step 0.
5. **Hue rotation** is implemented as a rotation of the (I, Q) chroma plane of YIQ (linear, differentiable); saturation as a blend with
   luminance; gamma on the un-normalised [0, 1] image; JPEG through PIL on CPU (n = 300, not differentiable — excluded from X.3's algebra).
6. **X.3's Lie algebra** has 6 affine coefficients (2 × 3 generator, exponentiated as a matrix) + 4 photometric (brightness, log-contrast,
   log-gamma, saturation) = 10, not 12 — there is no differentiable "hue" or "blur" generator in this parametrisation. Objective: mean over
   t ∈ {.25, .5, .75, 1} of q_tr(exp(tA)) − q_untr(exp(tA)) at the penultimate layer, A constrained to unit norm, Adam, 60 steps, 10 restarts;
   "within cosine 0.8 of a library family" = cosine between the 10-vector A and the family's own generator direction in the same algebra
   (rotation, iso-scale, aniso-scale, shear, translation, brightness, contrast, gamma, saturation are representable; blur, noise, JPEG,
   cutout, elastic, hue are not and cannot be recovered by construction).
7. **P5′ ResNet (a)** scores misclassification on the 2,000 CIFAR-10 test images of Run 6 (test error 22%). "κ_s^learned" uses the families
   with λ ≥ 0.5 at the penultimate layer on *that seed*, 3 magnitudes each (the middle three of the five); if fewer than one family
   qualifies the cell is reported as "no learned symmetry found" and fails.
8. **R** fits w by least squares of the one-hot target over all 113 candidates c on 1,000 training pairs (features cos(2πk(a+b−c)/113) for
   the discovered k, plus sin, plus a bias); ablation in the original = projecting the k-th Fourier mode out of W_E[:113]; γ = fraction of
   the residual's variance above the Marchenko–Pastur edge of a shuffled residual (n × 113 matrix, 5 shuffles).
9. **Run-6 K numbers are reused** as the built-in-group baseline for K′.2 (same models, same 2,000 test / 2,000 SVHN / 2,000 CIFAR-100 draws, rng 0).

## Specimens (no new training except P5′ Z113)
ResNet clean+aug 5, 6 (Run 6); memorisers 5, 6, 7 (Run 6) and their step-0 checkpoints as the untrained seeds. Z113 clean 9–14, memorisers
15–17, decoys 12–13 (Runs 5–6). **New:** Z113 clean seeds **18, 19, 20**, 12,000 steps, checkpoint every 2%; P5′ uses, per seed, the earliest
checkpoint with test accuracy in [0.70, 0.90] (if none lands in the window, the checkpoint closest to 0.80).

## X — the extractor
X.4 Pass iff (i) ≥ 3 library families with λ_penult ≥ 0.5 on both clean ResNets; (ii) shift and flip have q_untr < 0.05 at the penultimate layer
(excluded); (iii) the contrastive search returns on 2/2 seeds a generator with cosine ≥ 0.8 to a representable library family that has λ ≥ 0.5;
(iv) memorisers 5, 6, 7 each have < 2 families at λ ≥ 0.5; (v) Z113: orbit translation λ_post < 0.2 on clean 9–14; swap λ_post ≥ 0.5 on clean
9–14 and < 0.2 on memorisers 15–17. Block passes iff all five.

## K′ — P4 on the CNN with learned symmetries
K′.2 Pass iff on both seeds and both of SVHN, CIFAR-100: AUROC(κ_s^learned) ≥ AUROC(κ_s^built-in, Run 6) + 0.20. Reported: max-softmax, entropy,
ODIN, Golan-72 (Run 6). Null: κ_s^learned on the untrained network (mean over CIFAR-10 test) must be < 0.9. K′.3 figure.

## P5′ — natural failure
Z113: AUROC(κ under the discovered generator, error vs correct on 2,000 held-out pairs) > AUROC(max-softmax) on 3/3 of seeds 18–20 at the
selected checkpoint (κ hard; κ_s reported). ResNet: pass iff on both seeds AUROC(κ_s^learned) > AUROC(msp) on (a) test misclassification
or (b) the 25 generated corruption cells pooled. Null: κ_s over the families with λ < 0.2 (non-learned) must give a lower AUROC than the learned set on every cell scored.

## R — reconstruction
On clean 9–14 (K from Run 5/6 characters, u the discovered step): rebuilt accuracy ≥ 0.98 on 2,000 held-out pairs on 6/6 (registered 3/3
in the brief; six are available and all are scored); ablation symmetry on ≥ 4/5 frequencies per model; γ < 0.2.

## P7 — persistence ≥ 0.85 on Run 6's three dense sweeps (0.871, 1.000, 0.970 are on file: this will pass; it is registered as the brief wrote it).

## U — open() on r6_resnet_cleanaug_s5 and r6_z113_clean_s12; seven cells with status; transcript saved.

## Cut order at the clock (from the brief): P5′ corruptions → X.3 → R's γ → K′.3. Never cut: X.1–X.2, K′.2, Z113 P5′, P7, U.
