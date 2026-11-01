# WHITEBOX — Run 7: The Extractor, and the End (brief as received 2026-09-22 ~08:05 CDT; equations de-duplicated, wording unchanged)
Idea: a learned symmetry is one the trained network keeps and the untrained network doesn't. Architecture is the null. Subtract it, and what
remains is what the network learned to ignore — on any model, without being told.
Budget: five hours wall-clock, hard stop; time boxes binding; cut order at the end. Reuse every trained specimen from Runs 5–6. Only new training:
three Z113 models stopped early for P5 (~15 min). SPEC7.md first. Thresholds fixed. Nulls on every positive. Post-hoc labelled. Never "complete."
Neighbours: LieGG; Bouchacourt et al. 2021; Immer et al. 2022; Golan & El-Yaniv 2018; Hendrycks & Gimpel 2017; Liang et al. 2018. Claimed:
the architecture-subtracted definition of a learned symmetry, its use as the group for orbit coherence on models with no discrete group, and
the one-object account across the seven problems.
Block X [1 h 30]. X.1 λ_ℓ(T) = (q_ℓ^untrained(T) − q_ℓ^trained(T)) / q_ℓ^untrained(T) ≥ 0.5 ⇔ learned symmetry at layer ℓ; built-in symmetries
(q_untrained ≈ 0) excluded by construction; learned invariance depth = first ℓ with λ ≥ 0.5. X.2 fifteen families × 5 magnitudes: rotation,
isotropic scale, anisotropic scale, shear, translation beyond padding, brightness, contrast, gamma, hue rotation, saturation, Gaussian blur,
additive noise, JPEG, cutout, elastic warp; untrained nulls: 3 untrained ResNet seeds (mean q). X.3 contrastive generator search: Lie-algebra
element A (8 affine + 4 photometric), minimise q_trained(exp(tA)) while maximising q_untrained(exp(tA)), t ∈ [0, 1], gradient descent, 10
restarts; report top-3 generators, λ, nearest library element. X.4 pre-registered: ≥ 3 families with λ ≥ 0.5 at the penultimate layer on both
augmented clean ResNets (prediction: photometric); shift and flip excluded (untrained q < 0.05); contrastive search recovers ≥ 1 generator within
cosine 0.8 of a library family with λ ≥ 0.5 on 2/2; memorisers have < 2 learned symmetries; Z113: translation λ ≈ 0 (a symmetry, not an
invariance), swap λ ≥ 0.5 on generalisers, < 0.2 on memorisers. Deliverable: whiteboxtopo/extractor.py + inventory per model (P1 on the CNN).
Block K′ [45 min]. κ_s over the learned-symmetry set (λ ≥ 0.5, 3 magnitudes each). K′.2: AUROC on SVHN and CIFAR-100 beats Run 6's built-in
κ_s by ≥ 0.20 on both seeds; reported vs max-softmax, entropy, ODIN, Golan-72; untrained null must be low. K′.3 knowledge map.
Block P5′ [45 min]. No PGD; ground truth = natural failure. Z113: 3 clean models stopped at 70–90% test (seeds 18–20); AUROC(κ) > AUROC(msp)
3/3. ResNet: (a) CIFAR-10 test misclassifications; (b) CIFAR-10-C 5 corruptions × 5 severities, 2,000 images each; κ_s^learned vs msp,
entropy, ODIN; beats msp on (a) or (b) on both seeds. Null: κ under non-learned families (λ < 0.2) must be worse.
Block R [30 min]. Rebuild the predictor from u and characters alone: ŷ = argmax_c Σ_k w_k cos(ω_k(a+b−c)), w_k by least squares on 1,000
training pairs; rebuilt accuracy ≥ 0.98 held-out 3/3; ablation symmetry ≥ 4/5 frequencies; coverage gap γ < 0.2 (structured residual under MP).
Block P7 [10 min]. Persistence at 0.85 on the three dense sweeps.
Block U [20 min]. open() end-to-end on one fresh clean ResNet and one fresh clean Z113, all seven cells with status; transcript saved.
Deferred: Čech gluing on GPT-2 IOI; learned-symmetry extraction on language models; continuous/non-abelian discovery beyond the Lie search.
Deliverables: SPEC7.md; FINDINGS7_X, K, P5, R.md; CLAIMS.md final; seven-run grid; figures (inventory, κ ROC learned vs built-in, natural-failure
ROC, reconstruction + γ, seven-cell demo). At 4 h 30 stop computing and write. Cut order: P5′ CIFAR-10-C → X.3 → R's γ → K′.3 renders. Never cut:
X.1–X.2 on both ResNets with untrained nulls, K′.2, Z113 P5′, P7, the U demo. "you have 5 hours...go"
