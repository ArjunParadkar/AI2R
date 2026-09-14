# Block O amendments

**#0 (2026-09-14, before any O compute — found by the trainer smoke test).** SPEC_O.md's Fourier-structure diagnostic used τ = 0.01 for the
frequency set K = {k : share_k > τ}. With p = 113 there are 56 folded frequencies, so a *flat* spectrum gives every frequency share ≈ 0.018 > τ
and κ = 1 at initialisation — the diagnostic could not fail. τ is raised to 0.05 (2.8× the uniform share; a grokked model puts ≥ 10 % on each of
its 3–6 frequencies, Nanda et al.). No thresholds on results were touched; no O run had executed.

**#1 (2026-09-14 15:20, ring budget).** ring_s0 reached angular error 0.155 at the 4 000-step budget with the curve still falling
(0.187 at step 2 400) — an under-trained budget, not a property of the specimen. Ring training budget raised to 8 000 steps (checkpoints
every 2 % = 160 steps); the inclusion threshold (0.15 rad on held-out length-200 sequences) is unchanged. Seed 0 re-trained from scratch;
its 4 000-step curve is kept as `results/curves/ring_s0_4000steps_excluded.csv`.

**#2 (2026-09-14 16:40, Fourier-structure diagnostic — written before κ_f was computed on any specimen).**
Pre-registered (SPEC_O O1.1, P1, P6; FINAL_SPEC §2.6, §6.7): κ on the *token embedding* (`extractors.fourier_irreps`, τ = 0.05)
> 0.9 for inclusion and P1, and κ > 0.9 (generaliser) / < 0.3 (memoriser) for P6. Outcome on the first finished specimens, recorded
here as the pre-registered result: grok_clean_s0 test acc 1.000, κ_emb 0.337; grok_planted_s0 test 1.000, κ_emb 0.604;
grok_memorize_s0 test 0.009, κ_emb 0.000 (grok_clean_s1 at epoch 24 000: test 1.000, κ_emb 0.000). The grokked embeddings are
not sparse in the Nanda sense: the effective number of frequencies exp H(share) is 26.9 (clean) / 26.2 (planted) of 56
(memoriser 55.8; a Gaussian embedding 55.8); 21 frequencies carry 90 % of the power; the largest single share is 0.06–0.07, so
almost nothing clears τ = 0.05. Likely reason: the pre-registered criterion was calibrated on Nanda et al.'s LayerNorm-free model,
whereas SPEC_O fixes a pre-LN ToyTransformer (LayerNorm rescalings are part of the transformer gauge group, FINAL_SPEC §2.3);
under pre-LN the token-axis DFT of the raw embedding is not the quantity the network computes with, and its power spreads over
~20 frequencies. Under the pre-registered rule every grokked specimen is excluded and P1/P6 fail on all of them. That outcome is
kept: `specimens.csv` carries `kappa_emb`, and the P1/P6 cells report the pre-registered estimator as FAILED alongside the amended one.

Amended diagnostic — **function-level Fourier concentration κ_f**. Let L[a, b, c] be the logit tensor over all p² pairs (planted
specimens: evaluated with the leaked shortcut token, i.e. the trained input distribution), centred over c (softmax invariance).
κ_f = fraction of the 3-D DFT power of L on the diagonal bins (k, k, −k), k = 1 … p−1. Any function of (a + b − c) mod p alone has
DFT support exactly on those bins, so κ_f is the fraction of logit variance explained by a translation-equivariant function of the
residue — precisely the statement "the network implements the character/Fourier algorithm of §2.6" — while a memoriser gives
≈ 2(p−1)/(p²(p−1)) ≈ 1e-4. κ_f is a function of the realisation map, hence invariant under all of G by construction (the 300
gauge samples are still run, K-O-2). Per-frequency shares s_k (k and p−k folded) give the frequency set K_f = {k : s_k > τ = 0.05},
the irrep amplitudes/phases used by R1, and the frequency count n_eff = exp H(s). Thresholds are unchanged: grokked iff test acc
> 0.99 and κ_f > 0.9; P6: κ_f > 0.9 / < 0.3; the memorising control must still have test acc < 0.05. κ_emb continues to be computed
and reported. Implementation: `sweep_tosho.fourier_kappa_f` (per checkpoint, stored in every `.pt`), `o11_diagnose.py` rebuilds
`specimens.csv` / `O1_1_excluded.csv` from the saved `final.pt` files under this rule (no specimen is retrained; s0 specimens were
trained before this amendment and are diagnosed post hoc from their checkpoints).

**#3 (2026-09-14 16:55, planted specimens under κ_f — written before computing it).** For a planted specimen the trained function
contains, by design, a term driven by the shortcut token s = (a+b) mod 8: any dependence on (s, c) that is not a function of
(a+b−c) has DFT support off the diagonal bins, so κ_f evaluated with the leaked s is < 1 exactly when the shortcut is used. That
is the P3 signature, not a failure of Fourier structure. Therefore: (i) the inclusion diagnostic "Fourier structure present" for
planted specimens is κ_f evaluated with the shortcut token *randomised* (the group-algorithm component of the function; threshold
0.9 unchanged), with the leaked-s test accuracy > 0.99 as before; (ii) κ_f with the leaked s is reported as well, and
1 − κ_f(leak) is recorded as the certified shortcut share of the logit variance; (iii) the accuracy drop under s-randomisation
(the SPEC_O ground-truth crystallisation quantity) is recorded per specimen. First outcome under #2 alone, recorded before this
amendment: grok_planted_s0 κ_f(leak) = 0.766 (test 1.000), grok_clean_s0 κ_f = 0.918, grok_memorize_s0 κ_f = 0.0001.
_#3 clarification (16:58, after one computation)._ "Randomised" is the wrong estimator: a single random draw of s does not remove
the shortcut term, it scrambles it (grok_planted_s0: κ_f 0.615 with one random draw, accuracy 0.996 — the shortcut is not needed
for the answer). The group-algorithm component is the logit tensor **marginalised over the 8 shortcut values** (the expectation
under the pre-onset, uninformative distribution of s); κ_f of that tensor is the planted inclusion/P1/P6 quantity. κ_f(leak),
the shortcut share 1 − κ_f(leak), and the accuracy drop under marginalisation are reported alongside.

**#4 (2026-09-14 17:25, planted onset + final-checkpoint stability — written before any recipe-2 specimen was trained).**
Evidence (grok_planted_s0, recipe 1, every 5th checkpoint): the leaked-s and shortcut-marginalised logits agree at *every* checkpoint
(accuracy identical, e.g. 0.9983/0.9983 at epoch 9 000 and 1.000/1.000 thereafter; κ_f(leak) − κ_f(marginal) ≤ 0.01 throughout).
The shortcut token was never adopted: it became informative at epoch 9 000 (30 % of the budget), i.e. exactly when the model
finished grokking (test 0.71 at 7 500 → 0.998 at 9 000), so there was no gradient pressure to use a redundant signal. As a P3
specimen it contains no bias; the SPEC_O ground truth (ablation Δacc > 5 points) can never trigger. Second observation on the same
trajectory: under constant-LR full-batch AdamW the function oscillates after grokking — κ_f 0.91–0.96 from epoch 12 000 on with
dips to 0.80 (21 000) and 0.77 at the final checkpoint 30 000 (clean_s0's final 0.918 is likewise one draw of this cycle) — so the
final checkpoint's diagnosis is a lottery, not a property of the specimen.
Amendments to the T1 grokking recipe (all three conditions, all seeds; "recipe 2"): (a) planted data onset 30 % → **10 %** of the
budget (epoch 3 000, when test accuracy is ≈ 4 %, well before the 7 500–9 000 grokking window measured above); the ground-truth
definitions of SPEC_O are unchanged. (b) learning rate constant 1e-3 for the first 90 % of the budget, then cosine decay to 0 over
the last 10 % (a standard convergence schedule; removes the terminal oscillation so the final checkpoint is the converged model).
Nothing else changes (budget, optimiser, weight decay, data split, checkpoint cadence, thresholds). Recipe-1 specimens are kept as
reference specimens, renamed `grok_clean_s0_v1`, `grok_memorize_s0_v1`, `grok_planted30_s0_v1` (listed in specimens.csv with
included = 0, reason "reference specimen (recipe 1, AMENDMENTS #4)"); `grok_planted30_s0_v1` is additionally the *inert-shortcut*
negative control for O1.3 (a planted token that was never adopted must produce no detection). Cost: clean_s0 and memorize_s0
retrained (≈ 45 min each on the shared GPU); s1 partial runs discarded.

**#5 (2026-09-14 17:05, revised 17:40 after AMENDMENTS #4 renumbered it; O1.3 Nishino Flower — logit-level ground truth for P3
crystallisation. Written BEFORE any O1.3 detector was run; no detector onset estimate was known when the rule below was fixed.)**
SPEC_O O1.1/§6.4 define the planted specimens' ground-truth crystallisation as *the first checkpoint at which ablating s
(replacing it by a random residue token) changes test accuracy by > 5 points*. On the recipe-1 planted specimen this quantity
never fires: `grok_planted30_s0_v1` reaches test accuracy 1.000 with the leaked s and 0.996 with s randomised (AMENDMENTS #3
clarification) — Δ = 0.4 points at the end of training — and AMENDMENTS #4 then established that this specimen never adopted the
shortcut at all, which is why it is kept as the *inert-shortcut* negative control. Recipe 2 moves the data onset to 10 % of the
budget so that the shortcut is available before grokking, but the accuracy-level ground truth can still be silent by construction:
the shortcut leaks only (a + b) mod 8, which the group algorithm already computes, so the network can put the shortcut into its
logits without ever *needing* it for the argmax. An accuracy-level definition of "the bias entered" is blind to exactly the kind of
bias P3 is about — a term in the function, not a crutch for the decision.

A second, logit-level ground truth is added. Per checkpoint, let κ_f(leak) be the function-level Fourier concentration of the logit
tensor evaluated with the leaked shortcut token (AMENDMENTS #2/#3) and let the **shortcut share** be σ(t) = 1 − κ_f(leak)(t): the
fraction of centred logit variance that is *not* a translation-equivariant function of (a + b − c) mod p, i.e. the part of the
function the planted token can carry. σ is a function of the realisation map and so is gauge-invariant by construction (K-O-2 still
measured). Clean specimens have no shortcut token; their σ is the off-diagonal *residue* of a pure group algorithm (finite width,
LayerNorm, the non-Fourier remainder), which is the correct null level for σ.

**Definition (logit-level crystallisation).** Let R(f) be the clean reference at training fraction f: the median over all *included
clean* specimens of σ at matched fraction f (their checkpoint grids linearly interpolated onto f). The logit-level crystallisation
of a planted specimen is the **first checkpoint with f ≥ f_onset (that specimen's own data onset, `cfg["onset"]` / budget: 0.10
under recipe 2, 0.30 under recipe 1) at which σ(f) > R(f) + δ and the inequality also holds at every later checkpoint** (a sustained
crossing), with the margin fixed at **δ = 0.05** of the centred logit variance.

Reasons. (i) *Matched-fraction reference rather than an absolute level*: σ is large and noisy early in training in every condition
(an untrained model's logits are not a group function at all), so a fixed absolute level would fire at initialisation; subtracting
the clean trajectory at the same training fraction removes the part of σ that is "not grokked yet" and leaves the part that is "uses
the token". (ii) *δ = 0.05*: the same 0.05 used throughout Block O as the threshold for "a Fourier component carries real power"
(AMENDMENTS #0/#2, τ = 0.05); it is 5 % of the whole logit variance, well above the checkpoint-to-checkpoint dispersion of σ on a
grokked clean specimen and well below the 0.234 shortcut share the recipe-1 planted specimen ends at (AMENDMENTS #3), so the rule
can both fire and fail to fire. We state plainly that those *final* values were already published in AMENDMENTS #3 when δ was
chosen; what was not known, and what this rule estimates, is the *time* at which σ separates. (iii) *Sustained crossing*: σ is
computed from one checkpoint's logits with no smoothing, and AMENDMENTS #4 measured a post-grokking oscillation in κ_f of up to
0.15 under recipe 1; a first-crossing rule without the "and stays" clause would be set by whichever checkpoint noise crossed first.
(iv) *f ≥ f_onset*: before the data onset s is uniformly random, so a crossing there would be a detector artefact, not a planted
bias; the rule is only allowed to look where the bias exists.
Sensitivity to δ ∈ {0.02, 0.05, 0.10} is reported as evidence in `results/open/nishino_flower_<specimen>.json`; the pre-stated rule
remains δ = 0.05. Both ground truths (accuracy-level, possibly undefined; logit-level, possibly undefined if σ never separates) are
carried in `results/nishino_flower_detections.csv`, and every detector is scored against each of them separately.

**#6 (2026-09-14 17:20, renumbered 17:55 — the number #4 went to the recipe-2 amendment; O1.4 Fine Motion — P4/P5 realisation on T1; written BEFORE any P4/P5 number was computed).**
SPEC_O §O1.4 and FINAL_SPEC §6.5/§6.6 fix the criteria (AUC > 0.9 beating softmax confidence, random-projection null;
Spearman(ρ, failure radius) < −0.5 at p < 0.01 beating the softmax-margin baseline, shuffled-ρ null). None of those
thresholds is touched. What follows is the realisation of the *inputs and the estimators* on the T1 specimens, where the
spec's wording assumes an architecture or an input domain the T1 specimens do not have.

(a) **"moduli and ranges unseen" is not realisable on T1.** A `grok_*` specimen has a fixed modulus p = 113 and its input
domain is the whole of Z₁₁₃ × Z₁₁₃; training uses 30 % of the p² pairs and the held-out 70 % exhausts the rest. There is no
unseen modulus and no unseen range for a model whose vocabulary *is* Z₁₁₃ ∪ {'='}. The P4 OOD set is therefore the
remaining SPEC_O categories, enumerated here and frozen: **corrupted tokens** — (i) `corrupt_eq`: a uniformly random
number token in place of '='; (ii) `swap_eq`: the positions permuted to [b, =, a] (planted: [b, =, a, s]); (iii) `dup_eq`:
the b token replaced by a second '=' ([a, =, =]); **out-of-vocabulary embeddings** — (iv) `oov_emb`: the embedded vector of
the a position replaced by a random direction of matched norm (the mean ‖wte‖ over the vocabulary), i.e. a token that
exists in embedding space but in no vocabulary; (v) `shortcut_tok`: a planted shortcut token (id ≥ p+1) placed in the a
position; **adversarial** — (vi) `adversarial`: iterated gradient ascent on the embedded input against the model's own
prediction, accepted at the smallest radius in the P5 grid at which the argmax flips (a genuinely minimal-radius flip is not
claimed; the radius grid is the resolution).

(b) **(v) `shortcut_tok` does not exist for clean and memorising specimens.** Their vocabulary is exactly p + 1 = 114 ids,
so "shortcut tokens fed to the clean model" has no referent: there is no token id outside the vocabulary to feed. The
condition is recorded as `not_reached` with that reason on clean/memorise specimens and run only on planted ones; the
out-of-vocabulary role on clean/memorise is carried by (iv) `oov_emb`, which is the same idea done in embedding space.

(c) **Lip(x) for the toy transformer is a Jacobian norm, not the exact path product.** `geometry.lipschitz_field` is exact
only for ReLU MLPs (`invariants.model.as_relu_mlp` refuses anything with attention or LayerNorm), and the T1 grokked
specimen is a pre-LN attention model. Lip(x) is therefore **the spectral norm of the autograd Jacobian of the last-position
logits with respect to the embedded input** (token + positional embeddings, flattened over the T·d input coordinates),
computed by 113 batched backward passes. *Failure boundary, recorded as part of the cell:* unlike the MLP path product,
which is the exact Jacobian on a whole gate region, this is a pointwise derivative — softmax attention and LayerNorm make
the logit map non-piecewise-linear, so ‖J(x)‖₂ bounds the local response only in a neighbourhood small compared with the
map's own curvature, and at the largest radii of the P5 grid it is an extrapolation. It is gauge-invariant by construction
(no element of G touches `wte`/`wpe`, and every element leaves the function unchanged, hence its Jacobian).

(d) **The failure radius is reimplemented, not taken from the library.** `geometry.perturbation_failure_radius` also goes
through `as_relu_mlp` and cannot accept the transformer, and its rule is "*any* of n_dirs directions flips" whereas SPEC_O
O1.4 as run here uses the **majority rule: ≥ 5 of 8 Gaussian draws flip the argmax**. The nested design is kept (8
directions drawn once per input, normalised to the unit sphere, reused at every radius); the radii r ∈ {0.01, 0.02, 0.05,
0.1, 0.2, 0.5} are **relative to the global std of the embedded inputs** (ring: of the hidden states), since SPEC_O states
them as bare numbers and absolute distances are not comparable across specimens. Inputs never flipped at r = 0.5 get
radius ∞ (ranked highest), as in the library; the tied fraction is reported, and a cell whose radii are almost all ∞ is
reported as failed/partial with that reason rather than rescued by widening the grid.

(e) **Ring baselines.** The ring RNN has no softmax. The pre-registered stand-in named in SPEC_O for P4 is the readout-norm
deviation |‖R(h)‖ − 1| ("the ring's confidence"); the same quantity is used as the P5 baseline in place of the
softmax margin, and is reported as such. Ring failure for P5 is angular error > 0.3 rad after 50 further steps (SPEC_O),
under the same majority-of-8 rule.

(f) **Gauge orbit size.** K-O-2 specifies 300 gauge samples. Each P4 recomputation is a full diffusion embedding
(Laplacian + eigendecomposition + Nyström) and each P5 recomputation is a 113-backward-pass Jacobian sweep plus 48 forward
passes; 300 of each per specimen is hours on a GPU shared with four training units and an enumeration job. The default here
is **20 gauge samples** (`--n-gauge`, recorded in every JSON and CSV row); the CV > 1e-3 downgrade rule is unchanged. A cell
certified at n = 20 is certified against a 20-sample orbit and says so in its provenance.

(g) **The ring RNN's gauge group is defined here** (`whiteboxtopo.symmetry` has no RNN backend). For S = P·D with P a
permutation and D a positive diagonal, h ↦ hS is an exact symmetry of h' = (1−α)h + α·ReLU(hW + uU + b) with
W ↦ S⁻¹WS, U ↦ US, b ↦ bS, R ↦ S⁻¹R, readout bias unchanged (ReLU commutes with S because D > 0, and h₀ = 0 is fixed).
`fine_motion.ring_gauge_apply` implements it; the K-O-2 orbit for ring cells is drawn from it.

(h) **Gauge fixing of the clouds (recorded).** Every activation cloud entering a diffusion embedding or a curvature
estimate is standardised per unit by the *in-distribution* cloud's per-unit std (`topology.scaling.standardized_cloud`
convention: divide, no centring; zero-scale units left as 0), and the query points are divided by the same reference
scales. This is the gauge fixing K-O-2 requires: without it the k-NN graph is not invariant under per-unit ReLU scalings.
The P4 in-distribution cloud is subsampled to at most 1200 points (deterministic, seeded) to keep the Laplacian in the
dense-eigensolver regime; the curvature neighbourhood uses k = 30 rather than the library default 15, because the fitted
intrinsic dimension may be up to `MAX_INTRINSIC_DIM` = 5 and the quadratic fit needs k > 1 + m + m(m+1)/2 = 21.

_#5 addendum (17:50, still before any O1.3 detector ran)._ The clean reference R(f) above is **not architecture-matched**:
a planted specimen has seq_len 4 and 8 extra vocabulary entries, and AMENDMENTS #3 already recorded that the recipe-1
planted specimen has σ = 0.234 against the clean specimen's 0.082 *while*, per AMENDMENTS #4, never having adopted the
shortcut at all. Under the clean-referenced rule that inert specimen would be handed a logit-level crystallisation it does
not have. A second, architecture-matched, within-specimen logit-level ground truth is therefore added and reported
alongside (not instead of) the first: with κ_f(marg) the function-level concentration of the shortcut-**marginalised**
logit tensor of the same checkpoint (AMENDMENTS #3), define
    σ_within(t) = κ_f(marg)(t) − κ_f(leak)(t) ,
the logit variance that the shortcut token moves off the group-diagonal *in this very model*, and take the logit-level
crystallisation to be the first checkpoint with f ≥ f_onset at which σ_within > δ = 0.05 and the inequality holds at every
later checkpoint. AMENDMENTS #4 measured σ_within ≤ 0.01 at every checkpoint of `grok_planted30_s0_v1`, so this estimator
is correctly silent on the inert-shortcut control — that is the reason it is added, and it was fixed from #4's published
numbers, not from any O1.3 output. Both logit-level ground truths appear in
`results/nishino_flower_detections.csv` as `gt_logit_frac` (clean-referenced, the #5 rule) and `gt_logit_within_frac`
(architecture-matched), and every detector is scored against both plus the accuracy-level ground truth.

**#6 addenda (2026-09-14 18:30, written AFTER the first two O1.4 smoke cells — recorded here because they are post-hoc,
and neither changes a pre-registered criterion.)**
(i) *Extended radius grid, diagnostic only.* On `grok_clean_s0_v1` the pre-registered P5 grid r ∈ {0.01,…,0.5} has **no
dynamic range**: 255 of 256 held-out inputs never flip at any radius in the grid (frac_inf = 0.996), so the measured
failure radius is a constant with one exception and the Spearman against it is not a test of anything. The pre-registered
cell is reported FAILED on that ground. Additionally, when frac_inf > 0.5 the runner re-measures the failure radius on
`RADII_EXT = RADII + (1, 2, 5, 10)` and records it under `evidence.diagnostics` with an explicit note. That number is a
**diagnostic**, never a criterion and never a certification: it says at what scale isotropic perturbations start flipping
this model, which is the information needed to decide whether the pre-registration was mis-scaled or the predictor is
wrong. The pre-registered verdict is unchanged by it.
(ii) *P4 diagnostics.* When a P4 cell is not certified the runner records, under `evidence.diagnostics`, the ambient
nearest-reference distance of every query group, the cloud's own nearest-neighbour distance, the AUC of that raw ambient
distance, and (grok) the same AUC with the in-distribution cloud enlarged from the 1200-point subsample to every training
pair. These separate "the diffusion embedding lost the signal" from "the activation cloud never placed these inputs
further out than held-out inputs" and from "the 1200-point subsample was too sparse". Diagnostics only; no criterion.
(iii) *Ring quantities are computed in the standardised coordinates.* Under the ring gauge group (#6g) the hidden state
transforms as h ↦ hS, so an isotropic Gaussian kick in raw coordinates and the raw one-step Jacobian norm
‖(1−α)I + α(W⊙mask)‖₂ are **not** gauge-invariant (J ↦ S⁻¹JS). Both are therefore taken in the per-unit-standardised
coordinates: the P4 off-attractor kick has per-unit scale s_j (the in-distribution per-unit std), the P5 perturbation
directions are scaled per unit by the same s_j, and the P5 Jacobian norm is ‖diag(s) J diag(1/s)‖₂. In those coordinates
the gauge acts by permutation conjugation only and all three are invariant. Recorded as part of the gauge fixing of #6h.

**#6 (2026-09-14 17:20, O1.3 zigzag budget — operational, measured on this box; no threshold on any result is touched).**
Z AMENDMENTS #2/#3 fixed the zigzag construction (per-checkpoint normalisation, β1-maximising Rips scale, the *same* probe
points at every checkpoint, N_SUB lowered until the union complex fits a 0.8 M simplex budget). On the O1.3 specimens that
budget is not binding and is not the cost: at the β1-maximising scale the T1 MLP clouds carry 45–78 H1 classes, and
Dionysus' zigzag cost grows with classes × positions rather than with points. Measured on `grok_planted30_s0_v1`:
84 983 simplices over 21 positions (11 clouds, 150 points) took > 4 min, and 91 370 simplices over 41 positions (21 clouds,
100 points) had not finished after 12 min while the box was also running four training units. For Block O the zigzag
simplex budget is therefore **30 000** with the subsample ladder 100 → 75 → 50 → 40, and the number of clouds handed to the
zigzag is capped at `ZZ_MAX_CK` (40, i.e. an onset resolution of 1/39 ≈ 2.6 % of training, inside the 5 % pass tolerance of
SPEC_O O1.3). `n_sub`, `budget_reduced`, `n_simplices`, `scale`, `scale_quantile` and `beta1_final_max` are recorded per
specimen in `results/open/nishino_flower_<specimen>.json`, so the coarsening is visible in every row it affects. This is a
compute limit, not a change to any detection rule or threshold; the zigzag detector's resolution limit is reported with its
onset estimate.

**#5 (2026-09-14 17:35, O1.2 instrument corrections from the reference-specimen smoke run — written before any recipe-2 specimen exists).**
Smoke run of meisho_doto.py on grok_clean_s0_v1 (recipe 1, reference only) exposed instrument defects, corrected here; thresholds untouched.
(a) **K-O-1 and R7 were tautological as pre-registered.** A rebuild Σ_k w_k cos(ω_k(a+b−c)) from *any* non-empty frequency set with
non-negative weights classifies Z_p addition perfectly (each term peaks uniquely at c ≡ a+b for prime p): the random-K rebuild scored
accuracy 1.000, as did the artifact-only rebuild, so an accuracy-based kill can never fire. K-O-1 and R7 are therefore measured on the
*function*: logit correlation with the model — random-K rebuild r < 0.3 (kill passes), artifact-only rebuild r < 0.3, and the certified-subset
degradation curve as Spearman(fraction kept, logit r) > 0. The accuracy values are still recorded as the tautology they are. (b) **R4** residual
on per-unit standardised activations (the recorded gauge fixing; the raw residual had σ² = 0 from sparse/dead ReLU units and the Gaussian null
itself read 0.68 — MP invalid), over the full p² domain (the earlier 3 000-pair subsample broke the (a, b) DFT used by R8), with the certified
feature basis = the degree-2 trigonometric basis of each certified irrep (adds cos/sin ω_k(a−b); the tensor square of irrep k contains both
a+b and a−b). (c) **P2** anchor evaluated as pre-registered "on regions of constant activation pattern": IG along the short path (1−δ)e → e,
δ = 0.05 (r_local); the zero-baseline global IG (r 0.045 on the reference specimen — the toy transformer with LayerNorm and softmax is not
piecewise linear, so a global path crosses many regions) is reported alongside. (d) **P0** verdict by the treewidth bound (the parameter of the
FPT conjecture, §2.14: tractable iff tw ≤ 8, marginal ≤ 32, else intractable); modularity Q is reported against its rewiring null (the reference
specimen's attribution graph is a sparse star: Q 0.06 < null 0.35, tw 3 — the old "Q > 0.3" clause mislabelled sparse graphs). (e) **P7**
irrep appearance = first checkpoint from which the frequency stays in K_f (at initialisation random frequencies exceed τ on a flat spectrum).
(f) **glue**: H1 > 0 on a clean specimen is reported as "instrument fails the sanity gate" (FINAL_SPEC §2.12), not as a property of the model.
Reference-specimen numbers kept for the record: P0 Q 0.06/tw 3; P1 κ_f 0.918, K_f 10 frequencies, phases |φ| < 0.005; P7 t_acc 0.21, t_κ 0.25,
spectral gap flat (0.24–0.31, n_small = 1 throughout → the gap-opening clause fails on this instrument); glue H0 3 / H1 27 (trivial null 1/0);
R1 logit r 0.842 (K_f), 0.956 (all 56 frequencies), 0.576 (embedding-amplitude path); R6 r 0.78–0.84 on all 10 objects (LayerNorm rescales the
remaining components per token when one Fourier component of the embedding is removed).
(iv) *Ring OOD points are transported, not redrawn, along the gauge orbit.* Drawing the off-attractor kick afresh inside
each gauge sample makes the K-O-2 CV measure the sampling noise of the random kick (a fixed unit-indexed Gaussian z is
isotropic *in law* but not permutation-equivariant *per sample*): measured CV 0.024 at n = 4, which would have downgraded
the cell to "failed (artifact)" for the wrong reason. The kick is therefore built once in the trained gauge and carried
to each gauge sample as h ↦ hS (`ring_on_orbit` passes S to the scoring function). With that, CV = 1.5e-4. The grok P4/P5
query sets need no equivalent because they live in embedding space, which no gauge element touches (CV = 0 measured).

**#7 (2026-09-14 17:35, O1.3 — the planted bias is TRANSIENT, so every ground truth becomes a window. Written before the
detectors were run on any recipe-2 specimen.)** Measured on `grok_planted_s0` (recipe 2, data onset epoch 3 000): at epoch
6 000 the shortcut is plainly in use (test accuracy 0.82 with the leaked token against 0.43 for the clean model at the same
epoch), but by the final checkpoint it is gone — κ_f(leak) 0.973 ≈ κ_f(marginal) 0.974, shortcut share 0.027, accuracy
without the shortcut 1.000. Weight decay prunes the shortcut once the full group algorithm makes it redundant. Two
consequences.
(i) *Ground truths are windows.* The accuracy-level ground truth (SPEC_O: ablation Δacc > 5 points) and both logit-level
ground truths (#5 and its addendum) each report an **entry** and an **exit** checkpoint. Detectors are scored against the
**entry** — that is the "when did the bias enter" question SPEC_O O1.3 and FINAL_SPEC §6.4 ask — and the exit is reported
alongside, because a P3 procedure that cannot see a bias that has since been pruned is a different (and also interesting)
failure from one that never saw it.
(ii) *The "sustained to the end of training" clause of #5 is withdrawn* — it was written when the only planted specimen
available had an inert shortcut, and on a transient bias it would report no crystallisation at all. It is replaced by a
**persistence window of 3 % of training**: a crossing counts as the entry once it holds for ⌈0.03·n_ckpt⌉ consecutive
checkpoints (minimum 2), and the exit is the first checkpoint after the entry at which the condition fails for the same
number of consecutive checkpoints. This keeps the reason the clause existed (AMENDMENTS #4 measured single-checkpoint κ_f
dips of up to 0.15, and one noisy checkpoint must not set the onset) while allowing the bias to end. A consequence stated
in advance: a crossing that begins in the last 3 % of training cannot be certified as sustained and is reported as no
crystallisation, with the raw trace in the JSON. The margin δ = 0.05, the matched-fraction clean reference, the
within-specimen estimator and the f ≥ f_onset restriction are all unchanged.

**#6 (2026-09-14 18:35, ring attractor sampling — written after the ring_s0 smoke run, before any other ring specimen was opened).**
Pre-registered ring P1 (SPEC_O/FINAL_SPEC §2.7: "sample activity {x(t)}, compute H1 of the Vietoris–Rips complex") on the driven
trajectories of ring_s0 (64 held-out sequences × 180 steps, per-unit standardised) gives H1 lifetime ratio 1.30 with 946 bars and circular-
coordinate agreement r 0.51 — no ring. Diagnosis: the driven cloud's leading principal component (41 % of variance) is the velocity input
(r 0.74 with ω; cos θ / sin θ load on PCs 2, 3, 5), so the Rips metric is dominated by the input, not by the attractor, in every gauge fixing
tried (raw, standardised, PCA-3/5/10/20, loud units only: ratios 1.0–1.8). §2.7 defines M as the attractor manifold of the *autonomous*
dynamics. Amended sampling: M is sampled as the states reached under zero velocity from 1 000 held-out random cues at t = 10 steps (settled
after the cue; the zero-input drift is 0.21 rad per 100 steps), per-unit standardised (recorded gauge fixing). On ring_s0 this cloud has
exactly one H1 bar (lifetime ratio ∞; feature-shuffle null 1.01, Gaussian null 1.09; a permutation+scaling gauge sample gives the identical
diagram, top bar 12.177 in both) — the driven-cloud outcome is recorded as the pre-registered result. Circular-coordinate agreement r is
defined as the mean resultant length of the phase difference after the coordinate's own gauge (orientation ± and offset) is fixed (1 − r ≈
σ²/2 for angular error σ); the Jammalamadaka–Sarma circular correlation is reported alongside; threshold 0.99 unchanged (ring_s0 at 400
cues, t = 5–20: 0.92–0.97 — may fail; that is the result). R2 extends the certified coordinate to driven trajectories through a linear
(cos, sin) decoder fitted on the attractor sample (recorded), then fits the tangent dynamics on those trajectories.
