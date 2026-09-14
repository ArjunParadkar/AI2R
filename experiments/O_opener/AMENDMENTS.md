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

**#4 (2026-09-14 17:20, O1.4 Fine Motion — P4/P5 realisation on T1; written BEFORE any P4/P5 number was computed).**
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
