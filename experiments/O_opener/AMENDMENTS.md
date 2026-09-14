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
