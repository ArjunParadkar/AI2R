# SPEC6 — Run 6 pre-registration ("Orbit-Closure")

Written 2026-09-21 22:52–22:58 CDT, before any Run-6 model is trained. Brief: `docs/RUN6_BRIEF.md`. Authors: Arjun Paradkar & Reva Paradkar
(independent); run executed by Claude. Rules as runs 1–5; amendments continue at A-15; never "complete". Cited in every findings file: Golan &
El-Yaniv 2018; Hendrycks & Gimpel 2017; Liang, Li & Srikant 2018 (ODIN); Azulay & Weiss 2019; Zhang 2019; LieGG (Moskalev et al. 2022).

## Stated before anything is computed
1. **κ and ε are built from the same numbers.** Both are functions of the set {f(gx) − ρ(g)f(x) : g ∈ G} — κ_s a cosine, ε a norm. The U test
   Spearman(κ, ε) < −0.6 is therefore partly a test of the construction and is scored as registered but not read as evidence that "two
   boundaries coincide". The evidence that matters is against *external* ground truth: unknown-set labels (K) and failure radius (F).
2. **The Z113 "different modulus" set cannot be fed as written.** The model's vocabulary is residues 0–112 plus NULL and EQ. Pairs from a
   smaller modulus (p′ = 97) are a subset of valid p = 113 inputs and the model cannot tell them apart: AUROC ≈ 0.5 is expected and it is
   scored as registered. Pairs from a larger modulus have no tokens. The two unknown sets are therefore: **OOV** = slot a replaced by a token
   never seen in an operand slot — the NULL or EQ id (structural, 2 × 1000 inputs) or a random vector at the embedding norm (continuous,
   1000 inputs); κ on the valid slot's orbit only (orbit translation of slot b by ±u, ρ = relabel by ±u; swap, ρ = id). **p′ = 97** = 2000 pairs
   with a, b < 97 (κ on the full group). K.2 passes iff AUROC(κ) ≥ 0.95 on both; expected: OOV passes or fails on its merits, p′ = 97 fails.
3. **ρ for Z113.** G = {translation (a, b) → (a ± u, b ∓ u), ρ = id; swap, ρ = id; orbit translation (a ± u, b), ρ = c → c ± u; (a, b ± u), ρ = c → c ± u},
   |G| = 7 (u = the discovered generator step; u = 1 if none). ResNet: G = circular shifts (±1, ±2, ±3 px) × {rows, cols} and horizontal flip,
   |G| = 13, ρ = id, applied with torch.roll / flip on the normalised tensor as in Run 5.
4. **The direction clause (F.2) is ResNet-only.** Z113 inputs are tokens; ∇_x ε does not exist. On Z113 the failure radius is Run 1's
   activation-space bisection at resid_mid and ε is computed on inputs; the direction clause is scored on the two clean ResNets and reported
   as "not applicable" for Z113. On ResNet the flipping perturbation is the margin-gradient direction d(x) of the PGD radius (Runs 1–5);
   cosine(d, ∇_x ε) with ε the *soft* mean defect ε̄ (differentiable), on 1000 test inputs; pass iff ≥ 70% of inputs have cosine > 0.5 on both seeds.
5. **Golan-style baseline.** 72 transformations = {identity, horizontal flip} × translations {0, ±8 px}² (circular) × rotations {0, 90, 180, 270}.
   Their paper trains a transformation classifier; the fair comparison with κ_s is the *consistency* version — mean cosine of f(Tx) with f(x)
   over the 72 designer transformations — since κ_s is exactly that over model-chosen ones. It is what is registered.
6. **ODIN** = max softmax at T = 1000 after an input step of ε = 0.0014 against the temperature-scaled cross-entropy of the predicted class.
7. **P6.c margin, on file.** Run 5 Δ: Z113 clean −0.39 … −0.44, Z113 mem −0.05 … −0.06; ResNet+aug clean −0.11, −0.12; ResNet mem −0.17, −0.24.
   With Δ < −0.25 (clean) and Δ > −0.10 (memoriser), **the ResNet cells are known in advance to fail on both sides** on Run-5-like models.
   Registered as the brief wrote it; if it fails there the instrument is transformer-only and that is the report.
8. **P6.a, b, e** are scored at Run 5's thresholds (four prior memorisers: q(swap) 0.465–0.676, shift consistency 0.17–0.20, no cycle).
9. Failure radius on ResNet is the PGD-direction bisection (primary) and the Gaussian six-radius test (reported), 1000 test inputs; the
   Spearman is on the finite-radius inputs. Softmax margin = top-1 minus top-2 softmax.

## Specimens (all UNSEEN; none exists under `runs/` now)
Z113 (25k steps, ckpt every 500): clean **12, 13, 14** (frac 0.3); memorisers = scrambled_decoy **15, 16, 17** (frac 0.3); decoys **12, 13** (frac 0.5).
ResNet-18 (Run-5 recipe): clean+aug **5, 6**; memorisers (scrambled, no aug) **5, 6, 7**. Gate as SPEC5. A gate-failed model fails its n/n clauses.

## P6 — closing the definition
| | clean iff | memoriser iff | scored on |
|---|---|---|---|
| P6.a q(swap) | < 0.10 | > 0.30 | clean 12–14; mem 15–17 |
| P6.b shift consistency | ≥ 0.75 | ≤ 0.35 | clean+aug 5, 6; mem 5–7 |
| P6.c Δ = slope − mean of 5 label-permuted slopes | < −0.25 | > −0.10 | Z113 clean 12–14, mem 15–17; ResNet clean 5, 6, mem 5–7 |
| P6.d slot shares | s_a ≥ 0.40 and s_b ≥ 0.40 and hint ≤ 0.05 | hint > 0.10 or max(s_a, s_b) < 0.40 | clean 12–14; mem 15–17 (untrained null must read memoriser-side) |
| P6.e full 113-cycle | found | not found | clean 12–14; mem 15–17 |
Block passes iff all five pass on every model they apply to. **P7.d** persistence ≥ 0.94 after first acceptance on clean 12, 13, 14 (every checkpoint swept).

## K — what it does not know
K.1 `whiteboxtopo/knowledge.py`: `orbit_coherence(f, x, group)` → κ, κ_s; `equivariance_defect` → ε, ε̄. Planted check: an exactly equivariant map has
κ = 1, ε = 0; a random map has κ ≈ 1/n_classes.
K.2 Z113, clean 12–14: AUROC(κ, in vs unknown) ≥ 0.95 on OOV and on p′ = 97 (see §2) on 3/3. Reported: κ_s, max softmax, entropy.
K.3 ResNet clean+aug 5, 6: in = CIFAR-10 test (2000); unknown = SVHN test (2000), CIFAR-100 test (2000), Gaussian noise (2000, at training mean/std).
   Pass iff on both seeds and on both of SVHN and CIFAR-100: AUROC(κ_s) > AUROC(max softmax), > AUROC(entropy), > AUROC(ODIN), and ≥ AUROC(Golan-72). Noise reported.
K.4 Memoriser median κ on its own training set < 0.3 on Z113 mem 15–17 and ResNet mem 5–7; clean > 0.95 on clean 12–14 (train pairs) and ResNet clean 5, 6 (train images). Pass iff all.
K.5 Figure only: κ histograms; the 16 lowest-κ CIFAR-10 test images per clean ResNet rendered; the lowest-κ Z113 pairs listed.

## F — where it will fail
F.1 ε, ε̄ as §3–4. Failure radius: Z113 resid_mid bisection (Run 1) on 1000 held-out pairs; ResNet PGD-direction bisection on 1000 test images.
F.2 Pass iff Spearman(ε, radius) < Spearman(softmax margin, radius) (both expected negative) on clean 12–14 and ResNet clean 5, 6 — 5/5;
   ε̄ reported. Direction clause (§4): ResNet only, ≥ 70% at cosine > 0.5 on both seeds.
F.3 Nulls: shuffled ε (Spearman ≈ 0); ε_random with 7 (Z113: random token substitutions in one slot, ρ = id) / 13 (ResNet: random 3-px
   translations with random sign and random pixel-permutation of one 4 × 4 patch) non-group transformations — group ε must have the more
   negative Spearman on every model, or the group is not doing the work.

## U — one object
`open()` gains κ and ε; the demo prints all seven cells on one fresh model. Registered: Spearman(κ_s, ε̄) < −0.6 over the pooled in + unknown
inputs on every clean model (with §1's caveat stated next to the number); memorisers reported as vacuous if their κ interquartile range < 0.1.

Deliverables: FINDINGS6_{P6,K,F,U}.md, CLAIMS.md (six-run grid), figures `run6_closure.png`, `run6_roc.png`, `run6_failure.png`, `run6_knowledge_map.png`,
`run6_kappa_eps.png`, `grid_six_runs.png`, demo transcript, report.
