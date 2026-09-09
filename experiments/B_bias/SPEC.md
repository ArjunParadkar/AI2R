# Block B — Bias: certify the real part, retire the rest (pre-registration)

_Naming: every run in this document is a horse from the Tracen roster (`experiments/HORSES.md`); the handbook codes below are the pre-registration labels and appear here only as such._

Status: SPEC written 2026-09-07 before any Block-B run. Novelty: **FLAGGED** (INLP/LEACE
concept-erasure literature; gauge-invariance of bias directions and subspace-rank-as-erasure-
predictor likely open but adjacent — search logged in NOVELTY.md before write-up).
Prior in-house (v1, 2026-06-28): dynamic directional erasure cut the WinoBias-style output bias
by 70.8 % at +1.9 % perplexity (GPT-2 small, blocks 1–3, α = 0.5); the "topological precedence
window" was a from-scratch phenomenon (30M/51M/124M: p ∈ [0.30, 0.97] — null); COMPAS race and
toxicity erasure failed to transfer (v1 P3 bins).

## B1 — Formal retirement  [LAPTOP]
`AUDIT.md` gets a "Bias precedence window — retired" entry with the v1 numbers (three model
sizes, permutation p-values) and the figures copied from
`~/Whitebox/v1/experiments/results/whitebox_breakthrough/*/` (phase-2 scaling figures).

## B2 — Is the bias direction real geometry?  [5070]
GPT-2 small (pretrained, no fine-tuning — the direction is measured, not trained), v1 WinoBias
setup (`build_corpus`: 100 male-stereotyped + 100 female-stereotyped sentences, mean-pooled
residual at blocks 1–3 and 6). g_l = (μ_M − μ_F)/‖·‖ in the residual stream. Renamings do not
act on the residual stream (as in K6), so the *residual* g_l is invariant by construction — the
non-trivial objects audited are (i) g_l expressed in **MLP-hidden coordinates** of block l
(direction of μ_M − μ_F of the post-GELU activations, 3072-dim; MLP-unit permutations act) and
(ii) g_l in **per-head output coordinates** (H × d_head; head permutations and within-head
rotations act), each through `audit_quantities` with the exact gauge aligners (`align="gauge"`)
and with blind Procrustes alignment on the cloud. Report: cosine after alignment (direction
stability, target ≥ 0.99 for the exact aligner = pipeline check; the blind-aligner cosine is the
finding) and magnitude covariance under LN scalings.
**Kill.** Random-split "bias direction" (random halves of the pooled probe set, 20 splits) must be
unstable: mean cosine between random-split directions < 0.5, vs > 0.9 between M/F-split
directions from disjoint sentence halves.

## B3 — Power curve  [5070]
BiasTop (W2 between group-A and group-B persistence diagrams, PCA-20 residual clouds) permutation
test (200 permutations) power vs n ∈ {25, 50, 100, 200, 400, 800} per group, at planted divergence:
group B = group A residuals + δ·g along the B2 direction, δ ∈ {0.25, 0.5, 1.0} × pooled sd, 50
trials per (n, δ); real M/F split as the reference. Report n for power 0.8 at each δ (linear
interpolation), false-positive rate at δ = 0 (must be ≤ 0.07 at α = 0.05).

## B4 — Separability rank  [5070]
Effective rank (participation ratio of the eigenvalues) of the between-group covariance
Σ_g n_g (μ_g − μ)(μ_g − μ)ᵀ-normalised difference structure — for a 2-group split this is rank
≤ 1 by construction, so the pre-registered comparison uses the **between-vs-within ratio
spectrum**: eigenvalues of Σ_within^{-1/2} Σ_between Σ_within^{-1/2} restricted to the top-50
PCs, and the participation ratio of the *discriminative* spectrum of a 20-bin split of each
attribute (gender: M/F only → the 2-group rank; COMPAS race: the v1 race groups; toxicity: 4
toxicity-score quartiles). Pre-registered: gender PR ≈ 1 (≤ 1.5); COMPAS and toxicity PR > 3.
Correlate PR with v1's erasure success (gender succeeded, COMPAS/toxicity failed): 3 points only —
reported as a consistency check, not a test.

## B5 — Per-family calibration  [HPCC]
`hpcc/mejiro_ryan.py`: GPT-2, Pythia-410M, Gemma-2-2B; BiasTop nulls; n requirements.
