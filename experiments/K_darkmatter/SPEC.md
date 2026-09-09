# Block K — Dark matter: what the tools miss (pre-registration)

_Naming: every run in this document is a horse from the Tracen roster (`experiments/HORSES.md`); the handbook codes below are the pre-registration labels and appear here only as such._

Status: SPEC written 2026-09-07 before any Block-K run. `results/` empty. Changes after
results exist go to `AMENDMENTS.md`. Novelty: **OPEN as application** (MP null on transformer
weights exists, Sept 2025; on SAE residuals not found — search logged in NOVELTY.md). K7
**FLAGGED** (Engels et al.) — search before write-up; K7 is HPCC anyway.
Prior in-house result: E6 (2026-08-30, GPT-2 small, `gpt2-small-res-jb` L2/6/10, 65k tokens):
MP spectral null passed everywhere (179–198 outliers of 768), the v1 topology null FAILED
everywhere (tc_ratio 0.11–0.46, *below* the shuffle null) → "spectrally structured only".
Block K re-runs the handbook's protocol (layers 3/6/9, 50k tokens, pre-registered >10 outliers,
ratio > 1.2), adds the predictive (K4) and nonlinear (K5) tests, and the invariance certification
(K6). E6 is cited as the prior, not reused as evidence.

## Data and models
* GPT-2 small via TransformerLens (`gpt2-small`, default processing) + SAELens release
  `gpt2-small-res-jb`, hooks `blocks.{3,6,9}.hook_resid_pre` (the release's SAEs; d_sae 24 576).
  50 000 tokens of OpenWebText-10k (400 sequences × 125 tokens after the BOS token).
* Gemma-2-2B + Gemma Scope (matching layers): **blocked** — `google/gemma-2-2b` is gated and this
  machine has no Hugging Face token (checked again 2026-09-07). Recorded as NOT RUN, not as a
  null; runs on HPCC (K7/K8 already HPCC) once a token is available.
* Environment: `~/Whitebox/v2-programme/.venv-tl` (TransformerLens + SAELens 6.50).

## K1 — Residuals  [5070]
e = x − SAE(x) per token (BOS excluded), saved with x, token ids, sequence id, position:
`cache/L{3,6,9}.npz`. Reconstruction check: FVU = Σ‖e‖² / Σ‖x − x̄‖² must be within 10 % of the
release's published value (SAELens `gpt2-small-res-jb` reports variance explained per layer; the
value read from the SAE config / release card is recorded in `results/K1_reconstruction.csv`).
If no published number is available for a hook, the check is reported as "no published value"
rather than passed.

## K2 — Spectral null  [5070]
Covariance of e across tokens (correlation form primary, covariance form secondary, as in E6);
`nulls.marchenko_pastur` with γ = d/n and σ² from the bulk median; outliers above the
Tracy–Widom edge at p < 0.01; structured fraction = Σ outlier eigenvalues / trace.
**Pre-registered: > 10 outliers per layer.**
Kills: (i) synthetic Gaussian e with matched marginals (per-column mean and sd) → ≤ 1 outlier;
(ii) variance-matched, feature-shuffled real e → ≤ 1 outlier. Sequence-block bootstrap (B = 50)
CI on the outlier count.

## K3 — Topological null  [5070]
v1 pipeline on e clouds: PCA-20, 2 000 points, 20 bootstraps (fresh 2 000-point subsamples);
total persistence and β1 (H1 bar count) vs `nulls.feature_shuffle` of the same subsample.
**Pre-registered: ratio TP(e)/TP(shuffle) > 1.2 with a 95 % bootstrap CI excluding 1.** The E6
covariance-matched Gaussian control (ratio ≈ 1 expected) is reported alongside as a calibration.

## K4 — Predictive content  [5070]
Ridge regression e → Δlogits, Δ = logits(clean) − logits(SAE-reconstruction patched in at the
hook), on the top-2 000 vocabulary entries by frequency in the corpus (memory), λ chosen on a
validation split of sequences; held-out R² (20 % of sequences) vs feature-shuffled-e R².
Pre-registered: R²(e) − R²(shuffled e) > 0.05.

## K5 — Nonlinear residue  [5070]
Random-projection quadratic features φ(e) = [e, (P e)²] with P a 2 000 × d Gaussian projection
(2 000 quadratic features); ridge to Δlogits; held-out variance explained beyond the linear
model (ΔR² = R²(φ) − R²(linear)). Pre-registered: ΔR² > 0.02.

## K6 — Certified dark matter  [5070]
K2 (outlier count, structured fraction) and K3 (ratio) statistics through `audit_quantities`
under GPT-2 renamings (head permutations, within-head basis changes, MLP-unit permutations,
LN-compatible rescalings; n = 20 per kind). The SAE dictionary is transformed consistently:
the residual stream is NOT acted on by any of these renamings (they are all inside blocks),
so e is expected exactly invariant — this is the sanity anchor for the certification, and the
non-trivial part is stated as such. Additionally the same statistics are computed on the
**MLP-hidden-space** residual of a per-layer PCA-k reconstruction (k = 256), where MLP-unit
permutations do act, so that the invariance class is not vacuous.

## K9 — Ship `residual_structure`  [LAPTOP]
`whiteboxtopo.darkmatter.residual_structure` already exists (E6 port). Unit test added:
synthetic noise → structured fraction < 0.02 (and ≤ 1 outlier). VALIDATION.md row updated.

## Pass/fail summary
| Item | Threshold |
|---|---|
| K1 | FVU within 10 % of published (or "no published value") |
| K2 | > 10 outliers per layer; kills ≤ 1 outlier |
| K3 | ratio > 1.2, CI excludes 1 |
| K4 | R²(e) − R²(shuffle) > 0.05 |
| K5 | ΔR² > 0.02 |
| K6 | invariance class reported; residual-stream anchor exactly invariant |
| K9 | noise → structured fraction < 0.02 |
