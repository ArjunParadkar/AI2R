# E6 — Dark-matter structure test (pre-registration)

Status: SPEC written 2026-08-30 **before** any extraction or analysis code in
`E6/` exists. `results/` is empty. Changes after the first run are logged as
numbered, dated amendments at the bottom; nothing above them is edited.

Rule-5 pause points were lifted by the owner for this experiment; all
judgment calls are made here and marked **[JC]**.

## Question

Is the SAE reconstruction error e = x − SAE(x) ("dark matter", Engels, Smith &
Tegmark 2024, *Decomposing the dark matter of sparse autoencoders*,
arXiv:2410.14670) isotropic noise, or does it carry structure — spectral
(Test 1), topological (Test 2), and predictive of the model's own computation
(Test 3)? Engels et al. already showed that a large part of the error is
*linearly predictable from x* and that error norms are predictable; we do not
re-claim that. What we add is (a) a random-matrix null (Marchenko–Pastur bulk
+ Tracy–Widom edge) for the error covariance with matched kill controls,
(b) the v1 topology-with-shuffle-null instrument on error point clouds, and
(c) a logit-delta probe with a shuffle control. Whether (a) is already in the
literature is checked by the novelty search reported in FINDINGS.md.

## Data and models **[JC]**

* Gemma-2-2b / Gemma Scope (Lieberum et al. 2024, arXiv:2408.05147) is the
  brief's first choice. `google/gemma-2-2b` is a gated repo and this machine
  has no HF token (checked 2026-08-30: `hf_hub_download` → 401
  `GatedRepoError`). **Gemma Scope is therefore NOT run**; the SAE weights are
  useless without the model's activations.
* Model: GPT-2 small (124M) via `transformer_lens.HookedTransformer`
  (`from_pretrained("gpt2-small")`, default TL processing: folded LayerNorm,
  centred writing weights — the same processing the SAEs below were trained
  against).
* SAEs: SAELens (Bloom, Tigges, Duong & Chanin 2024) release
  `gpt2-small-res-jb` (Joseph Bloom's residual-stream SAEs, d_sae = 24 576,
  standard ReLU/L1), hook `blocks.{L}.hook_resid_pre` at
  **L = 2 (early), 6 (middle), 10 (late)**. The SAE's own `encode`/`decode`
  are used as-is (including any input normalisation the SAE config specifies
  — we call `sae(x)` = decode(encode(x)) so that any such preprocessing is
  internal and e is measured in the model's own residual-stream units).
* Text: OpenWebText sample (`stas/openwebtext-10k` on the HF hub — the
  training distribution of these SAEs; fallback if unavailable:
  `wikitext-103-raw-v1` test split, and FINDINGS.md states which). Documents
  are tokenised with the GPT-2 tokeniser, prepended with BOS, and cut/packed
  to sequences of 512 tokens (no document crosses a sequence boundary;
  documents shorter than 512 are used as-is, padded sequences dropped). We
  take the first 128 sequences that reach 512 tokens → 65 536 token positions.
* **Position 0 (BOS) is excluded** from every analysis: the residual stream
  at BOS is a single fixed vector with a known enormous norm and its error is
  a single point repeated 128 times, which would produce a spurious rank-1
  outlier. N = 128 × 511 = 65 408 tokens per layer (≥ 50k as required).
* Precision: fp32 throughout. GPU budget ≤ 3 GB; all activations are moved
  to CPU and cached under `E6/cache/` (gitignored).

## Definitions

For a layer L, X ∈ ℝ^{N×d} (d = 768) are the resid_pre activations, R = SAE(X)
the reconstructions, E = X − R the errors. FVU = ‖E‖²_F / ‖X − mean(X)‖²_F is
reported per layer as a sanity check against the published values for these
SAEs (roughly 0.05–0.2 range; a value far outside means the SAE is being
misused and the run stops).

### Test 1 — RMT null

**Primary statistic (correlation form).** Standardise each error dimension
across tokens (subtract mean, divide by SD) → Ẽ; sample correlation matrix
C = ẼᵀẼ / N. Its eigenvalues λ₁ ≥ … ≥ λ_d are compared to the Marchenko–Pastur
law (Marchenko & Pastur 1967) with aspect ratio γ = d/N and σ² = 1:
λ± = (1 ± √γ)². **[JC]** The correlation form is primary because the MP law
with a single σ² assumes an isotropic diagonal; the error covariance has a
strongly heterogeneous diagonal (some residual-stream dimensions have far
larger variance), and an isotropic-σ² MP fit to such a matrix produces
outliers from the diagonal alone. Kill (i) below tests exactly this.

**Secondary statistic (covariance form).** Same on the raw covariance
S = EᶜᵀEᶜ / N (Eᶜ mean-centred) with σ² = trace(S)/d (mean diagonal). Reported
alongside; expected to over-count.

**Edge threshold.** Tracy–Widom (Tracy & Widom 1994; Johnstone 2001, *On the
distribution of the largest eigenvalue in principal components analysis*,
Ann. Stat. 29(2)): for a white Wishart with n samples and p dims,
λ_max·n ≈ μ + σ·TW₁ with μ = (√(n−1)+√p)², σ = (√(n−1)+√p)(1/√(n−1)+1/√p)^{1/3}.
An eigenvalue λ of the σ²-scaled sample covariance is an **MP outlier** if
λ > σ²(μ + σ·t_{0.99})/n, where t_{0.99} = 2.0234 is the 99th percentile of the
TW₁ distribution (Bejan 2005 tables). Since only the largest bulk eigenvalue
follows TW₁, the expected false-outlier count under the null is ≈ 0.01, so
"≈ 0" in the kills below means **≤ 1**.

**Structured fraction** = Σ_{outliers} λ / Σ λ (as the brief defines it).
We also report the **excess fraction** Σ_{outliers}(λ − λ₊)/Σλ, which removes
the bulk's share of each outlier and is the honest lower bound.

**Uncertainty.** Sequence-level block bootstrap (resample the 128 sequences
with replacement, B = 50), 95 % percentile CI on n_outliers and both
fractions.

**Kill attempts (all required; all on the same N, d, and same thresholds):**

* (i) **Gaussian, matched diagonal**: G ~ N(0, diag(Var(E))) i.i.d. across
  tokens. Expected: correlation form ≤ 1 outlier. The covariance form is
  *expected to produce outliers here* (it is the known weakness of an
  isotropic MP fit); that is why it is secondary.
* (ii) **Raw activations X** through the identical pipeline. This measures
  the "generic signal spectrum" of the residual stream. Pre-registered
  reading: if n_out(E) is comparable to or larger than n_out(X), the SAE is
  not removing the covariance structure of the residual stream — the error
  then inherits it and the outlier count says little about *extra*
  structure. We additionally report the outlier count of E after projecting
  out the top-k principal components of X (k = 10, 50) **[JC]**: outliers
  surviving that are not simply X's leading directions leaking through.
* (iii) **Feature-shuffled error**: each column of E independently permuted
  across tokens (destroys all cross-dimension correlation, preserves every
  marginal). Pass criterion for the test itself: ≤ 1 outlier in the
  correlation form. **If the shuffle produces > 1 outlier the test is broken
  and Test 1 is void** for that layer.

### Test 2 — Topology with shuffle null (v1 pipeline, reused)

`whitebox.signals.memorization.memorization_features` is the v1 instrument:
PCA to 30 dims (fitted on the cloud itself), Vietoris–Rips persistence
(ripser, filtration truncated at the 95th percentile of pairwise distances),
TC(H₁) = Σ(d−b)² (v1 Definition 4), β₁ at the median bar midpoint, and the
v1 **feature-shuffle null** (each PCA coordinate permuted independently
across points, n_shuffle = 5). We call it unchanged **[JC]**: 30 PCA dims
(brief allows ≤ 50), point cloud = a uniformly random subsample of
**n = 2 000 tokens** (from the N = 65 408, seed 0), the same for E, X, and the
Gaussian control.

Reported per layer: tc_ratio = TC/TC_shuffled, β₁, β₁ of the shuffles,
persistent entropy. **Bootstrap CI**: 20 independent 2 000-token subsamples
(seeds 1..20), 95 % percentile interval of tc_ratio.

Kills / references: the same statistic on (a) X, (b) Gaussian with the
error's full covariance N(0, Cov(E)) — a linear-Gaussian cloud whose TC ratio
tells us what "covariance but no topology" scores on this instrument; if
E's ratio is not above the Gaussian-matched-covariance ratio, the topology
signal is explained by second-order statistics alone.

### Test 3 — Predictive structure (following Engels et al. 2024)

**Target definition [JC].** The *logit delta* at token position t is
Δ_t = logits_t(clean run) − logits_t(run with resid_pre_L patched to SAE(x_t)
at every position), i.e. the full downstream effect of the error at layer L
on the model's output distribution at the same position (∈ ℝ^{50 257}).
Patching every position at once is the standard "SAE-spliced" evaluation
(same as the SAE's own CE-loss-recovered metric).

**Probe.** Ridge regression E → Δ (λ = 1.0 on standardised E, closed form via
768×768 normal equations, accumulated streaming on the GPU so Δ is never
stored). Train/held-out split by **sequence** (every 5th sequence held out,
26 of 128). Metric: held-out variance explained
R² = 1 − Σ‖Δ − Δ̂‖² / Σ‖Δ − Δ̄_train‖² over all vocab entries.
**Shuffle control**: the same probe fitted on feature-shuffled E (columns
permuted across tokens, within the train split; the permutation is applied
to E only, so the pairing with Δ is destroyed). Expected R²_shuffle ≈ 0 (≤
0.01). We also report R² for the probe **x → Δ** as an upper reference
(Engels et al.'s finding is that e is largely predictable from x, so this is
the ceiling of a linear account).

**Next-token identity.** Ridge regression from E to a one-hot of the
*next* token restricted to the V = 2 000 most frequent next tokens in the
train split (positions whose next token is outside V are dropped from this
sub-test; coverage reported). Held-out **top-1 and top-5 accuracy** vs (a)
the same on shuffled E, (b) the majority-class baseline, (c) the same probe
from X (reference ceiling). **[JC]** Ridge-to-one-hot instead of softmax
regression because it is closed-form and the question is presence of linear
information, not calibrated probabilities.

## Pre-registered hypothesis and criteria

**H-E6 (per layer): the dark matter is structured if n_outliers > 10 in the
primary (correlation-form) Test 1 AND tc_ratio > 1.2 in Test 2.**

Operationalisation **[JC]**:
* n_outliers > 10 with the bootstrap 95 % CI lower bound > 10, AND the
  kill (iii) shuffle gives ≤ 1 outlier (otherwise the test is void, not
  passed).
* tc_ratio point estimate > 1.2 AND its 95 % bootstrap CI lower bound > 1.0
  (so the interval excludes "no topology beyond marginals").
* Both must hold for the layer to pass. Test 3 is **not** part of H-E6's
  pass/fail; it is reported as the predictive-structure deliverable with its
  own shuffle control (R² > 0.05 on held-out with shuffle ≤ 0.01 is called
  "linearly predictive").

Interpretive commitments made now:
* If E passes Test 1 but the Gaussian-with-Cov(E) control matches its
  tc_ratio, the verdict is "spectrally structured, topologically
  Gaussian-like" — not "structured computation".
* If n_out(E) ≈ n_out(X) and the PC-projected-out counts drop to ≤ 10, the
  spectral structure is inherited residual-stream covariance, and we say so.
* e is a deterministic function of x, so "e contains information about x"
  is a triviality. Test 3's Δ probe is about the *downstream effect* of e,
  which is not trivial; the X → Δ ceiling calibrates it.
* Negatives are deliverables.

## Controls on the instrument itself (unit tests, `E6/tests`, run before the
real data)

1. Pure i.i.d. Gaussian (N = 4 000, d = 200): n_outliers ≤ 1 in both forms;
   structured fraction < 0.02.
2. Planted rank-k signal (k = 5 spikes with eigenvalue well above the BBP
   threshold (Baik, Ben Arous & Péché 2005), i.e. spike ≥ 5·(1+√γ)² ) on
   i.i.d. noise: n_outliers = 5 exactly.
3. The feature-shuffle on the planted data: ≤ 1 outlier (null works).
4. `residual_structure` on a planted-signal cloud returns the fields the
   Stage-2 `whitebox.darkmatter` API requires (structured_fraction,
   n_outliers, mp_edge, tc_ratio, ci).
5. Test 3 probe machinery: a synthetic linear target recovers R² ≈ 1 on
   held-out and ≈ 0 under the shuffle.

## Figures (300 dpi)

`fig_spectrum_L{2,6,10}.png` — eigenvalue histogram of C vs MP density with
the TW edge and outliers marked, with the Gaussian/shuffle/X kills as
insets or panels; `fig_tc_ratio.png` — TC ratios with CIs for E, X, Gaussian
per layer, the 1.2 line drawn; `fig_probe.png` — held-out R² and top-k
accuracy for E, shuffled E, X per layer.

## Runtime budget

Extraction: 3 layers × 128 sequences, GPT-2 small; minutes. Test 3 needs
three extra forward passes per layer (clean + patched, twice for train/test
accumulation); minutes. Test 2: PH on 2 000 points × (1 + 5 shuffles) × 21
subsamples × 3 clouds × 3 layers ≈ 1 100 ripser calls on 30-dim clouds —
the most expensive part; expected < 1 h on 20 cores, run with
multiprocessing.

## Amendments

(none yet)

### Amendment 1 (2026-08-30, after the first extraction run; before any Test 1/2 analysis)

The first extraction gave FVU = 619 / 2.8 / 0.58 at L2/6/10 — the SPEC's own
stop condition. Two causes, both loader errors on our side, neither a change
to the hypotheses or criteria:

1. The `gpt2-small-res-jb` SAEs were trained on activations from
   `HookedTransformer.from_pretrained_no_processing("gpt2-small",
   center_writing_weights=True)` (recorded in the SAE metadata
   `model_from_pretrained_kwargs`), i.e. **without LayerNorm folding**. The
   SPEC's "default TL processing" statement was wrong; the model is now
   loaded exactly as the metadata prescribes.
2. The SAEs were trained with `context_size = 128`. On 512-token sequences the
   reconstruction is fine for positions 1–127 (FVU 0.059 at L2, L0 ≈ 19) and
   catastrophic for positions ≥ 128 (FVU ≈ 10³, L0 ≈ 2000) — the SAE is
   out of distribution there, and its "error" is not dark matter but
   nonsense. Sequences are therefore **128 tokens** long (BOS + 127), and we
   use **512 sequences** → N = 512 × 127 = 65 024 tokens per layer (still
   ≥ 50k). Position 0 still excluded. The held-out split is still every 5th
   sequence (103 of 512); the block bootstrap resamples the 512 sequences.

The first run's probe numbers are discarded (they measured an OOD SAE).
The FVU sanity range in the Definitions section (0.05–0.2) now applies.

### Amendment 2 (2026-08-30, after Test 1 on layer 2; before layers 6/10 were inspected)

The sequence-level bootstrap in `block_bootstrap_outliers` resamples sequences
**with replacement**. Applied to the feature-shuffled matrix this duplicates
whole rows of a matrix whose columns were shuffled independently, which
re-introduces cross-column dependence (a duplicated row is identical in every
column) and produced a spurious shuffle CI of ~(57, 71) outliers while the
full-data shuffle count was 0. This is the same resampling trap as E1
amendment 5. Fix (supplementary script `extra_e6.py`, results in
`e6_test1_shuffle_seeds.csv` / `e6_test1_subsample_ci.csv`):
* the shuffle control's uncertainty is reported over **20 independent shuffle
  seeds** on the full data (the correct null replicate);
* for E and X we add a **without-replacement** subsample CI (20 draws of 80 %
  of the sequences) alongside the with-replacement CI; both are reported, the
  without-replacement one is used for the H-E6 criterion "CI lower bound > 10".
Nothing else changes. The Test 2 point estimate (seed-0 subsample) and its CI
(20 further subsamples) are reported as computed; where the point falls outside
its own CI we quote the bootstrap mean alongside, and the SPEC criterion is
applied to both.

### Amendment 3 (2026-08-30, after Test 3 numbers were seen; before Test 1/2 were complete)

Test 3a's target Δ = logits(clean) − logits(patched) has a held-out mean norm of
1 200–2 300 over 50 257 vocab entries (≈ 5–10 logits per entry), far larger
than the CE-loss cost of splicing these SAEs suggests. The raw-logit
difference contains a per-token **common-mode** component (the same shift on
every vocab entry) that the softmax ignores; it dominates Σ‖Δ‖² and is
irrelevant to the model's output distribution. We therefore add a second,
better-defined target and report both: Δ_lp = log_softmax(clean) −
log_softmax(patched) (the change in next-token log-probabilities, invariant
to common-mode shifts). Same probe, same split, same shuffle control
(`probe_logprob_e6.py`, `e6_probe_logprob.csv`). The pre-registered
statement that Test 3 is not part of the H-E6 pass/fail is unchanged.

### Amendment 4 (2026-08-30, write-up pass by the reviewing agent, after a machine reboot at
~13:14 killed `run_e6.py` mid-run)

The reboot hit during Test 1's `np.savez` call for layer 10 (immediately
after `e6_test1_rmt.csv` and `fig_spectrum_L10.png` had already been written
to disk, so Test 1 is complete and unaffected for all three layers; only
`e6_spectra_L10.npz` was left as a 0-byte truncated file). Test 2
(topology bootstrap) had not yet run for layer 10 at all -- `e6_test2_topology.csv`
had rows for L2/L6 only, and `fig_tc_ratio.png`, `fig_probe.png`, and
`e6_verdict.csv` (all computed after the full per-layer loop in `run_e6.py`)
did not exist. Three gap-filling, no-new-judgment-call actions, all CPU-only
(no GPU touched; the SAE weights and `HookedTransformer` were not reloaded --
`cache/L10.npz`, already on disk, has everything Test 2 needs):

1. `extra_e6_test2_L10.py` reruns exactly `run_e6.py`'s `test2()` for layer
   10 (E, X, gauss_cov(E); n_points=2000, n_boot=20, same seeds), appends to
   `e6_test2_topology.csv`, and regenerates `e6_spectra_L10.npz` from
   `cache/L10.npz`. Took ~30 min wall-clock (vs. the SPEC's ~20 min/layer
   estimate) because the machine was concurrently running E1 and E3 jobs at
   load average 30-37 on 20 cores; this is a compute-environment fact, not a
   methods change.
2. `run_e6.py`'s `fig_tc()` has a latent bug never previously triggered
   (the original run never reached it): it passes `yerr` to
   `matplotlib.errorbar` without clipping, and layer 2's `tc_ratio` point
   estimate (seed-0 subsample) falls *above* its own 95% bootstrap CI
   (built from 20 different, seeds 1-20, subsamples) -- `errorbar` raises on
   a negative `yerr`. Fixed by clipping `yerr` at 0 in the regenerated
   `fig_tc_ratio.png`; no numbers changed, only the plotting call.
3. The point-estimate-outside-its-own-CI case is exactly what Amendment 2
   said would be handled by quoting the bootstrap mean alongside; the
   verdict table (`e6_verdict.csv`) was written for the first time by this
   pass and reports both `tc_ratio_E_point` and `tc_ratio_E_boot_mean`
   against the H-E6 threshold (`test2_pass_point`, `test2_pass_bootmean`).
   Separately, the **original (never-run) verdict-table code in
   `run_e6.py`** used the *with-replacement* block-bootstrap CI
   (`e6_test1_rmt.csv`'s `n_outliers_ci`) for the H-E6 Test-1 CI criterion,
   which Amendment 2 already says is not the one to use (that amendment
   predates this code path ever executing) -- the without-replacement
   subsample CI (`e6_test1_subsample_ci.csv`) is used instead in
   `e6_verdict.csv`, per Amendment 2. This is a conformance fix to
   already-decided text, not a new judgment call, and does not change any
   pass/fail outcome (both CIs clear the >10 bar by two orders of
   magnitude at every layer).

No hypothesis, criterion, or dataset changed. All of Test 1, Test 2, Test 3a,
3a', and 3b are now complete for all three layers; see FINDINGS.md.
