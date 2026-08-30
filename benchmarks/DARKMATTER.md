# `whitebox.darkmatter` — the E6 port (Stage 2B, 2026-08-30)

Input: E6 of whitebox-v2 (its result tables are copied verbatim into
`results/darkmatter/`; its code was ported, not imported — the library has
no dependency on `transformer_lens` / SAELens / torch for this module).
E6's own docs: `E6/SPEC.md` (pre-registration + four amendments) and
`E6/FINDINGS.md` (verdict) in the whitebox-v2 experiment tree.

## What ships

`whitebox.darkmatter.residual_structure(errors)` — an **instrument**, not a
claim. Two independent readings of a reconstruction-error cloud e = x −
recon(x) (e.g. an SAE's error, but the function takes plain numpy arrays and
has no opinion about where `errors` came from):

* **Test 1 (spectral)**: Marchenko–Pastur bulk + Tracy–Widom edge null on
  the sample correlation matrix of `errors` (`form="correlation"` is
  primary — an isotropic-σ² MP fit to a heterogeneous-diagonal covariance
  produces spurious outliers from the diagonal alone; `form="covariance"`
  is shipped too, documented as over-counting). `spectral_test_valid` is
  the feature-shuffle kill (columns permuted independently — marginals
  kept, cross-dimension correlation destroyed): if the shuffle itself
  produces > 1 outlier, Test 1 is **void** on that data and
  `spectrally_structured` is forced `False` regardless of the raw count.
  `n_outliers_ci` is the **without-replacement** subsample CI (E6 SPEC
  Amendment 2 — see "judgment calls" below for why with-replacement is
  wrong here).
* **Test 2 (topological)**: the v1 TC(H₁)-with-feature-shuffle-null
  instrument (`whitebox.signals.memorization.memorization_features`,
  reused unchanged — PCA→30, VR persistence via `whitebox.core`, 95th-
  percentile filtration truncation) on a random subsample of `errors`,
  `tc_ratio` = TC(H₁) / TC(H₁) of the feature-shuffled null, with a
  bootstrap CI over independent subsamples and a **covariance-matched
  Gaussian control** (`N(mean(errors), Cov(errors))` — "covariance but no
  extra topology"; `topologically_structured` additionally requires
  `tc_ratio` to exceed this control's own ratio, per E6's pre-registered
  interpretive commitment that merely matching the covariance-only control
  means "topologically Gaussian-like", not "structured computation").

`residual_structure` also accepts `acts=`/`recon=` instead of a
precomputed `errors=` array (`errors = acts - recon` is computed for you —
the "(acts, recon) pairs" calling convention from PROGRAM.md).

Test 3 (E6's logit-delta ridge probe) is **not ported**: it answers a
different question (is e linearly predictive of the model's own output — a
predictive-structure deliverable, never part of H-E6's pass/fail) and needs
a live model + SAE (torch, transformer_lens, SAELens) this library does not
depend on. `E6/e6lib/probe.py` has the reusable streaming-ridge machinery if
a future module wants it.

## Verdict discipline (mirrors `whitebox.legibility`)

`DarkMatterReport.verdict ∈ {"neither", "spectrally_structured",
"topologically_structured", "both"}` describes **which instrument(s)
cleared their own pre-registered bar on this cloud** — it is not a claim
that dark matter, in general, is structured. E6's own pre-registered
hypothesis (H-E6: n_outliers > 10 with CI lower bound > 10 AND tc_ratio >
1.2 with CI lower bound > 1.0, **both** required) **failed at every layer
tested**. On the only battery run so far every layer's verdict was
`"spectrally_structured"` — never `"topologically_structured"` or `"both"`.
A `tc_ratio` below 1.0 (as E6 found, at every layer) means the error cloud
is **less** topologically complex than its own feature-shuffled null — a
specific negative finding, not "no structure".

## The E6 battery (GPT-2 small, 768 dims, 65k tokens; one release; Gemma untested)

`gpt2-small-res-jb` (SAELens, standard ReLU/L1 residual-stream SAEs,
d_sae = 24 576), `blocks.{2,6,10}.hook_resid_pre`, OpenWebText sample, 512
sequences × 127 non-BOS tokens = 65 024 tokens/layer (E6 SPEC Amendment 1:
sequences are 128 tokens long, matching these SAEs' training
`context_size` — the SPEC's original 512-token sequences put the SAE badly
out of distribution past position 128).

| Layer | n_outliers (of 768) | CI (no-replace) | shuffle valid | Test 1 | tc_ratio(E) | CI | gauss-cov control | Test 2 | **H-E6** |
|---|---:|---|---|---|---:|---|---:|---|---|
| 2  | 198 | (189.0, 192.0) | yes (0) | **PASS** | 0.462 | (0.348, 0.428) | 0.874 | **FAIL** | **FAIL** |
| 6  | 194 | (182.0, 184.0) | yes (0) | **PASS** | 0.112 | (0.090, 0.117) | 0.995 | **FAIL** | **FAIL** |
| 10 | 179 | (169.5, 172.0) | yes (0) | **PASS** | 0.155 | (0.123, 0.164) | 1.011 | **FAIL** | **FAIL** |

Test 1 passes cleanly and by a wide margin at every layer. Test 2 does not
just miss the 1.2 bar — `tc_ratio(E)` is **below 1.0** at every layer, and
well below the covariance-matched Gaussian control (which correctly
calibrates to ~1.0, as the SPEC expected of "covariance but no extra
topology"). Kill (i) (Gaussian, matched diagonal) gives 0 outliers in the
correlation form at all three layers — the spectral outliers are not an
artifact of the error's heterogeneous per-dimension variance. Kill (iii)
(feature-shuffle) gives exactly 0 outliers at every layer, across 20 shuffle
seeds — Test 1 is not void. Full account, including the PC-projection
refinement of kill (ii) (spectral structure survives removing the raw
activations' top-50 principal directions) and Test 3's numbers: E6's
`FINDINGS.md`.

Battery record copied here: `results/darkmatter/e6_test1_rmt.csv`,
`e6_test1_shuffle_seeds.csv`, `e6_test1_subsample_ci.csv`,
`e6_test2_topology.csv`, `e6_verdict.csv`. `tests/test_darkmatter.py`
reproduces the headline numbers directly from these files.

## Judgment calls made porting E6 → the library

* **Grouping default.** E6 resamples whole *sequences* (documents); the
  library's `residual_structure` has no notion of a sequence unless the
  caller supplies `groups=` (a per-row id). Default: every row is its own
  group, i.e. i.i.d. row subsampling. A caller with sequence-structured
  data (tokens within a document are not exchangeable) should pass
  `groups=` to reproduce E6's exact resampling unit.
* **With-replacement vs without-replacement CI (E6 SPEC Amendment 2).**
  `block_bootstrap_outliers` (with replacement) is shipped for comparison
  only, with a loud docstring warning: resampling with replacement
  duplicates whole groups, and a duplicated row is identical in every
  column of a feature-shuffled matrix, which re-introduces the very
  cross-column dependence the shuffle exists to destroy — the same trap as
  whitebox-v2 E1's Amendment 5. `residual_structure` uses
  `subsample_outliers` (without replacement) by default, per Amendment 2.
* **`n_shuffle_seeds` default is 5, not E6's 20** (speed; the library
  default is enough to catch a broken null on non-Gaussian data — see
  `tests/test_darkmatter.py`'s heavy-tailed control — but a user chasing a
  borderline case should raise it).
* **`gaussian_control=True` gates `topologically_structured`** on
  `tc_ratio(E) > tc_ratio(gauss_cov(E))`, not just on the raw 1.2/CI bar.
  This was E6's stated interpretive commitment (SPEC: "if E passes Test 1
  but the Gaussian-with-Cov(E) control matches its tc_ratio, the verdict is
  spectrally structured, topologically Gaussian-like — not structured
  computation") but was never mechanically enforced in E6's own
  `e6_verdict.csv` (which only checked the 1.2/CI bar, because tc_ratio
  never got close to 1.2 in the real battery). Encoding it as a gate here
  is a genuine (if inert on the one existing battery) strengthening; flagged
  in case a future battery disagrees with this reading.

## Scope

One battery: GPT-2 small (124M), one SAELens release
(`gpt2-small-res-jb`, standard ReLU/L1, d_sae = 24 576), 3 layers, one
dataset (OpenWebText sample). Gemma-2-2b / Gemma Scope was never run (gated
HF repo, no token on this machine — checked and logged in E6/SPEC.md before
any compute was spent). Nothing here has been checked on a second SAE
release, a second model family, or a second architecture (e.g. attention
SAEs, transcoders). The topology instrument (Test 2) inherits v1's own
scope notes (`benchmarks/PHASE6.md`, `PHASE2.md`): TC/β₁/PE are size-
dependent (compare at fixed subsample size only) and PCA→30 is a lossy
reduction whose adequacy was checked on v1's manifolds, not on SAE error
clouds specifically — the noisy-circle-in-d-dims planted control in
`tests/test_darkmatter.py` is the closest available check.
