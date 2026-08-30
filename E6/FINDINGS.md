# E6 — Dark-matter structure test: FINDINGS

**Status: COMPLETE (2026-08-30).** The original run was killed mid-way by a
machine reboot at ~13:14 (Test 1 had just finished layer 10 and was
mid-`np.savez`; Test 2/topology for layer 10 had not started). The gap was
CPU-only and cheap to fill from the already-cached activations
(`cache/L{2,6,10}.npz`) -- no GPU touched, no model or SAE reloaded. See
SPEC.md Amendment 4 for exactly what was missing and what was re-run. All
three layers now have complete Test 1, Test 2, Test 3a, 3a' (amendment 3),
and 3b results. `tests/test_e6.py`: **5/5 green**.

GPT-2 small (124M), `gpt2-small-res-jb` SAELens residual-stream SAEs,
`blocks.{2,6,10}.hook_resid_pre`, OpenWebText (`stas/openwebtext-10k`),
512 sequences x 127 non-BOS tokens = 65 024 tokens/layer (Amendment 1: the
SAEs were trained at `context_size=128`, not 512 -- the SPEC's original
512-token sequences put the SAE badly out of distribution past position
128; fixed before any Test 1/2/3 numbers were taken). Gemma Scope was not
run (no HF token, gated repo, checked 2026-08-30 -- SPEC's own account).

## Verdict

**H-E6 (per layer): dark matter is structured if n_outliers > 10 (Test 1,
CI lower bound > 10) AND tc_ratio > 1.2 with CI lower bound > 1.0 (Test 2).
Both must hold.**

| Layer | Test 1 (n_out, corr. form) | CI (no-replace, the amendment-2 criterion) | Test 1 valid (shuffle <= 1) | Test 1 pass | Test 2 tc_ratio (E) | CI | Test 2 pass | **H-E6** |
|---|---|---|---|---|---|---|---|---|
| 2  | 198 | (189.0, 192.0) | yes (shuffle n_out = 0) | **PASS** | 0.462 (boot mean 0.385)* | (0.348, 0.428) | **FAIL** | **FAIL** |
| 6  | 194 | (182.0, 184.0) | yes (shuffle n_out = 0) | **PASS** | 0.112 | (0.090, 0.117) | **FAIL** | **FAIL** |
| 10 | 179 | (169.5, 172.0) | yes (shuffle n_out = 0) | **PASS** | 0.155 | (0.123, 0.164) | **FAIL** | **FAIL** |

\* Layer 2's point estimate (seed-0 subsample) falls *above* its own 20-seed
bootstrap CI; both point and bootstrap-mean readings fail the same way (SPEC
Amendment 2's contingency). Doesn't matter for the verdict either way.

**H-E6 fails at every layer.** Test 1 passes cleanly and by a wide margin
(n_outliers ~180-200 of 768 dimensions, CI lower bound two orders of
magnitude above the >10 bar). Test 2 does not just fail to clear 1.2 -- the
error clouds' TC(H1) ratio is **below 1.0 at every layer** (0.11-0.46),
meaning the error point cloud has *less* total persistence than its own
feature-shuffled null, and is also well below the raw-activation clouds
(0.81-0.94) and the covariance-matched Gaussian control (0.87-1.01, which
correctly calibrates to ~1 as the SPEC expected). So the combined reading is
**not** "spectrally structured, topologically Gaussian-like" (the SPEC's
pre-registered fallback framing for a Test-1-only pass) -- it is spectrally
structured and topologically *simpler* than a linear-Gaussian model matched
to the same covariance, by this instrument. See "what the data show" below
for what that number can and cannot mean.

## What the data actually show

1. **The spectral outliers are real, robust, and not simply the SAE's own
   isotropic-diagonal artifact.** Kill (i) (Gaussian, matched diagonal)
   gives 0 outliers in the correlation form at all three layers, confirming
   the primary statistic is not just picking up the error's heterogeneous
   per-dimension variance (the SPEC's own concern about the isotropic-sigma^2 MP
   fit). The covariance form *does* pick up spurious outliers here (91/30/25
   at L2/6/10) exactly as the SPEC predicted -- supporting the SPEC's choice
   of the correlation form as primary.
2. **Kill (iii), the instrument sanity check, passes at every layer**:
   feature-shuffling E gives exactly 0 outliers (correlation form), both on
   the full data and across 20 independent shuffle seeds
   (`e6_test1_shuffle_seeds.csv`, max over 60 seed x layer draws = 0). Test 1
   is not void.
3. **Kill (ii), the "is this just X's covariance leaking through" check, is
   genuinely ambiguous on the first reading and resolved by the
   pre-registered PC-projection refinement.** n_out(E) is *comparable to or
   larger than* n_out(X) at every layer (198 vs 194 at L2, 194 vs 180 at L6,
   179 vs 151 at L10) -- by the SPEC's first-order reading this alone would
   mean "the outlier count says little about extra structure." But
   projecting out X's top-10 or top-50 PCs from E does **not** collapse the
   outlier count to <=10 (it stays at 184-220, if anything rising slightly
   at k=50): 201/199/184 (k=10), 212/220/208 (k=50) at L2/6/10. So the
   spectral structure in E is not simply X's leading directions leaking
   through the SAE unmodified -- whatever generates it lives in directions
   beyond X's top-50 principal subspace. This is a genuine, non-trivial
   finding: **structured, but not because the SAE failed to remove the
   residual stream's own dominant covariance.**
4. **Test 2's negative result is the most informative single number here
   and cuts against a naive reading of Test 1.** If the RMT outliers
   reflected "structured computation" in any sense that a shuffle-null
   topology instrument should pick up, we'd expect tc_ratio(E) to at least
   match X's (0.81-0.94) or the Gaussian-matched-covariance control's
   (0.87-1.01). Instead E's ratio is 4-8x *smaller*. A plausible (not
   confirmed) mechanical reading: these SAEs' L0 ~ 15-30 ReLU/L1 sparsity
   means many tokens' reconstructions differ from each other only in a
   handful of active-feature directions, which can make pairwise distances
   in the 2 000-point subsample *more redundant* (many near-equal
   short/long distances) than an equivalent Gaussian -- exactly the kind of
   degeneracy that *lowers* total persistence relative to a shuffle that
   spreads the same marginals independently. We flag this as a hypothesis
   for anyone extending this measurement, not a conclusion; we did not run
   the follow-up (e.g., relating tc_ratio to the SAE's L0) that would test
   it.
5. **Predictive structure (Test 3, not part of H-E6) is real but small, and
   grows with depth -- with one genuine surprise.** On the honest,
   softmax-invariant target (Amendment 3, log-prob delta): held-out
   R^2(E->Delta_lp) = 0.022 (L2, **below** the 0.05 "linearly predictive"
   bar), 0.066 (L6, above), 0.283 (L10, above, and *higher than the
   X->Delta_lp ceiling*, 0.264). Shuffle control is ~0 everywhere (-0.012 to
   -0.015, i.e. no signal, as expected -- small negative held-out R^2 is the
   normal signature of a useless predictor under ridge, not evidence of
   anything). The L10 reversal (E beats X) is not paradoxical: Delta_lp is
   *defined* as the effect of splicing SAE(x) for x, so to leading order
   Delta_lp ~ -J(x->logits)*e is a direct (near-linear, for a fixed local
   Jacobian) function of e itself, whereas x's relationship to Delta_lp is
   only indirect (through the same Jacobian applied to whatever part of x
   maps to e via the SAE). This is expected once you write down what Delta
   is a function of, not evidence that e "knows more about the model's
   computation" than x does. On the raw (pre-amendment) logit-delta target
   the numbers are noisier and mostly uninformative (R^2(E) as low as
   -0.015, as high as 0.113) -- Amendment 3's diagnosis (a large common-mode
   term the softmax ignores) is why we do not lead with that target.
6. **Next-token identity (Test 3b) confirms modest but real, above-null
   information in the error.** Held-out top-5 accuracy for E: 0.32/0.28/0.31
   (L2/6/10) vs shuffled-E 0.17/0.17/0.17 vs majority-class 0.047 vs the
   X ceiling 0.45/0.48/0.55. E carries clearly more next-token information
   than chance or its own shuffle, at roughly 60-70% of X's ceiling --
   again, this is expected (e is a deterministic function of x, so some
   leakage of x's own predictive content through e is not surprising) and
   calibrated by the X ceiling as the SPEC intended.
7. **FVU sanity check passes** (0.055 / 0.122 / 0.186 at L2/6/10, inside the
   SPEC's 0.05-0.2 band; L10 is close to the top of that band but inside
   it).

## Kill attempts (SPEC's required list, all run)

* (i) Gaussian, matched diagonal: 0 outliers (correlation), confirming the
  primary statistic isn't an artifact of the heterogeneous diagonal.
  Covariance form does show outliers here (25-91), exactly as SPEC
  predicted for the secondary/weaker form.
* (ii) Raw activations X through the identical pipeline: n_out(X) is
  comparable to n_out(E) at every layer (see point 3 above) -- the
  PC-projection refinement (k=10, 50) shows the outliers survive removal of
  X's leading directions, so this kill does **not** reduce Test 1 to
  "inherited covariance."
* (iii) Feature-shuffled error: 0 outliers everywhere, both on the full data
  and across 20 shuffle seeds -- Test 1 is valid (not void) at every layer.
* Test 2's own kill (Gaussian with Cov(E)): calibrates to tc_ratio ~ 1.0
  (0.87-1.01) as intended, and E's ratio (0.11-0.46) is well below it -- the
  opposite direction from "spectrally structured but topologically
  Gaussian-like."
* Test 3's shuffle control: R^2 ~ 0 (-0.012 to -0.019) at every layer on
  both targets -- the probe machinery is not finding spurious signal.

## Judgment calls and amendments

* Amendment 1: model must be loaded via `from_pretrained_no_processing`
  (not the SPEC's stated default TL processing) and sequences must be
  128 tokens (matching the SAEs' training `context_size`), not 512 -- the
  first extraction's FVU (619/2.8/0.58) was the SPEC's own stop condition,
  and both causes were loader errors, not framing changes.
* Amendment 2: the block bootstrap (with replacement over sequences)
  produces a spurious CI when applied to a feature-shuffled matrix (~(57,71)
  outliers where the full-data shuffle count is 0) -- same trap as E1's
  amendment 5. Fixed with 20 independent shuffle seeds (full data) for the
  shuffle null's uncertainty, and a without-replacement subsample CI for
  E/X, used for the H-E6 criterion.
* Amendment 3: Test 3a's raw-logit-delta target is dominated by an
  irrelevant common-mode shift; added the log-prob-delta target as the
  honest metric, reported alongside.
* Amendment 4 (this pass): documents the reboot gap (Test 2 layer 10, the
  three downstream figures/verdict table that depend on all three layers)
  and three code-conformance fixes made while filling it -- none change a
  pre-registered hypothesis, criterion, or dataset; see SPEC.md for the
  full account. Concretely: Test 2 for layer 10 (E, X, gauss_cov(E)) was
  re-run from the untouched cached activations (`extra_e6_test2_L10.py`,
  CPU-only, ~30 min under heavy concurrent load from E1/E3 -- no GPU used);
  a plotting bug in `fig_tc()` (unclipped negative `yerr`, triggered by
  layer 2's point-outside-CI case) was fixed; and the verdict table
  (`e6_verdict.csv`, computed for the first time here since the original
  run never reached that code) uses the without-replacement CI per
  Amendment 2, not the with-replacement one the never-executed
  `run_e6.py` verdict block would have used -- this does not change any
  pass/fail outcome (margins are ~180 vs the >10 bar either way).

## Novelty check (see `results/novelty_check.md` for the full record)

The RMT (Marchenko-Pastur bulk + Tracy-Widom edge) null applied to the SAE
*reconstruction error* is not taken in what web search found. MP has been
applied to SAE *feature* (decoder-direction) covariance (Li et al.,
*Geometry of Concepts*, finding a power law instead of MP) and to
transformer *weight* matrices (two 2026 preprints), and "does the residual
look like a random matrix" is a known idea in matrix completion. Framed
honestly as a measurement contribution (a standard instrument applied to a
new target with kill controls), not a new theory. Caveat: web search only,
no arXiv full-text search -- a workshop paper could have been missed.

## Relation to the literature

* Engels, Smith & Tegmark 2024 (arXiv:2410.14670): we do not re-claim their
  finding that e is largely linearly predictable from x (Test 3's X->Delta
  ceiling and the X->next-token ceiling are exactly that finding, used here
  as calibration, not a new result). Our addition is the RMT null (Test 1)
  and the topology-with-shuffle-null instrument (Test 2), both of which turn
  out to complicate rather than confirm a "structured computation" story:
  Test 1 finds real, robust spectral structure that survives the PC-removal
  kill; Test 2 finds the error is topologically *less* complex than a
  covariance-matched Gaussian, which is hard to square with "the error
  encodes extra computation" under this particular instrument.
* Li, Michaud, Baek, Engels, Sun & Tegmark (*Geometry of Concepts*, Entropy
  27(4):344): nearest prior MP use in this literature, on SAE feature
  directions rather than the reconstruction error; they find a power-law
  spectrum rather than counting MP outliers.

## Limitations (state plainly)

* Single SAE release (`gpt2-small-res-jb`, standard ReLU/L1, d_sae=24576) on
  one model (GPT-2 small); no Gemma Scope / Gemma-2-2b run (no HF token --
  this was flagged before any compute was spent, per SPEC).
* Test 1's spectral-outlier count (~25% of all 768 dimensions) is large by
  absolute standards; the SPEC's own >10 bar was set low enough that almost
  any real structure in a 768-dim residual-stream error would clear it --
  the interesting content of Test 1 is in the kill-(ii)/PC-projection
  result (point 3 above), not the bare count.
* Test 2's negative result is reported honestly as a negative, but we do
  not have a mechanistic account of *why* the error cloud is
  topologically simpler than its covariance-matched Gaussian counterpart --
  point 4 above is a hypothesis, not a finding, and no follow-up experiment
  was run to test it (would need, at minimum, relating tc_ratio to the
  SAE's L0/sparsity across a sweep of SAE widths or L1 coefficients, which
  is out of scope for this SPEC).
* Test 3's probes are linear (ridge); "not linearly predictive" (L2 on the
  log-prob-delta target) does not mean "not predictive by any method" -- a
  nonlinear probe was never pre-registered or run.
* One dataset (OpenWebText sample, 512 sequences), one held-out split
  scheme (every 5th sequence); no cross-dataset generalization check.

## Files

`results/e6_test1_rmt.csv` (all clouds x forms x 3 layers, with both
with-replacement block-bootstrap CIs and, for E/X, cross-referenced against
`e6_test1_subsample_ci.csv`'s without-replacement CIs -- the latter is the
one used for the H-E6 criterion per Amendment 2), `e6_test1_shuffle_seeds.csv`
(20 shuffle-seed replicates x 3 layers), `e6_test1_subsample_ci.csv` (20
without-replacement 80%-subsample draws x E/X x 3 layers),
`e6_test2_topology.csv` (E/X/gauss_cov(E) x 3 layers, complete),
`e6_probe.csv` (Test 3a raw-logit-delta + Test 3b next-token, 3 layers),
`e6_probe_logprob.csv` (Test 3a' log-prob-delta, Amendment 3, 3 layers),
`e6_verdict.csv` (per-layer H-E6 table, written for the first time by this
pass), `e6_spectra_L{2,6,10}.npz` (raw eigenvalues per cloud/form, L10
regenerated after the reboot truncated it to 0 bytes),
`fig_spectrum_L{2,6,10}.png`, `fig_tc_ratio.png`, `fig_probe.png` (all
present, 300 dpi, all three layers), `novelty_check.md`, `run.log` /
`extract.log` / `extra.log` / `probe_logprob.log` (original logs) plus this
pass's `extra_e6_test2_L10.py` / `extra_e6_figs_verdict.py` (gap-filling
scripts, not part of the original pre-registered pipeline but reusing its
functions unchanged).
