# PAUSE 3 — Block K, dark matter (K1–K9) — 2026-09-08

Written as a report, not a stop (standing instruction). Runs: `wb-gpu-queue` K1 (TL venv) + `wb-k2k6` (rerun after
the `mp_edge_hi` attribute bug, commit 71d223c) on 2026-09-07; K6 (`run_K6.py`, written 2026-09-08 — the original
runner never implemented it) as `wb-k6`. Gemma-2-2B / Gemma Scope: **NOT RUN** (gated model, no HF token on this
machine) — recorded as not run, not as a null. K7/K8 are HPCC and not run.

## 1. Plain English
The SAE residual of GPT-2 small (the part of the residual stream the public `gpt2-small-res-jb` dictionary does not
reconstruct: 8–15 % of the variance at layers 3/6/9) is not noise. Its covariance has ~180 eigenvalues above the
Marchenko–Pastur/Tracy–Widom edge where noise would have ≤ 1, and the pre-registered kills (matched-marginal
Gaussian, feature-shuffled residual) both give 0 outliers. But the covariance-matched Gaussian control reproduces the
outlier count essentially exactly (186/185/179 vs 184/182/173), so what the spectral test certifies is second-order
structure and nothing more. The topological test does not show what the plan expected: the residual cloud has *less*
total persistence than its own feature-shuffle (ratio 0.24–0.32 with CIs far from 1) while the covariance-matched
Gaussian sits at 1.0, i.e. the residual deviates from the null in the direction opposite to the pre-registered one —
the same result E6 found in August. The residual carries predictive content about the model's output at layers 6 and 9
(ridge from e to Δlogits beats shuffled e by 0.09 and 0.33 R²; layer 3 by 0.03, below threshold), and a
random-feature quadratic map adds nothing over linear at any layer (ΔR² negative). K6 (this run) certifies which of
these statistics are gauge-invariant: see §2/§7.

## 2. Pre-registered thresholds and outcomes
| Item | Threshold | Result |
|---|---|---|
| K1 | FVU within 10 % of published, or "no published value" | FVU 0.076 / 0.122 / 0.153 (L3/L6/L9; variance explained 0.924 / 0.878 / 0.847). SAELens card value not machine-readable → **"no published value"** (as the SPEC prescribes; not reported as passed) |
| K2 | > 10 outliers per layer; kills ≤ 1 | **PASS** — 184 / 182 / 173 outliers (correlation form; covariance form 171 at every layer), structured fraction 0.50 / 0.41 / 0.40, sequence-block bootstrap CI (185–190) / (186–189) / (179–181); kills: matched-marginal Gaussian 0 / 0 / 0, feature-shuffled 0 / 0 / 0. **Covariance-matched Gaussian control: 186 / 185 / 179** (see §4) |
| K3 | TP(e)/TP(shuffle) > 1.2, CI excludes 1 | **FAIL** (wrong direction) — 0.32 [0.30, 0.39] / 0.24 [0.22, 0.28] / 0.32 [0.28, 0.35]; β1 177 / 83 / 159 vs Gaussian control β1 840 / 939 / 1015 and control ratio 0.96 [0.92, 1.09] / 1.02 [0.93, 1.11] / 1.05 [0.90, 1.07] |
| K4 | R²(e) − R²(shuffled e) > 0.05 | L3 **FAIL** 0.032 (R² 0.005); L6 **PASS** 0.087 (R² 0.061); L9 **PASS** 0.332 (R² 0.305); λ = 100 chosen on the validation split each time |
| K5 | ΔR²(quadratic − linear) > 0.02 | **FAIL** all layers: −0.111 / −0.093 / −0.060 |
| K6 | invariance class reported; residual-stream anchor exactly invariant | see §7 (filled from `results/K6_certification.csv`) |
| K7 | width scaling | HPCC, not run; **TAKEN** (Engels et al. 2410.14670) — would run as replication only |
| K8 | frontier residue | HPCC, not run |
| K9 | noise → structured fraction < 0.02, ≤ 1 outlier | **PASS** — `tests/test_darkmatter.py::test_k9_synthetic_noise_structured_fraction_below_002` |

## 3. Kill-test outcomes
* K2 kill (i) matched-marginal Gaussian: 0 outliers at every layer (≤ 1 required) — pass.
* K2 kill (ii) variance-matched feature-shuffled residual: 0 outliers at every layer — pass.
* K3 calibration: covariance-matched Gaussian ratio ≈ 1 with CI covering 1 at every layer — the statistic is calibrated;
  the residual's departure (ratio ≈ 0.3) is real, but downward.
* K9 unit test passes.
* K6 kills are the anchors: the residual-stream statistics must not move at all under any renaming (§7).

## 4. What contradicts the plan's framing (stated first, not buried)
1. **The spectral "dark matter" is second-order.** The covariance-matched Gaussian control reproduces the outlier count
   to within 1–3 % at every layer (186 vs 184, 185 vs 182, 179 vs 173). The K2 pass therefore certifies "e has a
   non-white covariance", which any linear residual of a non-linear encoder will have; it does not certify structure
   beyond covariance. The handbook's "dark matter" language must be read as "structured covariance" in the write-up.
2. **The topological test fails in the opposite direction, again.** E6 (2026-08-30) found tc_ratio 0.11–0.46 below the
   shuffle null; Block K with the handbook's protocol finds 0.24–0.32. The pre-registered direction (> 1.2) was wrong
   twice. A residual cloud whose feature-shuffle has *more* H1 persistence is what strongly correlated, heavy-tailed
   coordinates produce; the covariance-matched Gaussian at 1.0 shows covariance alone does not do it — the effect is
   non-Gaussian (tails/clustering along a few directions), but it is not the "extra loops" the v1 pipeline was built
   to find. Verdict for the library stays "spectrally structured only", as in `residual_structure`.
3. **Nonlinear residue is absent at this budget.** K5 is negative at every layer; note the design weakness that λ was
   chosen on the linear validation split and reused for the 2 768-dimensional quadratic features (ridge on 2 000 random
   quadratic features with the linear λ can only hurt). This is an under-powered test of nonlinearity, not evidence
   against Engels et al.'s nonlinear error component, which is defined on x, not on Δlogits.
4. **K6 was missing from the runner.** The 2026-09-07 runner implemented K2–K5 only; K6 was written and run on
   2026-09-08 (`run_K6.py`). No K result was changed by this; it is recorded so the omission is visible.
5. Gemma-2-2B was never run (gated). Every K number is GPT-2 small, one SAE release, one 50 k-token corpus.

## 5. Novelty searches (K_darkmatter/NOVELTY.md, 2026-09-07)
* "Marchenko-Pastur random matrix null sparse autoencoder reconstruction error residual structure" — no hit applies an
  MP/TW null to SAE residuals (nearest: MP-guided pruning 2606.02608, RMT-SVD pruning 2303.08986). **OPEN as
  application** — but see §4.1: what is open is a weak result.
* "Engels dark matter sparse autoencoders scaling width nonlinear error" — Engels, Smith & Tegmark 2410.14670
  (nonlinear error constant with width; ~50 % of error linearly predictable from x). **K7 TAKEN.** K4's Δlogit target is
  distinct from their x-target and is positioned as such; must cite.

## 6. Claimable / not claimable
**Claimable.** (a) On GPT-2 small with `gpt2-small-res-jb`, the SAE residual at layers 3/6/9 has a covariance spectrum
with ~180 Tracy–Widom outliers (structured fraction 0.40–0.50), reproduced by a Gaussian with the same covariance and
destroyed by feature shuffling — "spectrally structured, second-order". (b) The residual predicts the model's own
Δlogits linearly at layers 6 and 9 (held-out R² 0.06 and 0.31 over shuffled −0.03), not at layer 3. (c) The v1
topological statistic is calibrated (Gaussian control at 1) and the residual departs from its null *downward*; the
pre-registered upward threshold failed. (d) `residual_structure` ships with the "neither / spectrally / topologically /
both" verdict and never says "dark matter"; K9 passes.
**Not claimable.** "Structured dark matter beyond covariance"; any width-scaling statement (K7 taken, not run); any
nonlinear-residue statement (K5 negative and under-powered); anything about Gemma or frontier models; the phrase
"topologically structured residual" in any direction the SPEC did not pre-register.

## 7. K6 — certification (filled from the run)
_pending — `wb-k6` in progress; table inserted when `results/K6_certification.csv` is complete._
