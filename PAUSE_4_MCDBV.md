# PAUSE 4 — Blocks M, C, D, B, V — DRAFT 2026-09-09 (Blocks B and D final; M/C/V running)

## Block B — Bias (B1–B5)

### 1. Plain English
The gender direction in GPT-2 small's residual stream is real geometry in the weak sense that it is invariant under
every renaming (it lives in the residual stream, which no renaming touches) and in MLP-hidden coordinates it becomes
invariant after exact gauge alignment. But it is not *stable*: directions estimated from disjoint halves of the probe
sentences agree only to cosine 0.65–0.79 (kill threshold 0.9) at every block, and random-split directions reach
|cos| 0.31–0.61 (threshold < 0.5 fails at two of four blocks). Both halves of the pre-registered kill fail, so the
"bias direction" of the v1 setup is largely a property of *which sentences* were probed. The v1 topological bias test
(BiasTop, W2 between H1 diagrams) has no power against the real male/female split at any n up to 400 (power ≤ 0.25),
and the pre-registered synthetic power curve could not have shown anything: it planted a translation, which a Rips
diagram cannot see. With a shape change it can see (stretch along the bias direction) power reaches ~0.6 at n = 200
and does not increase monotonically with the effect size. Separability rank behaves as predicted for gender (PR 1.0)
and COMPAS race (3.26 > 3) but not toxicity (1.74 < 3, three annotator classes).

### 2. Pre-registered thresholds and outcomes
| Item | Threshold | Result |
|---|---|---|
| B1 | AUDIT.md retirement entry | done 2026-09-07 (commit ca1f2cc) |
| B2 exact-aligner (pipeline check, cosine ≥ 0.99 ⇔ aligned CV ≈ 0) | **PASS** after the aligner fix (rerun 2026-09-08): g_mlp aligned CV 2–4e-6; g_heads_raw aligned CV 5–8e-6 under perm / GL-rot / combined at every block. Raw (unaligned) CVs 3–7 = artifact, as expected for unit-/head-indexed coordinates. The normalised g_heads keeps a 0.04–0.06 aligned residual under GL rotations = the global norm is not GL-invariant (recorded, not a bug). Residual-stream g: invariant to 1e-6 by construction |
| B2 blind (Procrustes/Hungarian) cosine | **NOT RUN** — the runner implements the exact aligner only; the SPEC's blind-alignment column is a gap, listed in §6 |
| B2 kill: random-split |cos| < 0.5 AND M/F disjoint-halves cos > 0.9 | **FAIL at all four blocks**: random |cos| 0.31 / 0.57 / 0.61 / 0.47; disjoint-halves 0.79 / 0.79 / 0.71 / 0.65 (blocks 1/2/3/6) |
| B3 FPR at δ = 0 ≤ 0.07 | shift δ = 0 (unique corpus): 0.10 / 0.02 / 0.00 / 0.02 at n = 25/50/100/200 — n = 25 cell exceeds 0.07 with 50 trials (binomial noise: P(≥5 of 50 at 0.05) ≈ 0.10); pooled 0.035 → **calibrated overall, one cell over** |
| B3 power curve (pre-registered translation) | **VOID** — translation-invariant statistic (AMENDMENTS #1); 19 cells run = FPR estimates 0.00–0.10 |
| B3 power curve, post-hoc stretch variant | δ 0.25 / 0.5 / 1.0: n=25 0.12 / 0.28 / 0.14; n=50 0.26 / 0.24 / 0.32; n=100 0.38 / 0.50 / 0.42; n=200 0.60 / 0.54 / 0.42 — **no cell reaches 0.8**, n₈₀ undefined |
| B3 real M/F reference | power 0.04 / 0.06 / 0.02 / 0.00 / 0.25 at n = 25/50/100/200/400 |
| B4 gender PR ≤ 1.5 | **PASS** 1.00 (rank 1 by construction for two groups) |
| B4 COMPAS PR > 3 | **PASS** 3.26 (6 race groups) |
| B4 toxicity PR > 3 | **FAIL** 1.74 (3 annotator classes; quartiles collapse on the discrete score, AMENDMENTS #3) |
| B5 | HPCC, not run |

### 3. Kill-test outcomes
B2 kill fails (both criteria, all blocks). B3's shift cells are the null by construction and are calibrated except one
noisy cell. No B4 kill was pre-registered (3-point consistency check only; 2 of 3 consistent with v1 erasure).

### 4. What contradicts the plan's framing
1. **The bias direction is not reproducible across sentence halves** (cos 0.65–0.79). The plan treated g as a fixed
   object to certify; the object moves by 35–50° between halves of a 100-sentence probe. Any "certified bias
   direction" claim needs a larger, more diverse probe set before invariance is even the question.
2. **BiasTop cannot see a mean shift.** The v1 "topological bias detection" statistic is translation-invariant; whatever
   v1 detected was a change of cloud shape, and against the real M/F split it has power ≤ 0.25 at n = 400 — i.e. the
   v1 detection claim does not replicate on this setup.
3. **The v1 corpus has 60 distinct sentences per group** (AMENDMENTS #2); every earlier n > 60 was a multiset. The
   power/n curves of v1 were curves in duplicate count.
4. The B2 aligner order bug (AMENDMENTS #4) affected only the combined-kind aligned column; the rerun gives exact
   alignment (5e-6) and the kill numbers are bit-identical to run 1 (deterministic seeds).

### 5. Novelty
Searches logged in `B_bias/NOVELTY.md` (2026-09-07): INLP/LEACE erasure literature; gauge-invariance of bias
directions and rank-predicts-erasure adjacent but unclaimed. After these results the claim that would have been
novel (certified invariant bias direction) is not supported; what is reportable is the negative.

### 6. Claimable / not claimable
**Claimable.** (a) Residual-stream bias directions are renaming-invariant by construction and MLP-hidden ones are
invariant after exact alignment (pipeline check); (b) the WinoBias-style bias direction is unstable across disjoint
sentence halves (cos 0.65–0.79); (c) a Rips-diagram bias statistic is blind to mean shifts and has ≤ 0.25 power on the
real split at n ≤ 400 here; (d) gender PR = 1, COMPAS 3.3, toxicity 1.7 (three points, consistency check only).
**Not claimable.** A certified bias direction; BiasTop as a bias detector; rank-predicts-erasure (3 points, one
inconsistent); anything about other model families (B5 not run); blind-alignment stability (not measured).

## Block M — Memorisation audit (M1 running), benchmark (M2 FINAL), per-class scoping (M3 running) — 2026-09-09

### 1. Plain English (M2)
On the retrained v1 CIFAR battery (ResNet-18, 5 000 images, label noise 0–100 % × 3 seeds, plus 3 full-data augmented
generalisers; 21 models, every noisy model reproduces 100 % of its flipped labels), the topological memorisation signal
separates memorising from generalising models with AUC 0.967 — one model misranked — while three of the baselines
(susceptibility to label flips, ELR early-loss statistics, and the oracle generalisation gap) reach AUC 1.00. The DeLong
test cannot distinguish them (p = 0.36, 21 models), so the pre-registered rule fires in its first clause: **the signal is
positioned as complementary, not deprecated.** The battery is too easy for the baselines to say more: with 40 epochs on
5 000 images every noisy model memorises everything, so any statistic that sees the training labels is trivially perfect.

### 2. Pre-registered thresholds and outcomes
| Item | Rule | Outcome |
|---|---|---|
| M2 verdict | if any baseline AUC ≥ topo − 0.02 with DeLong p > 0.05 → "complementary"; if a baseline beats it at p < 0.05 → deprecated-in-favour-of | topo **0.967**; susceptibility 1.00, ELR early-loss 1.00, ELR consistency 1.00, generalisation gap 1.00 (all p = 0.36); weight-norm ratio 0.967 (p = 1.0); effective rank 0.00 raw / 1.00 in its best orientation (sign not pre-registered → not counted) → **complementary; not deprecated** (`results/M2_delong.json`) |
| M2 regression target (`memorized_flipped_frac`) | Spearman per method | **degenerate**: every noisy model reproduces 100 % of its flips, the target is constant on the positive class (empty column in `M2_roc.csv`) |
| M1 | every shipped feature component invariant under channel permutations; scaling verdict reported | _running (`wb-m1`, 216 clouds)_ |
| M3 | AUC per method, no claim | _running_ |

### 3. Kill-test outcomes (M2)
* Retrained-vs-cached check (SPEC): `topo_score_v1cache` equals `topo_score` on every model (the retrained probe
  activations reproduce the cached features exactly at the shipped calibration).
* The baselines' perfect AUCs are a property of the battery (all noisy models at train_acc_noisy = 1.000), not evidence
  of a better detector; the M3 per-class study and M4 (HPCC) are where the baselines are expected to win on merit.

### 4. What contradicts the plan's framing
* The regression half of M2 cannot be evaluated on this battery (constant target); a graded-memorisation battery (fewer
  epochs, so noisy models memorise 30–80 % of their flips) would be needed — not built in this sprint.
* "Grokking-without-test complexity proxy" as effective rank comes out with AUC 0.00 in the pre-registered orientation
  (higher rank ↔ memorising here); flipping the sign post hoc would make it perfect, so it is reported as 0.00.

### 5. Novelty
TAKEN (NOVELTY_BOARD #9): Topo² (2608.30487) is the direct topological neighbour — memorisation and generalisation as
separable PH-H1 channels on VGG/CIFAR with noisy labels — and must be cited next to MOIT / CrossSplit / ELR; our
`memorization_signal` remains a v1 calibrated signal audited (M1) and benchmarked (M2), not a new detector.

### 6. Claimable / not claimable (M2)
* Claimable: "on the v1 battery the topological signal is statistically indistinguishable from label-aware baselines
  (AUC 0.967 vs 1.00, p = 0.36) while using only a 500-point probe cloud and no training labels".
* Not claimable: superiority over anything; graded-memorisation tracking; anything cross-family until M4.

## Block C — _pending (C1–C3 in gpu queue)_
## Block D — Attribution, anchor only (D1) — FINAL 2026-09-09

### 1. Plain English
The auditor reproduces the known Implementation-Invariance result exactly, on every one of the 657 converged
specimens of Block F, on both the truth-table and the 256-point probe cloud: integrated gradients and gradient×input at
the input level are invariant to the float floor (max combined CV 1.8e-15 over 2 628 rows); neuron-level attribution
*values* are artifacts of the unit labelling (median CV 2.2–2.4 under permutation) whose *sorted multiset* is exactly
invariant (1e-15) — the scale-freeness of a_u ∂/∂a_u shows up as scale CV 1e-15 even for the raw values. Per-unit mean
ablation effects are label artifacts too. No claim is made; this is the anchor row of the audit table, and it agrees with
F11's GPT-2 row (per-unit gradient×input artifact 7.6 → invariant 6e-6 after alignment).

### 2. Pre-registered thresholds and outcomes
| Expectation (SPEC §D1) | Outcome (`results/D1_audit.csv`, 657 specimens × 2 clouds, 100 renamings per kind) |
|---|---|
| IG (input level) → invariant, CV < 0.01 | invariant on 657/657 specimens, both clouds; median CV 2e-17 (truth table) / 1.2e-15 (probe) — **met** |
| gradient×input (input level) → invariant | 657/657, median CV 6e-17 / 1.2e-15 — **met** |
| neuron attribution values → artifact under permutation | artifact 657/657; median perm CV 2.39 / 2.24; scale CV 8e-16 / 1e-15 (scale-free, as E2 measured) — **met** |
| sorted multiset of neuron attributions → invariant | invariant 657/657, median CV 8e-16 / 1e-15 — **met** |
| per-unit mean-ablation effect (not pre-classified) | artifact 657/657 under permutation (median CV 1.94 / 2.02); under scaling the median CV is 2e-15 but only 69 % of components are within 0.01 per specimen (IQR 0.63–0.77) — reported, see §4 |
| GPT-2 variant (IG on embeddings vs MLP-neuron attributions, n = 20) | **not run**: F11 already carries the neuron-level row on GPT-2 (grad×input artifact 7.6 → 6e-6 aligned); the embedding-level IG row is missing and stays missing (no claim depends on it) |

### 3. Kill-test outcomes
None pre-registered beyond the expectations themselves (the block is an anchor). The E2 controls of the auditor
(planted invariant passes, planted non-invariant fails, non-symmetry detected) are the relevant kills and are unchanged.

### 4. What contradicts the plan's framing
* The ablation-effect row: mean ablation of unit u replaces a_u by its mean, which commutes with a positive rescaling of
  a_u, so the effect should be exactly scale-invariant — yet ~30 % of components per specimen fail the 0.01 CV bar while the
  median component sits at 1e-15. The plausible cause is the CV statistic itself on components whose base value is ≈ 0
  (dead or inert units: a tiny absolute wobble divided by a tiny mean). **Not verified** (the audit CSV keeps no
  per-component base values); recorded as an open item for the auditor, not for the attribution method. A robust rule
  (skip components with |base| below a floor, or use MAD/|base|+ε) would be an AMENDMENT to S2 and is not made now.
* The specimen count is 657 converged nets, not the 760 (5 seeds × 152 cells) the SPEC quoted: 86 % convergence in F1
  (PAUSE_1) — same subset as F9/F10.

### 5. Novelty
TAKEN (Sundararajan, Taly & Yan 2017; the July-2026 survey's axiom table). NOVELTY_BOARD #10 adds 2606.09899 ("When
attribution patching lies", small-gradient failure modes of attribution patching) as the neighbour for the
ablation-vs-attribution discrepancy at the neuron level — orthogonal to gauge, cited for completeness.

### 6. Claimable / not claimable
* Claimable: "the auditor reproduces the Implementation-Invariance axiom table on 657 nets to the float floor" — as a
  validation of the instrument, nothing more.
* Not claimable: anything about attribution methods themselves; the embedding-level GPT-2 IG row.

## Block V — _pending (V1 sharded, then V2/V3, V5, V6, V4/V8 analysis)_
