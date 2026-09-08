# PAUSE 4 — Blocks M, C, D, B, V — DRAFT 2026-09-08 (Block B final; M/C/D/V running)

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

## Block M — _pending (M1 in cpu queue after D1; M2/M3 in gpu queue after V2/V3)_
## Block C — _pending (C1–C3 in gpu queue)_
## Block D — _pending (D1 in cpu queue after F8–F10)_
## Block V — _pending (V1 sharded, then V2/V3, V5, V6, V4/V8 analysis)_
