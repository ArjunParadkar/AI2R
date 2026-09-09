# PAUSE 4 — Blocks M, C, D, B, V — DRAFT 2026-09-09 (Blocks B, C, D, M final; V running)

_Naming: every run in this document is a horse from the Tracen roster (`experiments/HORSES.md`); the handbook codes below are the pre-registration labels and appear here only as such._

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
Re-search 2026-09-09 (NOVELTY_BOARD #8): **Jin et al., arXiv 2608.10566 (Aug 2026) show that iterative erasure counts
are not an affine-invariant concept dimension** — an information-preserving reparameterisation changes the count. B4's
separability rank is therefore reported as a *procedure-relative* quantity (a property of the erasure procedure and the
representation's geometry), not as a concept dimension; the "rank predicts erasure" reading is TAKEN in the negative and
B4 makes no such claim. B2's exact-gauge audit of the direction is the complementary check the paper does not run.

### 6. Claimable / not claimable
**Claimable.** (a) Residual-stream bias directions are renaming-invariant by construction and MLP-hidden ones are
invariant after exact alignment (pipeline check); (b) the WinoBias-style bias direction is unstable across disjoint
sentence halves (cos 0.65–0.79); (c) a Rips-diagram bias statistic is blind to mean shifts and has ≤ 0.25 power on the
real split at n ≤ 400 here; (d) gender PR = 1, COMPAS 3.3, toxicity 1.7 (three points, consistency check only).
**Not claimable.** A certified bias direction; BiasTop as a bias detector; rank-predicts-erasure (3 points, one
inconsistent); anything about other model families (B5 not run); blind-alignment stability (not measured).

## Block M — Memorisation audit (M1 FINAL), benchmark (M2 FINAL), per-class scoping (M3 FINAL) — 2026-09-09

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
| M1 | every shipped feature component invariant under channel permutations; scaling verdict reported | **met**: 216 cached probe clouds (21 CIFAR ResNet-18 models × 2 layers × epochs, 10 GPT-2 fine-tunes × 2 layers × 3 epochs), 30 renamings per kind, 9 features × raw/standardised = 3 888 rows (`results/haru_urara_audit.csv`): permutation CV ≤ 1.6e-14 on **every** row (max 1.6e-14, CIFAR; 7e-14 GPT-2 standardised `med_dist`). Per-unit scaling: raw `tc`, `tc_shuffled` covariant k = 2.000 (R² 1), `med_dist` covariant k = 1.000; raw `tc_ratio` **artifact** (median scale CV 0.095 CIFAR / 0.13 GPT-2), `b1`, `n_bars`, `tc_norm`, `tc0_norm` artifact; `pe1` invariant on 91 % of CIFAR clouds (median scale CV 0.007) but artifact on GPT-2 (0.046); every standardised feature exactly invariant (CV ≤ 1e-13) |
| M3 | AUC per method, no claim | 3 seeds × 10 classes (classes 0–2 with 40 % within-class flips, 12 epochs): ELR early loss **0.95** (best orientation; 0.85 pooled in its natural orientation), topological per-class score **0.90 best orientation but 0.085 in the shipped orientation** (per seed 0.19 / 0.10 / 0.00), class train accuracy 0.50 (uninformative: every class fits its noisy labels) — `results/M3_summary.csv`, `M3_perclass.csv`; existing methods win, as the SPEC expected |

### 3. Kill-test outcomes (M1, M2)
* M1: the shipped `tc_ratio` is *permutation*-invariant to the float floor, so no `memorization_signal` number depends on
  channel labels; under per-unit positive scalings it moves by ~10 % on a ReLU-pooled CIFAR cloud, which is why VALIDATION.md's
  scope note and the refused `_std` calibration exist (PHASE6). For GPT-2 the "scaling" column is a data transform (the
  residual stream has no per-unit scaling symmetry — noted in the CSV), so the 0.13 there is not a gauge artifact.
* Retrained-vs-cached check (SPEC): `topo_score_v1cache` equals `topo_score` on every model (the retrained probe
  activations reproduce the cached features exactly at the shipped calibration).
* The baselines' perfect AUCs are a property of the battery (all noisy models at train_acc_noisy = 1.000), not evidence
  of a better detector; the M3 per-class study and M4 (HPCC) are where the baselines are expected to win on merit.

### 4. What contradicts the plan's framing
* **M3 sign flip.** Per class, corrupted classes get *lower* `tc_ratio` (median 0.61 vs 1.15 for clean classes) and hence
  a *lower* memorisation score (0.21 vs 0.79) — the opposite of the model-level calibration, where memorising models score
  higher. `M3_summary.csv` reports the best-orientation AUC (0.90) because the SPEC asked for a scoping AUC; in the
  orientation the shipped `memorization_signal` would actually output, the per-class AUC is 0.085. So the per-class
  application is not merely weaker than ELR, it points the wrong way: a corrupted class's probe cloud shows the H1
  *collapse* that AUDIT.md filed under "failure-to-learn", not the memoriser signature. This is exactly the scope limit
  VALIDATION.md already states ("probes ≠ ~500 points; layers other than the penultimate") — 150-point per-class clouds are
  outside the calibration. Recorded here so that nobody reads 0.90 as a per-class detector.
* The regression half of M2 cannot be evaluated on this battery (constant target); a graded-memorisation battery (fewer
  epochs, so noisy models memorise 30–80 % of their flips) would be needed — not built in this sprint.
* "Grokking-without-test complexity proxy" as effective rank comes out with AUC 0.00 in the pre-registered orientation
  (higher rank ↔ memorising here); flipping the sign post hoc would make it perfect, so it is reported as 0.00.

### 5. Novelty
TAKEN (NOVELTY_BOARD #9): Topo² (2608.30487) is the direct topological neighbour — memorisation and generalisation as
separable PH-H1 channels on VGG/CIFAR with noisy labels — and must be cited next to MOIT / CrossSplit / ELR; our
`memorization_signal` remains a v1 calibrated signal audited (M1) and benchmarked (M2), not a new detector.

### 6. Claimable / not claimable (M2, M3)
* M3: nothing claimable; `memorization_signal` must not be applied per class (documented in §4; VALIDATION.md row already
  excludes it by probe size).
* Claimable: "on the v1 battery the topological signal is statistically indistinguishable from label-aware baselines
  (AUC 0.967 vs 1.00, p = 0.36) while using only a 500-point probe cloud and no training labels".
* Not claimable: superiority over anything; graded-memorisation tracking; anything cross-family until M4.

## Block C — Audited concept map (Gold Ship: C1–C3) — FINAL 2026-09-09

### 1. Plain English
A 4 096-latent k-sparse dictionary fitted on GPT-2 small's layer-6 MLP-hidden activations (50 k tokens, FVU 0.255) was
re-encoded under 20 exact renamings of each kind (MLP-unit permutations, head permutations, GL(d_head) rotations, LN
scalings) with the dictionary transported consistently. Everything measured on it is invariant to the float floor: the 20
part-of-speech probe R² values (CV ≤ 3e-7), the decoder-direction geometry (CV ≤ 4e-7), and feature identity (99.97 % of
latent activations equal to 1e-3). A *blind* certification — Hungarian matching of latents by activation profile, with no
knowledge of the renaming — recovers the transported feature for 99.4 % of latents under every kind, so the certified
inventory is 4 072 of 4 094 alive features (99.5 %); the 22–24 uncertified ones fail under every kind (cert counts are 0 or
10, nothing in between), i.e. they are near-dead latents with no profile to match. Certified decoder directions are close
to orthogonal (mean |cos| 0.022) and their direction cloud carries 8 104 finite H1 bars (total persistence 2.69); the Mapper
graph of MLP space through the top-2 certified features is one component with cycle rank 6, against an uninformative 2-node
graph through the uncertified lens (there are only 22 of them).

### 2. Pre-registered thresholds and outcomes
| Item | SPEC | Outcome (`results/gold_ship_audit.csv`, `_inventory.csv`, `_geometry.csv`, `_mapper.csv`) |
|---|---|---|
| C1 residual SAE (`gpt2-small-res-jb`) | invariant by construction; float-floor check CV < 1e-4 | **stated, not run**: the runner audits only the MLP-space dictionary (K6 already measured the residual SAE's set statistics invariant to the floor, PAUSE_3 §7) |
| C1 MLP-space dictionary, exact (`align="gauge"`) | probe R² and geometry invariant | invariant: probe R² CV 1e-7 – 3e-7, geometry 0 – 4e-7, all four kinds — **met** |
| C1 blind alignment | Hungarian on decoder cosines | run on **activation profiles** instead (decoder columns are in unit coordinates and permute trivially — the runner's note): blind certified fraction 0.994 for every kind |
| C2 certified inventory | fraction with matched cosine ≥ 0.95 in every renaming | **99.5 % of alive latents (4 072 / 4 094)** over 10 combined renamings; uncertified latents are all-or-nothing (0 of 10) |
| C3 certified geometry | angle matrix, persistence of the direction cloud | mean \|cos\| 0.022; 8 104 H1 bars, total persistence 2.69 (single base run) |
| C3 Mapper certified vs uncertified lens, stability across renamings | β₀/β₁ and their stability | certified: β₀ = 1, β₁ = 6; uncertified: 2 nodes, β₀ = 2, β₁ = 0 (22 features — not a comparable lens); **stability across renamings not computed** (Mapper was run on the base model only) |

### 3. Kill-test outcomes
None pre-registered. The structural anchor holds: with the dictionary transported by the renaming, exact identity is
recovered to 1e-3 on 99.97 % of latent activations.

### 4. What contradicts the plan's framing — stated first
* **C2 is nearly tautological as designed.** Transporting the dictionary with the gauge makes the features the same features
  by construction; the blind certification then measures the *recall of the Hungarian matcher* on activation profiles
  (99.4 %), not whether an independently fitted dictionary finds the same concepts. The interesting instability — different
  seeds resolving the same subspace differently (2606.12138, NOVELTY_BOARD #7) — was not pre-registered and was not run.
  What is claimable from Gold Ship is therefore the negative-space statement: *if* a concept inventory is transported with
  the network's gauge, nothing about it moves; any instability observed across seeds is not a gauge artifact.
* The uncertified lens comparison in C3 is empty (22 near-dead latents), so the "certified vs uncertified Mapper" contrast
  says nothing here; and Mapper stability across renamings was not measured.
* The residual-SAE half of C1 was not executed by this runner (stated as by-construction, backed by K6).

### 5. Novelty
NOVELTY_BOARD #7: seed-consistency of SAE features is TAKEN (2606.12138); the gauge half is what Gold Ship measured, and it
comes out invariant-by-construction. TopoAct is the neighbour for the Mapper view (cite).

### 6. Claimable / not claimable
* Claimable: "an MLP-space concept dictionary transported with the exact gauge group is invariant to the float floor, and a
  blind activation-profile matcher recovers 99.4 % of its latents — so cross-seed SAE feature instability cannot be blamed
  on the network's symmetries".
* Not claimable: a certified inventory in the sense of concepts stable across independently fitted dictionaries; any C3
  stability statement; anything at scale (C4 is the HPCC job).

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
