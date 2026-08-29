# WHITEBOX audit — what the library is allowed to build on

Scope: every result and script in `/home/god/whitebox_experiments_clone` (GitHub
`ArjunParadkar/AI2R`, commit 95e7a5b) plus the paper (`paper/WhiteBox_ArjunParadkar_v2.pdf`).
Sources for each number are given as clone-relative paths. Nothing in the
clone was modified; this library is a new tree.

## Bin 1 — robust results that replicate (signals are built on these)

| Result | Evidence | Where | Verdict |
|---|---|---|---|
| **Memorization / TC collapse** (P6, H4) | ResNet-18/CIFAR-10: TC_final clean 1.461 vs 100 % noise 0.001 (~1000×), β₁ 295→100, peak SV 13.4 vs 1.3. GPT-2 124M/LM-target noise: TC 607.7 vs 0.211 (~2900×), β₁ 14→1, critical noise ≈ 25 %. Two architectures, two modalities, two label-noise mechanisms. | `results/REPORT.md` Table 3; `results/whitebox_3hour/*/tables/A_memorization.csv`; scripts `exp_memorization.py`, `expA_memorization.py` | **Robust.** Single seed each, but the effect size is 3 orders of magnitude. Phase 2 flagship. Needs: multi-seed, intermediate noise levels, and a calibration battery before it is a *number*. |
| **FPL ↔ per-class accuracy** (P1) | Mean top-3 H1 lifetime after crystallisation vs per-class CIFAR-10 test accuracy: Spearman ρ = −0.84, p = 0.002, 2 seeds × 60 epochs. | `results/whitebox_3hour/*/tables/B_crystallization.csv`; `expB_crystallization.py::top3_fpl` | **Robust for n=10 classes.** Only 10 data points; needs replication on another dataset before exposure as a signal. Instrument (`top_k_lifetime`) ships now; the signal waits for Phase 4 evidence. |
| **P4 failure neighbourhoods** | Layer-3 bootstrap TC of misclassified CIFAR-10 test inputs 0.037 vs correct 0.044, Mann–Whitney p = 8.2e-6 (40 bootstrap draws of 150 pts). β₁ n.s. (p=0.99). | `expC_failure.py`; `results/whitebox_3hour/*/raw/C_*.json` | **Supported, one model.** Effect is small (16 %) but significant; bootstrap draws are not independent samples, so p is optimistic. Phase 4 benchmark item. |
| SV phase-transition peak (H1) | Every layer shows an SV peak in epochs 2–5; cross-seed ρ = 0.93; loss has no matching feature. | `results/REPORT.md` Table 2 | Replicates across 3 seeds. Descriptive, not a decision signal by itself; shipped as the `sv_trajectory` instrument. |

## Bin 2 — shaky or failed (NOT built on; documented so nobody re-derives them)

| Claim | What happened | Where |
|---|---|---|
| **Bias precedence window** (H3: topology leads output bias) | From-scratch TinyGPT: t_bias=2 < t_output=4 at layer 0 only; deeper layers tied. Pretrained GPT-2: reversed (t_output=5 < t_bias=8). Phase-2 scale sweep: null at 3 scales. | `results/REPORT.md` Table 4; `results/whitebox_breakthrough/PROGRESS.md`; `phase2_scaling.py` |
| Bias intervention transfer | 70.8 % gender-stereotype reduction at +1.9 % ppl in the controlled GPT-2 setup, but COMPAS gap 0.220→0.223 (rank-1) / 0.205 (rank-8) and toxicity identity-gap got *worse* (0.158→0.219). Prescriptive, not diagnostic — out of scope for this library. | `phase1_intervention.py`, `phase4_compas.py`, `phase4b_toxicity.py` |
| Fragility under shift (P5) | Residual SV vs accuracy drop under Gaussian noise: ρ=0.05, p=0.88; FPL variant ρ=0.33, p=0.35. Clean negative. | `expC_failure.py` P5 |
| Cross-modal universality (H5/P7) | Matched from-scratch ViT vs TinyGPT SV peaks at t=0.03 vs 0.60; the ViT "peak" is an epoch-1 transient. Refuted. | `expD_crossmodal.py` |
| Crystallisation order ↔ distinctness (H2) | ρ=0.585 p=0.075 (3 seeds), then ρ=0.32 p=0.36 (2 seeds, 60 ep). Cross-seed order ρ=−0.007. Not stable. | `results/REPORT.md` Table 1; expB |

## Bin 3 — instruments (claim-independent; the library core)

All were in `whitebox.py` (125 lines) or inlined in experiment scripts; now
in `whitebox/core` and `whitebox/metrics` with tests.

| Instrument | Definition | Original location | Library |
|---|---|---|---|
| Activation point cloud + fixed reducer | PCA→30 fitted at epoch 0, held fixed | `whitebox.fit_pca`, `exp_vision.extract_activations`, `expA.resid` | `core.reduce`, `activations` |
| Persistence diagrams | VR, maxdim 1, threshold = 95th pct pairwise distance, infinite bars dropped | `whitebox.persistence_diagrams` | `core.persistence.compute_diagrams` (ripser + giotto-ph backends, bit-identical on test clouds) |
| Topological film | diagrams per (layer, epoch) | ad hoc lists in every script | `core.film.Film` |
| SV | W₂ between consecutive diagrams (persim) | `whitebox.structural_velocity` | `metrics.sv` |
| TC | Σ(d−b)² | `whitebox.topological_complexity` | `metrics.tc` (+ Rucco persistent entropy as a scale-free companion) |
| β_k at scale | bars alive at s | `whitebox.betti_at_scale` | `metrics.betti` |
| FPL | (a) top-k lifetime — the one with the P1 result; (b) run-length — paper Def. 5, no result | `expB.top3_fpl`; `whitebox.feature_persistence_lifetimes` | `metrics.fpl` |
| BiasTop + permutation test | W₂ between subgroup diagrams | `whitebox.bias_top`, `permutation_pvalue_bias` | `metrics.biastop` |

### Inconsistencies found and how the library handles them

1. **Two β₁ scale conventions.** Vision runs used the median *pairwise distance*; the GPT-2 run used the median *bar midpoint*. The 295 vs 14 β₁ numbers are therefore not comparable across papers. Library exposes `betti_at_scale(D, s)` with explicit `s` and `median_scale(D)` for the diagram-only convention; the Phase 2 signal will fix one convention and say which.
2. **Two FPL definitions.** Only the top-k one has a result. Both kept, documented.
3. **PCA dimension varied** (30 vision, 20 GPT-2) and **probe size varied** (500 vision, 150 GPT-2). Phase 1 makes both explicit parameters with bootstrap error bars so the choice is visible.
4. **TC is scale-dependent** — the raw 607 vs 0.2 numbers depend on activation norm, PCA dim and threshold. That is exactly why Phase 2 calibrates rather than exposes TC.
5. **Run-length FPL uses strict `>` median**, so a constant film yields no run. Faithful to the original; tested; not recommended.
6. **Original `ripser` threshold at 95 %** means H1 bars dying above the 95th-percentile distance are silently dropped as "infinite". Documented in `compute_diagrams`; Phase 1's exact-vs-approx validation will use `thresh_percentile=100` as ground truth.

## What this means for the build

* Phase 2's `memorization_signal` = TC/β₁ collapse (bin 1 row 1). Calibrated, not raw.
* `top_k_lifetime` and P4 TC are candidates for Phase 4 only after a second dataset.
* No API for bias precedence, bias intervention, fragility, or cross-modal timing.
* BiasTop stays as an instrument (it measures a real distance) with no claim attached.
