# WHITEBOX audit — what the library is allowed to build on

Scope: every result and script in `/home/god/whitebox_experiments_clone` (GitHub
`ArjunParadkar/AI2R`, commit 95e7a5b) plus the paper (`paper/WhiteBox_ArjunParadkar_v2.pdf`).
Sources for each number are given as clone-relative paths. Nothing in the
clone was modified; this library is a new tree.

## Bin 1 — robust results that replicate (signals are built on these)

| Result | Evidence | Where | Verdict |
|---|---|---|---|
| **Memorization / TC collapse** (P6, H4) | ResNet-18/CIFAR-10: TC_final clean 1.461 vs 100 % noise 0.001 (~1000×), β₁ 295→100, peak SV 13.4 vs 1.3. GPT-2 124M/LM-target noise: TC 607.7 vs 0.211 (~2900×), β₁ 14→1, critical noise ≈ 25 %. Two architectures, two modalities, two label-noise mechanisms. | `results/REPORT.md` Table 3; `results/whitebox_3hour/*/tables/A_memorization.csv`; scripts `exp_memorization.py`, `expA_memorization.py` | **Downgraded after Phase 2.** The 1000×/2900× collapses came from 100 %-noise models that had NOT fit the noise (ResNet train acc 0.10; GPT-2 fits 39 % of targets) — a failure-to-learn signature. Models that genuinely memorise show no TC collapse. A memorisation signature exists but its sign flips between CNN and LM (`benchmarks/PHASE2.md`); shipped per family. |
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

* Phase 2's `memorization_signal` was planned on TC/β₁ collapse (bin 1 row 1); the battery showed that collapse is a failure-to-learn artefact. The shipped signal uses scale-free features calibrated per architecture family (PHASE2.md).
* `top_k_lifetime` and P4 TC are candidates for Phase 4 only after a second dataset.
* No API for bias precedence, bias intervention, fragility, or cross-modal timing.
* BiasTop stays as an instrument (it measures a real distance) with no claim attached.

## v2 findings (whitebox-v2 Stage 1, 2026-08-30) — what Phase 6 builds on

Sources: `/home/god/whitebox-v2/E{1,2,4,6}/FINDINGS.md` and `results/`; the
E2 and E4 tables are copied into `benchmarks/results/phase6/`, the E6 tables
into `benchmarks/results/darkmatter/`. E3 was still running when this table
was last updated and is not summarised here.

| Experiment | Pre-registered claim | Result | Library consequence |
|---|---|---|---|
| **E1** gauge orbits vs circuit multiplicity (687+ converged ReLU MLPs on 6 Boolean functions, exact enumeration of minimal circuits and interpretations) | H-E1a: gauge (permutations × ReLU scalings) explains ≥ 50 % of explanation multiplicity; H-E1b: orbits grow slower with width than explanations | **Both fail in every cell**: orbits/explanations ∈ [0.997, 1.000] at every ε; slope difference exactly 0. Even an exactly invariant coarse functional key collapses at most ⅓ of the multiplicity. Méloux-type multiplicity is genuine (distinct neuron subsets with distinct gate assignments), not symmetry seen twice. | **Negative.** No "orbit-collapse" API. Documented so nobody re-derives it. |
| **E2** gauge-invariant observables (same nets, 3 sampling groups × 100 orbit reps, 840 untrained nulls) | H-E2a–f: which quantities are constant on the gauge orbit | All six hypotheses pass as pre-stated. **Invariant** (CV at the 1.3e-15 float floor): logits, input Jacobian, W-product, TC of the per-unit-standardised cloud, sorted grad×act / ablation / gate multisets. **Not invariant**: TC / β / PE of the *raw* activation cloud (scale CV 1.4–3.0 > between-net CV 0.6–1.6), per-layer singular values (CV ≈ 1), raw |activation| and weight norms even after sorting, every raw-index per-neuron reading (top neuron agreement 0.10–0.17, "gate of neuron i" ≈ 1/w). Attribution *values* are scaling-invariant per unit; only their *labels* are gauge. Untrained null: 0/93 verdicts differ. | `whitebox.invariants` ships the invariant column only, each with an orbit test; `check_invariance` is E2's tool. **The v1 TC / β / PE / RTD instruments on raw ReLU activations are coordinate-dependent** (VALIDATION.md). Phase 6 measured what the repair costs `memorization_signal` (PHASE6.md). |
| **E4** legibility phase diagram (87 small models: mlp/tf2/tf4 × modadd/sort/copy × 8 regularisation arms + random-label / untrained / block-masked controls; fixed EAP + greedy-prune discovery pipeline) | H-E4 (i) ρ(Q₁₀, S) > 0; (ii) a knee; H-E4b treewidth/N predicts S and cost; H-E4c Q survives control for sparsity and N | **All fail.** ρ(Q₁₀, S) = −0.37 [−0.55, −0.17] — wrong sign and an architecture offset (within-arch CIs all cross 0; partial ρ | arch = +0.03 [−0.26, 0.33]); no knee (ΔBIC −2.6); ρ(tw/N, S) = −0.11 [−0.35, 0.13]; cost/N ∈ [1.0, 2.1] by construction of a greedy pass. Attribution graphs are *less* modular than their shuffle null in 90 % of models. Golechha et al.'s "clusterability loss ⇒ smaller circuits" replicates, but not via attribution-graph Q in transformers. Untrained models "succeed" trivially without an accuracy guard. | `whitebox.legibility` ships the **profile** (Q sweep + shuffle null, treewidth bound, sparsity, guarded discovery outcome) as an instrument; **`legibility_score` raises `NotCalibratedError`** with these numbers. No FPT conjecture is stated. |
| **E6** dark-matter structure of SAE reconstruction error (GPT-2 small, `gpt2-small-res-jb` residual-stream SAEs, 3 layers, 65k OpenWebText tokens/layer; MP/TW spectral null + v1-topology-with-shuffle-null) | H-E6 (per layer): n_outliers > 10 (CI lower > 10) **AND** tc_ratio > 1.2 (CI lower > 1.0) | **Fails at every layer, and asymmetrically.** Test 1 (spectral) **passes** cleanly and by a wide margin (n_outliers 179–198 of 768; CI lower bound two orders of magnitude above the >10 bar; kill (i) matched-diagonal-Gaussian gives 0, kill (iii) feature-shuffle gives 0 across 20 seeds — the test is not void; kill (ii)'s PC-projection refinement shows the outliers are not simply the raw activations' top-50 directions leaking through). Test 2 (topology) **fails, and not narrowly**: tc_ratio(E) = 0.11–0.46, **below 1.0** at every layer — the error cloud is topologically *less* complex than its own feature-shuffled null, and well below the covariance-matched Gaussian control (0.87–1.01, correctly calibrating to ~1). Combined reading: real, robust spectral structure that is not accompanied by extra topological complexity by this instrument — not "spectrally structured, topologically Gaussian-like" (E's ratio is *below* the Gaussian-like control, the opposite direction), and not "structured computation". Test 3 (linear predictive structure, not part of H-E6) is real but small and grows with depth (held-out R² on log-prob-delta: 0.022/0.066/0.283 at L2/6/10; shuffle ≈ 0). | `whitebox.darkmatter` ships **both instruments** (`residual_structure`) with a documented `verdict` field; every layer in the one existing battery reads `"spectrally_structured"`, never `"topologically_structured"` or `"both"` — the module docstring states this plainly so nobody mistakes "the instrument exists" for "dark matter is structured". See `benchmarks/DARKMATTER.md`. |

Cross-cutting: every Stage-1 positive was an instrument-level fact (a quantity
is invariant; a measure can be computed); every *predictive* claim tested
(gauge explains multiplicity, structure predicts tractability, spectral
structure comes with matching topological structure) failed its
pre-registered criterion. v2 therefore adds instruments and (so far) zero new
signals; the one candidate signal question it raised — is the calibrated
memorisation signal gauge-invariant, and does making it so cost power — is
answered in `benchmarks/PHASE6.md`.
