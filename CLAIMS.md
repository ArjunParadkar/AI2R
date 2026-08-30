# CLAIMS — the integrity backbone of this library

Every element of the public API (`whiteboxtopo.__all__` and the documented
submodule imports in `README.md`) has one entry below: the claim it
supports (or "instrument only — no claim" / "REFUSED — no claim"), the
evidence for it, the null(s) it was run against, its known failure modes /
do-not-use conditions, and — for refused APIs — why they refuse, naming the
`whitebox-v2` experiment (E1–E6) or benchmark Phase that produced the
refusal.

**Sourcing rule:** every number here is copied from `VALIDATION.md`,
`AUDIT.md`, or `benchmarks/*.md`. Nothing is asserted here that isn't
already asserted, with evidence, in one of those three files. Candidate
signals that never made it into the public API at all (P1, P4, P5, P6-as-
originally-claimed, H2/H3/H5, E1's orbit-collapse hypothesis) are *not*
re-listed here — they are catalogued in `VALIDATION.md`'s "Candidates that
did NOT pass" table and `AUDIT.md`'s Bin 2, and this file only maps
things that actually ship.

Legend: **SIGNAL** = calibrated, has a ground-truth battery and a
threshold/verdict a user can act on. **INSTRUMENT** = a measurement with
characterized reliability, no claim attached. **REFUSED** = the API exists
(a name a user can import/call) but deliberately does not return the answer
a user might expect; the refusal itself is the shipped behavior.

---

## Core measurement pipeline — `whiteboxtopo.core`

### `fit_reducer` / `Reducer`, `compute_diagrams` / `Diagrams` / `available_backends`, `Film` / `FilmRecorder` / `film_from_activations` / `topological_film`, `bootstrap_metric` / `BootEstimate`

- **Claim:** none — this is plumbing (fixed-PCA reduction, Vietoris–Rips
  persistence, per-(layer,epoch) diagram bookkeeping, subsampling CIs), not
  a claim about any model.
- **Evidence:** `compute_diagrams` is exact VR persistence (maxdim 1,
  threshold = 95th-percentile pairwise distance, infinite bars dropped);
  ripser and giotto-ph backends are bit-identical on 24 test clouds; the
  95th-percentile threshold "loses nothing" on those same clouds. Timing:
  0.11 s at n=500, 39 s at n=8000 (≈ n^2.3); a 5-layer frame with 20
  bootstrap subsamples ≈ 7 s on a laptop GPU, verified to GPT-2 XL.
  (`VALIDATION.md` Instruments row 1; `benchmarks/PHASE1.md`.)
- **Nulls:** none applicable (this layer computes a deterministic geometric
  object, not a statistic being tested against a null).
- **Failure modes / do-not-use:** TC and β₁ are **size-dependent** (TC ≈
  linear in n) — only compare films/diagrams computed at a fixed probe
  size. The `landmark` backend (approximate PH via maxmin landmarks)
  preserves TC *ordering* (ρ 0.98) but not magnitude (7–19% error) and
  destroys β₁; no shipped signal uses it. Two incompatible β₁ scale
  conventions exist across the original paper's runs (median pairwise
  distance vs median bar midpoint) — `betti_at_scale(D, s)` takes `s`
  explicitly and `median_scale(D)` gives the diagram-only convention; state
  which you used (`AUDIT.md` inconsistency #1).
- **Detail:** `benchmarks/PHASE1.md`.

---

## v1 metrics — `whiteboxtopo.metrics` (diagram functionals)

### `structural_velocity`, `topological_complexity`, `persistent_entropy`, `betti_at_scale`, `betti_curve`, `median_scale`, `top_k_lifetime`, `feature_persistence_runs`

- **Claim:** instrument only — no claim. These are the raw functionals the
  v1/v2 signals are built from.
- **Evidence:** tested against known diagrams (SV(D,D)=0; W₂(single bar, ∅)
  = lifetime/√2). Two FPL definitions exist (top-k lifetime — has a
  positive result below; run-length — paper Def. 5, no result, uses strict
  `>` median so a constant film yields no run). `top_k_lifetime` is a
  **Phase-4-only candidate**: FPL↔per-class-accuracy replicated at ρ=−0.84
  (p=0.002, n=10 classes, 2 seeds) in the original paper but is unreplicated
  on a second dataset — ships as an instrument, not a signal.
  (`VALIDATION.md` Instruments row 2; `AUDIT.md` Bin 1 row 2, Bin 3.)
- **Nulls:** none at the instrument level; nulls are applied by the signals
  built on top (below).
- **Failure modes / do-not-use:** raw TC/β₁/PE values are **not comparable
  across activation scale, PCA dimension, or probe size** — this is exactly
  why Phase 2 calibrates rather than exposing raw TC as a signal (`AUDIT.md`
  inconsistency #4). On a ReLU-MLP raw activation cloud specifically, TC/β/PE
  are additionally **not gauge-invariant** — see the v2 scope-note entry
  below (E2).
- **Detail:** `benchmarks/PHASE1.md`, `AUDIT.md` Bin 3.

### `bias_top`, `bias_top_permutation_test`

- **Claim:** instrument only — measures a real W₂ distance between two
  subgroups' diagrams; no bias-detection claim ships (no API translates a
  significant result into "this model is biased").
- **Evidence:** false-positive rate at α=0.05 is **0/30 and 0/30**
  (conservative, not anti-conservative).
- **Nulls:** the permutation null itself, on 30+30 synthetic trials.
- **Failure modes / do-not-use:** power at n=100/group is 0.4–0.6 only when
  subgroup B is an entirely different class; 0.0–0.3 at realistic 10–50%
  mixing — **underpowered for realistic bias**. Use n≥300/group and treat
  p>0.05 as "not detected," never as "no bias."
- **Detail:** `VALIDATION.md` Instruments row 3.

---

## v1 signals — `whiteboxtopo.signals`

### `memorization_signal(..., family="cnn_classifier")`, `memorization_features`

- **Claim (SIGNAL):** a calibrated ranking on [0,1] (memorising ↔
  generalising) for ResNet-style CNN classifiers, penultimate pooled
  features, ~500-point training-set probe.
- **Evidence:** 21 ResNet-18/CIFAR-10 models spanning label noise 0–100%,
  every noisy model verified to reproduce 100% of its flipped labels;
  leave-one-seed-out AUC **0.98** (in-sample 1.00); monotone in noise;
  `tc_ratio` is size-stable (subsample/full ratio 0.96).
- **Nulls:** the battery *is* the null-and-positive design (noise 0% through
  100%); LOSO cross-validation is the held-out generalization check.
- **Failure modes / do-not-use:** **sign flips** across architecture
  families (applied to GPT-2, AUC drops to 0.67 — do not cross-apply);
  checkpoints before the loss plateaus (classes overlap epochs 10–20);
  probes other than ~500 points; layers other than the penultimate pooled
  features; the score is a **ranking**, not a calibrated probability —
  thresholds are battery-relative. CNN intervals are wide (±0.15–0.25):
  expect `verdict="uncertain"` unless the point estimate is clearly on one
  side.
- **Detail:** `benchmarks/PHASE2.md`, `VALIDATION.md` Signals row 1,
  `AUDIT.md` Bin 1 row 1 (the collapse this signal replaces was a
  failure-to-learn artifact, not memorization — see there).

### `memorization_signal(..., family="lm_transformer")`

- **Claim (SIGNAL):** the same ranking, calibrated separately for GPT-2 124M
  fine-tunes on LM-target noise.
- **Evidence:** 10 GPT-2 124M fine-tunes, LM-target noise 0–100% (+
  pretrained reference); noisy models reproduce 33–39% of corrupted
  targets; **all** fine-tunes overfit to some degree; LOSO AUC **1.00** on 8
  labelled models.
- **Nulls:** same battery design as the CNN family.
- **Failure modes / do-not-use:** tiny battery (2 seeds); the
  "generalising" reference class is itself overfit, so this calibration
  separates *degrees* of overfitting, not memorising-vs-clean; applying it
  to a CNN gives AUC **0.00** (perfectly anti-correlated — do not
  cross-apply, this is a hard opposite-sign failure, not a weak one).
- **Detail:** `benchmarks/PHASE2.md`, `VALIDATION.md` Signals row 2.

### `memorization_signal(..., standardize=True)` — REFUSED for gauge-invariant `_std` calibration

- **Claim:** REFUSED. `Calibration.load(family, standardize=True)` raises;
  there is no shipped `_std` calibration for either family, despite the
  feature itself (per-unit standardised, gauge-invariant) existing and
  being exactly invariant under ReLU-MLP unit scaling by unit test.
- **Why it refuses (E2 + Phase 6):** the pre-registered rule was
  "standardisation must keep the shipped-feature AUC" — it **failed in
  both families**. CNN `pool`: standardisation is nearly free (`tc_ratio`
  0.981→0.963 LOSO AUC, paired-diff CI [0.00, 0.11]; `pe1`/`tc_norm`/
  `tc0_norm`/combos hold at 1.0) — close, but the rule as pre-registered
  still requires shipping to fail if any feature drops, and GPT-2 `L12`
  fails hard: `pe1` AUC 1.00→0.833 [0.42, 1.0], graded-noise ρ 0.96→0.37.
  For GPT-2 the per-unit z-score is a **data transform, not a gauge
  repair** — a residual stream has no per-unit scaling symmetry, so the
  raw default was never a gauge artifact to begin with. n=15/8 held-out
  pairs — CIs are wide.
- **Detail:** `VALIDATION.md` v2-modules row "`memorization_signal(...,
  standardize=True)`"; `benchmarks/PHASE6.md`.

### `representation_correspondence`, `correspondence`, `rtd`

- **Claim:** **instrument only**, despite being named like a signal — no
  calibrated claim ships. `score = 1 − RTD/RTD_null` (RTD, Barannikov et
  al. ICML 2022, reimplemented exactly).
- **Evidence:** on synthetic manifolds of known topology (circle/torus/
  sphere) the `iso` control scores 1.000 on every manifold — the
  reimplementation is exact; untrained nets score 0.80–0.95.
- **Nulls:** the RTD null itself (random re-pairing of activations) is
  built into the score; the pre-registered hypothesis (Sub-H4: layers that
  generalize better preserve more input topology) was tested on 33 real
  models with known test accuracy.
- **Failure modes / do-not-use:** the pre-registered hypothesis was
  **refuted** — generalisers score *lowest* on images (CIFAR 0.05, AUC
  gen-vs-mem 0.62 — near chance; MNIST memorisers 0.25 vs generalisers
  0.125, the *opposite* direction); no relation to test accuracy (ρ=−0.20 /
  −0.41, both n.s.). On real images the input cloud is pixel space and
  **good classifiers score near 0 on purpose** (they discard input
  topology) — do not read a low score as "bad representation."
- **Detail:** `benchmarks/PHASE3.md`, `VALIDATION.md` Signals row 3.

---

## v2 — `whiteboxtopo.invariants` (E2)

### `check_invariance`, `InvarianceReport`, `audit_metrics`

- **Claim:** INSTRUMENT — an audit tool, not itself a claim about any
  metric. Empirically measures whether a user-supplied metric is constant
  on a ReLU-MLP's gauge orbit (hidden-unit permutations × per-unit positive
  scalings), reporting a verdict (`exact`/`approximate`/`not_invariant`)
  and `broken_by`.
- **Evidence (E2):** float floor 1.3e-15 (max 2.2e-15) measured on 687
  trained + 840 untrained nets; orbit representatives are far from the
  base point (median relative parameter distance 1.35 perm / 2.56 scale /
  3.15 combined — the audit isn't trivially sampling near-identity
  transforms); untrained null: 0/93 verdicts differ from trained. Unit
  tests: a planted invariant metric passes (CV<1e-10), a planted
  non-invariant metric (`W0[0,0]`) fails in all three groups, an
  intentionally-broken-symmetry construction (a layer-0 permutation not
  propagated to layer 1) is correctly flagged not-invariant; numpy MLP and
  torch `nn.Sequential` agree.
- **Nulls:** the untrained-network null (0/93 verdict changes) and the
  hand-built planted-invariant / planted-non-invariant unit tests above
  serve as the tool's own self-check.
- **Failure modes / do-not-use:** **ReLU MLPs only** — transformers
  (LayerNorm, head permutations, residual rotations), gated or non-ReLU
  units have a different, larger symmetry group; a quantity found `exact`
  here can only move to non-invariant when *more* symmetries are added, so
  an `exact` verdict is conditional on `G` and not a universal statement.
  E1's hidden flat directions (found in 92% of nets) are **not sampled** by
  this audit. float32 torch models floor at ≈1e-7 (reads `approximate`;
  call `.double()`).
- **Detail:** `VALIDATION.md` v2-modules row 1, `benchmarks/PHASE6.md`,
  `tests/test_invariants.py`.

### `logits`, `input_jacobian`, `weight_product`, `standardized_tc`, `sorted_gradxact`, `sorted_ablation`, `gate_multiset` (the E2 invariant column)

- **Claim:** INSTRUMENT set — these specific quantities *are* exactly
  gauge-invariant for a ReLU MLP (this is the positive claim: "these are
  well-defined function-level statements, unlike their raw-neuron
  counterparts").
- **Evidence (E2):** median orbit CV 1.2–1.3e-15 across permutation, scale
  and combined groups (gate multiset CV exactly 0); between-network CV
  ≥0.39 (they're not trivially constant across all networks, only across
  the orbit of one); each has its own orbit unit test on hand-built, random
  and trained nets (CV ≤1e-10).
- **Nulls:** the orbit-sampling test itself is the null design (CV under
  the known symmetry group vs CV between independently trained networks).
- **Failure modes / do-not-use:** `weight_product` is the **linear part
  only** (equals the end-to-end map only inside the all-active ReLU
  region) — not the full function outside that region. `standardized_tc`
  is a **different metric** from raw TC (per-unit scale removed with the
  gauge) — do not compare its values to raw-TC benchmarks. The sorted
  multisets (`sorted_gradxact`, `sorted_ablation`, `gate_multiset`) say
  nothing about **which** unit produced which value — only the multiset is
  invariant, individual per-index attributions are not.
- **Detail:** `VALIDATION.md` v2-modules row 2, `benchmarks/PHASE6.md`.

### Raw-cloud `topological_complexity`, `betti_at_scale`, `persistent_entropy`, `rtd` applied to ReLU hidden activations — scope caveat (not a new function; a warning on the v1 instruments above)

- **Claim:** none — this entry documents where the v1 instruments (above)
  are **not** safe to use.
- **Evidence (E2):** exactly permutation-invariant, but **not
  scaling-invariant**: per-unit-scale CV of TC is 1.4–3.0 (larger than the
  between-network CV of 0.6–1.6, i.e. the scale artifact swamps the signal
  you'd want to measure); β₁ CV 0.5–1.1; PE(H1) CV 0.11. Under *uniform*
  scaling specifically, TC is covariant of degree 2 (fitted exponent
  2.000±1e-9) and β/PE are invariant — so uniform rescaling is fine, but
  the ReLU-MLP gauge group is *per-unit*, not uniform, scaling.
- **Nulls:** same E2 orbit-sampling design as above.
- **Failure modes / do-not-use:** a raw-coordinate TC on a ReLU layer, or a
  ratio of two such values across checkpoints, **is a statement about one
  weight representative, not about the function** — use `standardized_tc`
  or `memorization_features(..., standardize=True)` (see its own refusal
  entry above) when a function-level claim is intended, and note that the
  `standardize=True` calibration itself is refused for both memorization
  families.
- **Detail:** `VALIDATION.md` v2-modules row 3, `benchmarks/PHASE6.md`.

---

## v2 — `whiteboxtopo.legibility` (E4)

### `legibility_profile`

- **Claim:** INSTRUMENT — a structural profile (Leiden modularity Q with a
  density sweep and shuffled-weight null / `Q10_excess`, min-fill treewidth
  upper bound, effective sparsity, and the guarded outcome of a fixed
  attribution-patching + greedy-prune discovery pipeline). No prediction
  attached.
- **Evidence (E4, 87 models):** a planted-modular graph recovers its 4
  planted communities at Q>0.6 vs random Q<0.4; treewidth bound is exact on
  trees (1), cycles (2), cliques (n−1), 4–5 on a 4×4 grid (true value 4 —
  correctly an upper bound); discovery recovers a planted 12-node circuit
  at F≥0.9 with nothing outside it; untrained models are correctly guarded
  (S=0 when accuracy<0.5, E4 kill 3).
- **Nulls:** the shuffled-weight null for Q; the untrained-model guard as a
  negative control; planted-graph / planted-circuit recovery as positive
  controls.
- **Failure modes / do-not-use:** small, node-instrumented models only (E4
  ensemble capped at N≤300 nodes); `cost/N` is 1–2 **by construction** of a
  greedy pass and carries no complexity statement; `treewidth_bound` is an
  upper bound, not exact; a faithfulness ratio is meaningless when the
  clean–corrupt gap ≈0.
- **Detail:** `VALIDATION.md` v2-modules row 4, `benchmarks/PHASE6.md`,
  `tests/test_legibility.py`.

### `legibility_score` — REFUSED

- **Claim:** REFUSED. Calling it raises `NotCalibratedError`; there is no
  number this function will ever return in the current library.
- **Why it refuses (E4, n=87):** the pre-registered calibration attempt
  failed on every axis tested. ρ(Q₁₀, S) = **−0.37** [−0.55, −0.17] pooled
  — wrong sign, and it's an **architecture confound**: within-architecture
  CIs all cross 0 (mlp +0.32 [−0.12, 0.68], tf2 −0.38 [−0.70, 0.04], tf4
  +0.21 [−0.20, 0.55]); partial ρ | architecture = +0.03 [−0.26, 0.33] —
  consistent with zero once architecture is controlled for. ρ(treewidth/N,
  S) = −0.11 [−0.35, 0.13]. No modularity "knee": ΔBIC = −2.6, bootstrap
  mass 44%, permutation p=0.06. Partial ρ(Q₁₀, S | sparsity, N) = −0.22
  [−0.43, 0.09]. **Any number this function returned would predominantly
  predict the architecture label, not tractability** — that's why it stays
  closed rather than shipping with a caveat.
- **Detail:** `VALIDATION.md` v2-modules row 5, `benchmarks/PHASE6.md`,
  `whiteboxtopo/legibility/profile.py::legibility_score` (the raised
  message embeds these numbers directly).

---

## v2 — `whiteboxtopo.darkmatter` (E6)

### `residual_structure`, `DarkMatterReport`

- **Claim:** INSTRUMENT — two independent readings (Test 1: MP/Tracy–Widom
  spectral-outlier count with a matched-diagonal-Gaussian kill and a
  feature-shuffle validity check; Test 2: v1 TC(H₁)-with-shuffle-null ratio
  with a bootstrap CI and a covariance-matched Gaussian control), combined
  into a documented `verdict` field. **The `verdict` field is not itself a
  claim that dark matter is structured** — it names which instrument(s)
  cleared their own pre-registered bar.
- **Evidence (E6, GPT-2 small, `gpt2-small-res-jb`, 3 layers, 65k
  OpenWebText tokens/layer):** Test 1 **passes at every layer** (n_outliers
  179–198 of 768 dims; CI lower bound two orders of magnitude above the
  >10 bar; kill (i) matched-diagonal-Gaussian gives 0 outliers; kill (iii)
  feature-shuffle gives 0 across 20 seeds — the test is not void; kill
  (ii)'s top-50-PC-of-x projection refinement shows the outliers are not
  simply raw activations leaking through). Test 2 **fails at every layer,
  and not narrowly**: tc_ratio(E) = 0.11–0.46, **below 1.0** — the error
  cloud is topologically *less* complex than its own feature-shuffled
  null, and below a covariance-matched Gaussian control that correctly
  calibrates to ≈1.0.
- **Nulls:** MP/Tracy–Widom bulk-edge null (Test 1); feature-shuffle null
  and a matched-diagonal-Gaussian kill (Test 1 validity); TC-with-shuffle
  bootstrap null and a covariance-matched Gaussian control (Test 2). Unit
  tests: pure-Gaussian control gives verdict `neither`; a planted rank-k
  spike above the BBP threshold is detected exactly with the shuffle null
  intact; a noisy-circle-in-d-dims control clears both the 1.2/CI bar and
  the Gaussian-control gate (verdict `topologically_structured`) —
  confirming the topology test *can* fire positive, it just didn't on the
  real E6 data; a heavy-tailed (Cauchy) control demonstrates
  `spectral_test_valid=False` (the kill correctly catches its own Gaussian
  assumption breaking).
- **REFUSED in practice — no `verdict='structured'` claim on the shipped
  battery:** every layer in the one existing battery reads
  **`"spectrally_structured"`**; **none** reads `"topologically_structured"`
  or `"both"` (those verdicts require clearing the covariance-matched-
  Gaussian control on top of the raw 1.2/CI bar — an E6 interpretive
  commitment made mechanical in the code, never actually triggered on this
  battery). Read the combined result as "real, robust spectral structure
  not accompanied by extra topological complexity" — explicitly **not**
  "structured computation" and **not** "dark matter is structured."
- **Failure modes / do-not-use:** `spectrally_structured` requires
  `spectral_test_valid`; if the feature-shuffle kill fails on your data
  (non-Gaussian marginals, small n relative to p), Test 1 is void and the
  field is forced `False`. One battery so far: GPT-2 small, one SAE
  release, 3 layers; Gemma Scope untested (no HF token when E6 ran). Test 3
  (linear predictive structure — real but small, held-out R² 0.022/0.066/
  0.283 at L2/6/10, growing with depth, shuffle≈0) is **not ported**: it
  needs a live model/SAE (torch, transformer_lens, SAELens), out of scope
  for a numpy-only instrument, and was never part of the pre-registered
  H-E6 pass/fail criterion.
- **Detail:** `VALIDATION.md` v2-modules row 6, `benchmarks/DARKMATTER.md`,
  `tests/test_darkmatter.py`.

---

## v2 — `whiteboxtopo.gluing` (E3)

### `explanation_obstruction`, `GluingReport`

- **Claim:** INSTRUMENT — a diagnostic measurement (frustrated-cycle
  obstruction Ω on the nerve of an overlapping cover of examples, Čech-
  style, + prompt/example-label shuffle null band, `p_null`, bootstrap CIs
  on Ω/J/m_e, blocking pairs, nerve cyclomatic number,
  `significant_vs_null` boolean). **Explicitly not a scored "distinct
  mechanisms" detector.**
- **Evidence (E3, GPT-2 small, IOI vs an induction control, 2400
  prompts/task, 32/491-edge EAP graph):** unit tests (numpy-only,
  model-free) — a homogeneous cover gives Ω≈0 inside its own null; a
  planted two-mechanism cyclic cover with one asymmetric bridging overlap
  gives Ω>null with the planted blocking pair recovered exactly, robust
  across 4 seeds; shuffling cluster membership collapses the planted Ω to
  ≈0; the closed-form obstruction matches brute-force cycle enumeration on
  a hand-built 5-cycle and 30 random small nerves; the E3 headline (both
  summary rows + the top IOI blocking pair) reproduces from the committed
  CSVs.
- **Nulls:** example/prompt-label shuffle null band for Ω itself; the
  pre-registered matched single-mechanism control (induction, which should
  land inside its own null if the instrument is trustworthy).
- **REFUSED by design — no `verdict`, no `localisation` claim, no
  `Ω > threshold ⇒ multiple mechanisms` gate anywhere in the API (checked
  by a unit test asserting the dataclass has no such field):** the
  pre-registered matched single-mechanism control **FAILED its own
  null-band criterion** — Ω=1.517 outside null band [0.564, 0.991],
  p_null=0.000 at the summary config, and outside the null in 34/40 swept
  grid cells. IOI clears its *own* null (Ω=0.224, p_null=0.005, two
  blocking pairs with CI on m_e excluding 0) but shows **smaller raw Ω than
  the failed control in 40/40 grid cells** — the pre-registered direction
  is wrong everywhere tested. The one IOI blocking pair that clears its own
  null does **not** survive a second-embedding (layer-6) robustness check.
- **Failure modes / do-not-use:** `significant_vs_null=True` supports
  "distinct mechanisms" **only if a caller independently establishes**
  (a) a matched single-mechanism control on the same architecture/embedding
  lands inside *its* null band, AND (b) the blocking pair survives a
  second, independent embedding — **neither held in E3's own battery**, and
  this module cannot check either condition for you. Frustration requires a
  cycle in the nerve; a clean two-mechanism split along a tree-shaped cover
  gives Ω=0 by construction — easy to misread as "the instrument found
  nothing" rather than "there was nothing non-2-colourable to find." One
  battery so far: GPT-2 small, 2 tasks, one embedding family, one cover
  construction (2-nearest-centre).
- **Detail:** `VALIDATION.md` v2-modules row 7, `benchmarks/GLUING.md`,
  `tests/test_gluing.py`.

---

## Experiments that produced no API at all

**E1 (gauge orbits vs circuit-explanation multiplicity)** is not listed
above because it shipped **zero** public API — both pre-registered
hypotheses failed in every tested cell (orbits/explanations ∈
[0.997, 1.000]; slope difference exactly 0 — gauge explains at most ⅓ of
the multiplicity even under an exactly invariant coarse functional key).
There is no "orbit-collapse" function to refuse; the negative is
documented in `AUDIT.md`'s v2-findings table so nobody re-derives it.

## What a researcher can act on today

Unchanged from `VALIDATION.md`'s closing section, restated here because it
is the one-paragraph version of everything above: a converged CNN
classifier or GPT-2-style LM whose `memorization_signal` interval sits in
the battery's memorising range is fitting noise (outside these two
families, run the battery scripts on your own architecture first — the
sign is not known); `topological_film` gives calibrated-uncertainty
trajectories cheaply; before publishing a per-neuron or activation-topology
claim about a ReLU MLP, run `check_invariance` on it; everything else in
this library — including "how legible is this model" (`legibility_score`),
"is dark matter structured" (`residual_structure`'s verdict), and "does
this explanation generalize" (`explanation_obstruction`) — is a measurement
or a documented negative result, not a verdict.
