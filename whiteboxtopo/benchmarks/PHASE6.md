# Phase 6 (v2) — `whiteboxtopo.invariants`, `whiteboxtopo.legibility`, and the gauge question for `memorization_signal`

Stage 2 of the whitebox-v2 programme (2026-08-30). Inputs: E2 and E4 of
whitebox-v2 (their result tables are copied into `results/phase6/`; their
code was ported, not imported — the library has no dependency on the
experiment tree). Figures: `phase6_figures.py` → `results/phase6/*.png`
(300 dpi). Benchmark: `phase6_standardize_memorization.py`.

## 1. `whiteboxtopo.invariants` — what ships and why (instrument)

E2 sampled 100 representatives of the gauge orbit (hidden-unit permutations
× per-unit positive scalings, log α ~ N(0, 1)) of 687 converged ReLU MLPs and
840 untrained ones, and measured the relative dispersion (CV) of 33
candidate quantities. `results/phase6/phase6_invariance_cv.png` is that
table. The library ships the **invariant column only**:

| function | E2 id | orbit CV (median, perm / scale / combined) | between-net CV |
|---|---|---:|---:|
| `logits` | I1 | 1.3e-15 / 1.3e-15 / 1.3e-15 | 0.60 |
| `input_jacobian` | I2 | 1.3e-15 / 1.3e-15 / 1.3e-15 | 0.81 |
| `weight_product` | I3 | 1.3e-15 / 1.2e-15 / 1.3e-15 | 0.60 |
| `standardized_tc` (H0, H1) | I8 | 1.3e-15 / 1.3e-15 / 1.3e-15 | 0.39 / 0.52 |
| `sorted_gradxact` | N2 sorted | 1.3e-15 / 1.3e-15 / 1.3e-15 | 0.74 |
| `sorted_ablation` | N3 sorted | 1.3e-15 / 1.3e-15 / 1.3e-15 | 0.53 |
| `gate_multiset` | N6 sorted | 0 / 0 / 0 | 0.70 |

Each is re-checked by `tests/test_invariants.py` on a hand-built XOR3 net,
random nets (1- and 3-output), and a torch `nn.Sequential` trained on XOR3,
in all three groups, at CV ≤ 1e-10; the orbit moves the parameters by a
median relative distance > 0.1 (typically 1–3), so the tests are not
near-copies; and the invariants vary across nets (vacuity control).

`check_invariance(metric_fn, model, n_reps=100, group="combined")` is E2's
audit as a tool: it accepts the E1/E2 numpy MLP (`.Ws/.bs`), a torch
`nn.Sequential` of Linear/ReLU, or `ReluMLP`; representatives are of the
input's type so a user metric keeps working; `broken_by` runs the two
subgroups separately and names the culprit. Controls in the tests: planted
invariant passes, planted `W0[0,0]` fails in every group, a non-symmetry
(layer-0 permutation not propagated) is detected, the E2 headline
reproduces (raw-cloud TC: perm exact, scale broken).

**Documented, not shipped** (the "not invariant" column; `metrics.py`
docstring): raw-cloud TC/β/PE (scale CV 1.4–3.0 for TC — larger than the
between-net CV 0.6–1.6, i.e. the gauge moves TC more than training does),
per-layer singular values (CV ≈ 1), raw |activation| and weight norms (broken
even sorted), every raw-index per-neuron reading (top-neuron agreement
0.10–0.17; "gate of neuron i" ≈ 1/w). The attribution *values* are exactly
scaling-invariant per unit; only the *labels* are gauge.

**Scope.** ReLU MLPs. The group for transformers (LayerNorm, head
permutations, residual-stream rotations), convolutions with BatchNorm, gated
or non-ReLU units is different; the tool needs a sampler for it before any
verdict there means anything. Hidden flat directions (E1: Jacobian-rank
deficits in 92 % of nets) are not sampled — `exact` is "under G".

Judgment calls: multi-output nets attribute to Σ_k f_k (grad×act) and use
‖Δf‖ (ablation); a float32 torch model is warned about (floor ~1e-7 lands in
`approximate`); `diagnose=True` costs 3× the orbit passes and is the default
because "which symmetry" is the finding.

## 2. `whiteboxtopo.legibility` — profile shipped, score refused

Port of `e4lib` (graph measures, shuffled-weight null, min-fill treewidth
bound, effective sparsity, the fixed EAP + greedy-prune discovery pipeline)
plus a reference node-instrumented two-hidden-layer `NodeMLP`; E4's
transformer models stay in the experiment tree (they are tied to E4's
tasks) but follow the same documented protocol. Tests: planted-modular graph
(Q > 0.6, 4 communities recovered) vs random; treewidth bound on trees /
cycles / cliques / a 4×4 grid; a planted 12-node circuit recovered at
F ≥ 0.9 with nothing outside it; profile end-to-end; the untrained guard
(S = 0 when accuracy < 0.5 — E4 kill 3: untrained models "succeed"
trivially at S_raw 0.93).

`legibility_score` **raises `NotCalibratedError`**. The pre-registered
calibration failed (`results/phase6/phase6_legibility_phase.png`,
`e4_tests.csv`): ρ(Q₁₀, S) = −0.37 [−0.55, −0.17] pooled over 87 models —
wrong sign, and entirely an architecture offset (MLPs Q ≈ 0.44 / S ≈ 0.19,
tf2 0.25 / 0.40, tf4 0.19 / 0.36; within-architecture CIs all cross 0;
partial ρ | arch = +0.03 [−0.26, 0.33]); no knee (ΔBIC −2.6); treewidth/N
predicts neither S (−0.11 [−0.35, 0.13]) nor cost (cost/N ∈ [1, 2.1] by
construction of a greedy pass). A number from this function would be the
architecture label. Scope of the profile: small models (N ≤ 300 nodes).

## 3. Does the gauge repair cost `memorization_signal` its power?

`phase6_standardize_memorization.py` recomputes the Phase 2 features on the
cached battery activations (21 ResNet-18/CIFAR-10 models at `pool` + `layer3`,
10 GPT-2 124M fine-tunes at `L12` + `L6`; 62 (model, layer) rows) twice — raw
(v1) and with the E2 repair (`standardize=True`: each unit z-scored over the
500-point probe, dead units dropped, then PCA→30→VR as before) — under the
same labels and leave-one-seed-out protocol as Phase 2, with a 2000-resample
bootstrap CI on each LOSO AUC and a paired bootstrap on the raw−standardised
difference. Full table: `results/phase6/phase6_standardize_auc.csv`; decision:
`phase6_standardize_report.json`.

**Pre-registered decision rule**: the default flips to `standardize=True`, and
a standardised calibration ships, only if the standardised LOSO AUC of the
*shipped* feature at the *shipped* layer is ≥ the raw one. **Both families
fail it; the default stays raw and no `_std` calibration ships.**

| family (shipped feature @ layer) | LOSO AUC raw | LOSO AUC std [CI] | paired raw−std [CI] | noise-rank ρ raw → std |
|---|---:|---:|---:|---:|
| `cnn_classifier` (`tc_ratio` @ pool) | 0.981 [0.889, 1.0] | 0.963 [0.839, 1.0] | [0.000, 0.107] | 0.77 → 0.78 |
| `lm_transformer` (`pe1` @ L12) | 1.000 [1.0, 1.0] | 0.833 [0.42, 1.0] | [0.000, 0.580] | 0.96 → 0.37 |

The two families fail differently:

* **CNN: the repair is almost free.** Every other feature set at `pool`
  (`pe1`, `tc_norm`, `tc0_norm`, `tc_ratio+pe1`, `all5`) holds LOSO AUC 1.0
  under standardisation with unchanged noise-rank ρ; only the shipped
  `tc_ratio` slips 0.981 → 0.963 (paired CI touches 0). A user who wants a
  gauge-invariant CNN memorisation readout can have one at essentially no
  cost via `memorization_features(..., standardize=True)` and their own
  calibration — the library just does not silently switch a shipped,
  battery-tested default for an equal-or-worse variant.
* **GPT-2: the repair deletes real signal.** Every single-feature set drops
  (pe1 1.0 → 0.833, tc_norm/tc0_norm 1.0 → 0.833, b1 1.0 → 0.708; at L6 b1
  inverts to 0.125), and the graded-noise Spearman collapses (pe1 0.96 →
  0.37, tc0_norm 0.98 → 0.20): part of what separates memorising from
  generalising GPT-2 runs lives in the per-unit activation scales that
  standardisation removes. `all5` recovers std AUC 1.0, but its raw AUC is
  0.83 on n = 8 held-out pairs — a 5-feature logistic fit on 8 models is not
  a calibration we will ship. Crucially this does **not** convict the raw
  signal of being a gauge artifact: a transformer's residual stream has no
  per-unit scaling symmetry (its group — LayerNorm, head permutations,
  residual rotations — is different, §1 scope), so for `lm_transformer` the
  per-unit z-score is a data transform, not a gauge repair, and the E2
  verdict does not transfer.

**Caveats.** n is small (15 CNN / 8 GPT-2 held-out (score, label) pairs; CIs
are wide); one battery per family; the shipped-feature rule was chosen before
the run but the per-family asymmetry above is a post-hoc reading. What ships:
`standardize=True` on `memorization_features` / `memorization_signal` (exact
per-unit-scaling invariance, tested), `calib_path(family, standardize=True)`
that refuses when no standardised calibration exists — which is currently
both families.
