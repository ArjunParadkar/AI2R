# Block M — Memorization: audit and benchmark the v1 survivor (pre-registration)

_Naming: every run in this document is a horse from the Tracen roster (`experiments/HORSES.md`); the handbook codes below are the pre-registration labels and appear here only as such._

Status: SPEC written 2026-09-07 before any Block-M run. Novelty: **TAKEN** (per-sample noisy-label
detection: MOIT, CrossSplit, ELR; Fisher-rank spectral signature 2026) — no "first" claim; M2/M3
are baseline comparisons. Prior in-house: whitebox-lib Phase 2 (2026-08-29): the signal exists
per family with LOSO AUC 0.98 (CNN) / 1.00 (LM) but its SIGN FLIPS between families
(cross-family AUC 0.00); Phase 6: per-unit standardisation is ~free for CNN, costs power on GPT-2.

## Battery (v1, cached)
`benchmarks/results/battery_cifar/` (21 ResNet-18/CIFAR-10 models: label-noise levels + full-data
controls; probe activations `layer3` (500×256) and `pool` (500×512) at epochs 10/12) and
`battery_gpt2/` (10 GPT-2 fine-tunes; `L6`/`L12` residual (500×768) at epochs 0/5/10), with
per-model JSON (train acc on noisy/clean labels, test acc, memorized_flipped_frac, gap). Model
checkpoints were NOT saved by v1; the audit therefore acts on the cached clouds directly.

## M1 — Audit the survivor  [LAPTOP]
Features: `memorization_features` (TC, TC_shuffled, TC ratio, persistent entropy H0/H1, β counts)
per (model, layer, epoch). Gauge action on a cached cloud: for ResNet-18 post-ReLU channel
activations, G = channel permutations × positive per-channel scalings (BatchNorm γ/β rescaled,
compensated in every conv/linear that reads the channel — exact for the residual-stream channels
because the skip path is the identity) — i.e. the ReLU-MLP action on columns; for GPT-2
residual-stream clouds, no internal renaming (head perm, rotation, MLP perm, LN scaling) acts on
the residual stream at all, so invariance is by construction and is reported as such (the
non-trivial GPT-2 audit is F11's). Procedure: `check_invariance`-style column renamings (100 perm,
100 scale log-uniform [0.1, 10], 100 combined) on every CNN cloud; per-feature verdict; the
standardised variant (`standardize=True`) audited alongside. **Rule:** any artifact-class feature
component is dropped from the calibration and the per-family calibration re-fitted
(`memorization_cnn_classifier.json`); LOSO AUC before/after reported.

## M2 — Benchmark  [5070]
On the identical battery (same seeds, noise levels, epochs — models retrained from the v1 recipe
`phase2_battery_cifar.py` / `phase2_battery_gpt2.py` because checkpoints were not kept; the
retrained models' probe activations are checked against the cached ones (feature agreement
within bootstrap CI) before use):
* **Susceptibility to noisy labels** (Forouzesh, Sedghi & Thiran 2022): change in prediction under
  label-flip perturbation on held-out samples, model-level mean.
* **Grokking-without-test complexity proxy**: weight-norm trajectory / effective rank of the
  penultimate layer, model level.
* **ELR-style loss dynamics** (Liu et al. 2020): per-sample early-epoch loss / prediction-flip
  statistics aggregated to the model level (mean early-learning consistency).
ROC per method for "model has memorized (noise level > 0)" and for the regression target
memorized_flipped_frac; DeLong test for AUC differences vs the topological signal.
**Verdict rule (pre-registered):** if any baseline AUC ≥ topological AUC − 0.02 (DeLong p > 0.05
for the difference), FINDINGS says so and the library README positions the signal as
complementary; if a baseline beats it by DeLong p < 0.05, `memorization_signal` is marked
deprecated-in-favour-of in CLAIMS.md.

## M3 — Per-class comparison (baseline study only)  [5070]
ResNet-18/CIFAR-10, classes {0, 1, 2} label-corrupted (40 % flips within class), 12 epochs, 3
seeds. Topological signal per class (cloud = probe samples of that class) vs ELR/MOIT-style
per-sample detection aggregated per class. Metric: separation of the 3 corrupted classes from the
7 clean (AUC over classes, 10 per seed). Expected: existing methods win. Scoping result only.

## M4 — Cross-family scope  [HPCC]
`hpcc/m4_cross_family.py`: recalibrate per family (CNN, LM, ViT), cross-family AUC (v1: 0.00).

## Pass/fail summary
| Item | Threshold |
|---|---|
| M1 | every shipped feature component invariant (CV < 0.01) under channel permutations; scaling verdict reported; artifacts dropped |
| M2 | verdict rule above; DeLong p reported |
| M3 | AUC per method reported; no claim |
