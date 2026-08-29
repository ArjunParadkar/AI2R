# Phase 2 — `memorization_signal`: calibrated, and honest about what it is

Battery (`phase2_battery_cifar.py`, `phase2_battery_gpt2.py`; raw per-model
JSON/NPZ under `results/battery_*`, features in `results/phase2_features.csv`):

| family | models | ground truth |
|---|---|---|
| ResNet-18 / CIFAR-10, 5k subset, no aug, 40 ep | noise {0,20,40,60,80,100}% × 3 seeds | **every** noisy model reproduces 100 % of its flipped labels (train acc 1.0); test acc 0.60 → 0.07 |
| ResNet-18 / CIFAR-10, full 50k + aug, 12 ep | 3 seeds | test acc 0.89, gap 0.04 — the strong generalisers |
| GPT-2 124M fine-tune, 2k sentences, 10 ep | LM-target noise {0,25,50,75,100}% × 2 seeds | noisy models reproduce 33–39 % of corrupted targets; **all** fine-tunes overfit (train NLL 1.0 vs held-out 5.4), so "generalising" here is weaker ground truth |

Labels: memorising := noise ≥ 50 %; generalising := 0 %; 20–40 % graded (monotonicity only).

## Finding 1 — the paper's "TC collapse" was a failure-to-learn signature, not a memorisation signature

Raw TC at the penultimate layer: clean-subset ResNets 105–119, **memorising
ResNets 85–110**, full+aug generalisers 15–18. There is no collapse when the
model actually memorises. The 1000×/2900× collapses reported in the paper
came from 100 %-noise models that had *not fit the noise* (train acc 0.10
after 12 epochs on 50k random labels) — a representation that never left
initialisation. We reproduce exactly that in the GPT-2 battery: the 100 %-
noise GPT-2s, which fit only 39 % of their targets, have TC 35–52 vs ~500
for everything else. **Raw TC detects "this model learned nothing", not
"this model is memorising".** Both are worth knowing, but they are different
claims; AUDIT.md bin 1 row 1 is downgraded accordingly.

## Finding 2 — there is a memorisation signature, and its sign depends on the architecture

Scale-free features (`memorization_features`): `tc_ratio` = TC / TC of the
feature-shuffled cloud, `pe1` persistent entropy (Rucco et al. 2016),
`tc_norm`, `b1`, `tc0_norm`.

| family, penultimate layer | memorisers vs generalisers |
|---|---|
| ResNet-18 | memorisers are a **more** entropic, **less** clustered cloud: `tc_ratio` 0.97–1.35 vs 0.40–0.93 (full+aug 0.40–0.50). A good classifier collapses classes into tight clusters — few loops, *below* its own shuffle null; a memoriser is a blob. |
| GPT-2 | memorisers are a **less** entropic, more collapsed cloud: `pe1` 5.33–5.52 (4.99–5.17 at 100 %) vs 5.60–5.62. |

Consequently a calibration fitted on one family applied to the other is
**anti-predictive**: every penultimate feature set gives cross-family AUC
0.00–0.67 (`phase2_calibration_table.csv`; the ROC figure's dotted curves).
The universality claim (P6 "cross-architecture") is refuted for the
memorising regime. The signal therefore ships **per family** with `family`
a required argument; there is no auto-detect.

## Shipped calibration (`whitebox/signals/calibration/memorization_<family>.json`)

| family | feature | in-sample AUC | leave-one-seed-out AUC | applied to the other family | n |
|---|---|---:|---:|---:|---:|
| `cnn_classifier` (ResNet-18 pool) | log `tc_ratio` | 1.00 | **0.98** | 0.67 | 15 (+6 graded) |
| `lm_transformer` (GPT-2 final hidden, mask-mean) | `pe1` | 1.00 | **1.00** | 0.00 | 8 (+2 graded) |

Score vs injected noise is monotone within family (Spearman 0.56 / ρ per
family in `phase2_report.json`). Absolute scores are compressed by the small
battery: clean-subset ResNets (which overfit, gap 0.40) land at 0.55–0.60 and
only the full+aug generalisers at ~0.10. The decision threshold is therefore
the midpoint between the battery's highest generalising and lowest memorising
score (0.625 for `cnn_classifier`, 0.62 for `lm_transformer`; stored in the calibration, reported by `MemorizationSignal.threshold`),
and every result states where it sits relative to the battery ranges
(`battery_position`). Treat the score as a calibrated *ranking* with an
interval, not as a probability of anything outside this battery.

**Along training** (`phase2_trajectory.png`) — a limitation: in ResNets the
classes overlap heavily at epochs 10–20 (0 %-noise subset models score
0.43–0.71, noisy ones 0.50–0.93) and separate cleanly only at epochs 30–40,
i.e. once the noisy labels are fit. In GPT-2 only the 100 %-noise runs rise
clearly by epoch 5. **The signal is validated on converged checkpoints; its
mid-training use is not validated** and a lab should not act on it before
the training loss has plateaued.

## Uncertainty

`tc_ratio` is size-stable (250-point subsample / 500-point full = 0.96), so
its subsample interval is meaningful. `pe1` is mildly size-dependent (0.84),
`tc_norm`, `b1`, `tc0_norm` strongly (0.44–0.59) — they are reported as
features but not used by the calibration. The interval is the subsample score
spread re-centred on the full-probe score.

End-to-end on battery activations with the shipped calibration: full+aug
ResNet 0.099 [0.00, 0.38] → generalising; clean-subset ResNet 0.60 [0.38, 0.68]
→ uncertain (it does overfit); 80 %-noise ResNet 0.76 [0.62, 0.90] → uncertain
by interval but inside the memorising range; GPT-2 clean 0.553 [0.549, 0.556]
→ generalising; GPT-2 75 % noise 0.835 → memorising. The CNN intervals are
wide because `tc_ratio` inherits the variance of its shuffle null at 250
points; `n_shuffle` and `subsample` are exposed to tighten them at extra cost.

## Cost

~0.7 s per feature evaluation (500 points, PCA→30, 5 shuffles) + 20
subsamples ≈ 8 s per call on this laptop; the forward pass dominates for
large models (Phase 1).

## Where it fails / do not use

* Any architecture family other than the two calibrated ones (ViTs, RNNs,
  GNNs): the sign is unknown. Calibrate first with `benchmarks/phase2_*`.
* Probe sizes far from 500 or a different layer than the calibrated one.
* Models that never fit their data: score is undefined; check raw `tc` and
  `med_dist` for collapse (`tc` ≪ its shuffle null and a tiny `med_dist`).
* The GPT-2 side rests on 8 labelled models whose "generalising" class is
  itself overfit; the pretrained model is the only clean generaliser.
