# VALIDATION — where each WHITEBOX output is reliable, and where it is not

Rule: nothing is exposed as a *signal* (a number a researcher should act on)
without a row here that reports its ground-truth check. Instruments (raw
measurements) are exposed with their measurement characteristics only.
All numbers come from `benchmarks/` (CSV tables and 300-dpi figures under
`benchmarks/results/`), hardware RTX 5070 Laptop 8 GB, dates 2026-08-28/29.

## Signals

| API | ground truth | reliability | fails / do not use | detail |
|---|---|---|---|---|
| `memorization_signal(…, family="cnn_classifier")` | 21 ResNet-18/CIFAR-10 models, label noise 0–100 %, every noisy model verified to reproduce 100 % of its flipped labels | leave-one-seed-out AUC **0.98**, in-sample 1.00; monotone in noise; `tc_ratio` size-stable (subsample/full 0.96) | other architecture families (sign flips: applied to GPT-2 AUC 0.67); checkpoints before the loss plateaus (classes overlap at epochs 10–20); probes ≠ ~500 points; layers other than the penultimate pooled features; absolute score is a ranking, thresholds are battery-relative | PHASE2.md |
| `memorization_signal(…, family="lm_transformer")` | 10 GPT-2 124M fine-tunes with LM-target noise 0–100 % (+ pretrained reference); noisy models reproduce 33–39 % of corrupted targets; **all** fine-tunes overfit | leave-one-seed-out AUC **1.00** on 8 labelled models | same as above, and: tiny battery (2 seeds), "generalising" class is itself overfit — the calibration separates *degrees* of overfitting, applied to a CNN AUC 0.00 | PHASE2.md |
| `representation_correspondence` | synthetic manifolds (circle/torus/sphere, known Betti numbers) and 33 real models with known test accuracy | **instrument only**: `iso` control = 1.000 on every manifold; untrained nets 0.80–0.95 | pre-registered hypotheses refuted: generalisers score *lowest* on images (CIFAR 0.05, AUC gen-vs-mem 0.62; MNIST memorisers 0.25 vs generalisers 0.125); no relation to test accuracy (ρ −0.20 / −0.41, n.s.). Measures "how much input topology a layer retains", not understanding | PHASE3.md |

## Candidates that did NOT pass (not in the public API)

| claim (paper) | test here | result |
|---|---|---|
| P4 failure neighbourhoods: misclassified inputs sit in sparser layer-3 regions | 2 ResNet-18 (test acc 0.89), 20 subsample TC draws per group, **label-permutation null** (the paper's Mann-Whitney over bootstrap draws is pseudo-replicated: p = 0.011 / 0.0003 there vs permutation p = **0.36 / 0.09**); per-sample kNN-radius as a misclassification score vs softmax confidence | direction reproduces (TC −13 % / −18 % for misclassified) but not significant under a proper null; per-sample AUC **0.54 / 0.53** vs softmax **0.90 / 0.90**, combining hurts (0.82). Softmax confidence already contains everything topology offers here. **Not shipped.** |
| P1: top-3 FPL predicts per-class accuracy (ρ = −0.84 on 5k-subset models) | same statistic on 2 full-data models | ρ = −0.53 (p = 0.12), −0.27 (p = 0.45). Same sign, not significant, n = 10 classes. **Not shipped**; `top_k_lifetime` stays an instrument. |
| P5: residual topology predicts fragility under input noise | per-class FPL / TC vs accuracy drop at σ = 0.1 | ρ 0.47 / −0.28 (FPL), 0.56 / 0.22 (TC); all p > 0.08. Null, as in the paper's own 3-hour run. |
| P6 memorisation = TC/β₁ collapse, architecture-universal | Phase 2 battery | the collapse is a **failure-to-learn** artefact (models that never fit their noisy labels); genuine memorisers show none; the real signature's sign flips between CNN and LM. Universality refuted. |
| H3 bias precedence, P3 intervention transfer, H5 cross-modal timing | not re-tested (bin 2 of AUDIT.md; already negative/mixed in the original record) | no API. |

## Instruments (measurement characteristics)

| API | what it measures | characteristics |
|---|---|---|
| `compute_diagrams` / `topological_film` / `FilmRecorder` | VR persistence (H0, H1) of a fixed-PCA probe cloud per (layer, epoch), with paired subsampling intervals | exact; 95th-percentile threshold loses nothing on 24 test clouds; 0.11 s at n = 500, 39 s at n = 8000 (≈ n^2.3); TC, β₁ are size-dependent (TC ≈ linear in n) — compare at fixed probe size only | 
| `structural_velocity`, `topological_complexity`, `persistent_entropy`, `betti_at_scale`, `top_k_lifetime` | diagram functionals | tested against known diagrams; SV(D, D) = 0, W₂(single bar, ∅) = lifetime/√2; two β₁ scale conventions exist (AUDIT.md) — state which you use |
| `bias_top_permutation_test` | W₂ between two subgroups' diagrams under a permutation null | false-positive rate at α = 0.05: **0/30 and 0/30** (conservative); power at n = 100 per group: 0.4–0.6 when subgroup B is an entirely different class, 0.0–0.3 at 10–50 % mixing → **underpowered for realistic bias**; use n ≥ 300 per group and treat p > 0.05 as "not detected", not "no bias" |
| `landmark` backend | maxmin-landmark approximate PH | preserves TC ordering (ρ 0.98) but not magnitude (7–19 % error) and destroys β₁; not used by any signal |
| `correspondence` / `rtd` | RTD (Barannikov et al. 2022) and 1 − RTD/RTD_null | exact reimplementation; ~1–2 s per cross-barcode at n ≤ 400, up to 35 s for layers with many long-lived features |

## What a researcher can act on today

1. A converged CNN classifier or GPT-2-style LM whose `memorization_signal`
   interval sits in the battery's memorising range is fitting noise: revisit
   labels, regularisation, or data size. Outside these two families, run the
   battery scripts on your own architecture first — the sign is not known.
2. `topological_film` gives calibrated-uncertainty trajectories cheaply
   enough to log every epoch; the phase-transition SV peak (paper H1) is
   reproducible across seeds but carries no validated decision rule.
3. Everything else in this repository is a measurement or a documented
   negative result.
