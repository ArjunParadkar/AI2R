# Phase 1 — Is the core measurement fast and trustworthy at scale?

Hardware: RTX 5070 Laptop 8 GB, 20 CPU threads. Backend `giotto-ph` (Ripser
core, parallel), distance matrix on GPU. All numbers reproduced by
`benchmarks/phase1_*.py`; raw CSVs and 300-dpi figures in `benchmarks/results/`.

## Gating result: **the topology is fast enough — no approximation is needed.**

The premise "PH on real activations is expensive" does not hold at the
operating point the WHITEBOX results were obtained at (probe ≤ 1000 points,
PCA→30, 95th-percentile threshold, H0+H1):

| probe n (GPT-2 L12) | exact PH seconds | H1 bars |
|---:|---:|---:|
| 150 | 0.02 | 62 |
| 500 | 0.11 | 347 |
| 1000 | 0.37 | 899 |
| 2000 | 1.2 | 2078 |
| 4000 | 6.4 | 4858 |
| 8000 | 39 (5.8 GB RAM) | 11349 |

Scaling ≈ n^2.3. A lab recording 5 layers per epoch with a 500-point probe
and 20 bootstrap subsamples pays **6.7 s/epoch at GPT-2 124M and 15.7 s at
GPT-2 XL (1.5 B, fp16, 3.7 GB VRAM)** — and most of that is the *forward pass*
(extraction: 2.4 s → 106 s from 124 M to 1.5 B), not the topology. The PH
cost is model-size-independent once activations are in hand.

Where it would break: probes above ~10⁴ points (RAM grows as n²; ripser's
H1 columns dominate). That regime is not needed for any signal in AUDIT.md.

## Backend findings (`phase1_gpt2_timing.csv`)

* `giotto-ph` without edge collapse is the fastest exact backend for n ≥ 500.
  **Edge collapse (Boissonnat–Pritam) is 10–25× slower** on these dense
  30-dim clouds (n=2000: 36 s vs 1.5 s) — now off by default.
* Single-threaded `ripser` is within 1.5× of giotto-ph up to n=1000; kept as
  the reference backend.
* Ripser++ (CUDA) not usable: no wheels past cp37, no `nvcc` on this box.
  Given the numbers above it would not change any decision.

## Approximation validation (`phase1_approx_summary.csv`, `phase1_approx_vs_exact.png`)

24 clouds, n=500: 6 synthetic manifolds × 3 seeds embedded in R³⁰ plus real
GPT-2 activations (layers 0/6/12) and their feature-shuffled counterparts.
Ground truth = ripser, full cloud, no threshold.

| method | Spearman ρ (log TC) | median rel. TC err | β₁ rank ρ | β₁ exact match | W₂ to exact H1 | s |
|---|---:|---:|---:|---:|---:|---:|
| thresh95 (original) | 1.000 | 0.00 | 1.00 | 100 % | 0.00 | 0.11 |
| giotto-ph (+collapse) | 1.000 | 0.00 | 1.00 | 100 % | 0.00 | 0.52 |
| landmark 300 | 0.984 | 0.07 | 0.65 | 0 % | 4.5 | 0.11 |
| landmark 200 | 0.983 | 0.09 | 0.24 | 0 % | 6.6 | 0.04 |
| landmark 100 | 0.967 | 0.19 | −0.12 | 0 % | 7.4 | 0.01 |
| subsample 300 ×10 | 0.990 | 0.15 | 0.67 | 0 % | 4.1 | 1.26 |
| subsample 200 ×10 | 0.964 | 0.29 | 0.50 | 0 % | 5.7 | 0.46 |
| subsample 100 ×10 | 0.945 | 0.60 | 0.13 | 0 % | 6.6 | 0.09 |

Conclusions:
1. The original 95th-percentile threshold loses **nothing**: no H1 bar in any
   of the 24 clouds dies beyond the 95th-percentile distance. It is exact for
   the finite diagram.
2. Landmark/witness-style sparsification preserves the *ordering* of TC
   across clouds (ρ ≈ 0.98) but not its magnitude (7–19 % error, saturating
   with n) and destroys β₁ entirely. Since exact PH is cheap, **the `landmark`
   backend is shipped but not used by any signal.**
3. **TC and β₁ are size-dependent** (TC ≈ linear in n; a 250-point subsample
   of a 500-point cloud carries ~45 % of its TC). Bootstrap subsampling
   therefore gives a *spread*, not a point estimate — `BootEstimate` reports
   the full-probe point value and the subsample CI separately, and the Phase 2
   signal must compare models at a *fixed* probe size or normalise.

## A finding that changes Phase 2

Feature-shuffled GPT-2 activations (each PCA coordinate independently
permuted across sentences — destroys all joint structure, keeps marginals)
still carry 77–86 % of the real cloud's TC (L6: 275 → 212; L12: 212 → 182).
**Raw TC magnitude is dominated by the spread of the activations, not by
their loop structure.** The memorization result (2900× collapse) is real, but
it works because label-noise training *shrinks* the representation, not
because loops vanish per se. The Phase 2 signal will therefore be calibrated
against a shuffle null (TC / TC_shuffled) and a scale-free companion (Rucco
persistent entropy), and validated against the label-noise battery — a raw
TC number would not survive a change of activation scale.

## Deliverable

`whitebox.topological_film(model, probe, layers, checkpoints=...)` and the
live `FilmRecorder`: per-(layer, epoch) diagrams on the full probe plus
20 paired bootstrap subsamples; `film.tc/sv/betti/fpl(layer)` return
`BootEstimate`s with 95 % percentile intervals; `film.summary()` for plotting.
Verified on GPT-2 124M → XL. Tests: `tests/test_core.py`.
