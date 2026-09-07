# PAUSE 0 — Shared infrastructure S1–S5 (2026-09-07)

Written as a report, not a stop (Arjun's standing instruction: run without pausing; review afterwards).

## 1. Plain English
The five shared tools are built and tested. One symmetry module now serves every experiment: it
renames a ReLU MLP, the toy attention model used in F7, and Hugging Face GPT-2 (head permutations,
within-head basis changes, MLP-unit permutations, ReLU and LayerNorm-compatible rescalings) with a
verified maximum output change of 1e-14 on MLPs and 4e-15 on the toy transformer over 3 400 random
renamings, and it detects planted hidden symmetries (a duplicated unit, a dead unit) as an excess of
flat directions over the known group. The auditor `check_invariance` samples permutation-only,
scaling-only and combined renamings separately, reports per-component CVs, and classifies a metric
as invariant / covariant (with a fitted scaling exponent) / artifact; it reproduces the three
handbook anchors. The nulls library (feature shuffle, label permutation, Marchenko–Pastur +
Tracy–Widom, planted structure) is in place; attribution graphs, Leiden modularity, treewidth bound,
sparsity and the frozen discovery pipeline are wrapped from E4; the topology backend gains zigzag
persistence (Dionysus), a two-parameter Hilbert function, Mapper (KeplerMapper), clique-complex graph
persistence (GUDHI) and a scale-normalised persistence statistic.

## 2. Pre-registered tests and outcomes
| Test (handbook) | Result |
|---|---|
| S1: 1000 random g on 10 random models, max output deviation < 1e-6 | **PASS** — MLP 1.4e-14 (3 000 renamings, 10 models, perm/scale/combined); toy transformer 4.4e-15 (400 renamings incl. GL(d_head) rotations); torch `nn.Sequential` round-trip < 1e-9 |
| S1: hidden-symmetry detector recovers a planted extra symmetry | **PASS** — planted duplicate unit: excess 4 = (fan_in+1)+n_out−1, identified as `duplicate (0,0,1)`, unexplained 0; planted dead unit: excess 4, identified `dead`, unexplained 0; plain net: excess 0 |
| S2: weight product → invariant; per-unit activation → artifact; singular values → invariant (perm) / covariant (scale) | **PASS** — CV 7e-16 / 0.76 (perm) / 1.6e-16 (perm) with exponent k = 1.000, R² = 1.000 |
| S2 align: gauge alignment repairs a unit-indexed metric | PASS (artifact → invariant after `align="gauge"`; Hungarian alignment repairs the permutation part) |
| S3: MP on Gaussian noise → 0 outliers at p < 0.01 over 100 trials | **Nominal, not literal**: 2/100 trials had ≥1 outlier at q = 0.99 (expected 1 % per trial); 1/100 at q = 0.999. Reported as is; test asserts ≤ 4 (binomial tail p ≈ 0.003) |
| S3: planted 5 spikes recovered | PASS (5/5) |
| S4: attribution graph / modularity / treewidth / sparsity; planted 4-block graph Q > rewired null; `find_circuit` deterministic and faithful (F ≥ 0.9) | PASS |
| S5: zigzag on a synthetic sequence with a loop planted in clouds 2–5 of 8 | **PASS** — exactly one long H1 bar, positions [3, 12) = clouds 2..5 inclusive, β1 curve 0 outside |
| S5: graph PH counts planted cycles; Hilbert function; Mapper cycle on a circle; normalised persistence invariant under per-unit scalings (1e-8) | PASS |

Full suite: **122 passed** (`pytest tests`, 33 s), including the 96 legacy tests.

## 3. Kill-test outcomes
Not applicable at S-level beyond the above (the planted-symmetry and planted-spike recoveries are the kills).

## 4. Contradictions with the plan's framing (stated first)
* The handbook's S3 literal criterion (0 false positives in 100 trials at p < 0.01) is not what a
  correctly calibrated 1 % test does; we ship the calibrated test and state the rate.
* Zigzag on dense clouds is expensive (Dionysus is pure C++ but the union complex grows as the cube
  of the cloud size): 8 clouds × 40 dense points took > 10 min; 12-point blobs take 2 s. Z1 must
  subsample (`max_points`) and the SPEC will fix the cap before running.
* GPT-2 has fewer exact renamings than the handbook's list implies: no ReLU-type scaling exists
  (GELU is not homogeneous) and `ln_f` cannot be rescaled (tied embedding). Only LayerNorm-
  compatible rescalings of `ln_1`/`ln_2` are exact; documented in `symmetry/gpt2.py`.

## 5. Novelty searches
None at S-level (infrastructure). F13 logs the two Block-F searches.

## 6. Claimable now
Nothing scientific; the instruments exist and pass their calibration tests. Block F launched the same
day (F1: 3 648 specimens trained, 86 % converged; F2 CPU cells and the planted kill tests running).
