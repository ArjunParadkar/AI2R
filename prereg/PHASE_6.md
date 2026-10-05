# PRE-REGISTRATION — WHITEBOX v2, Phase 6 (biological circuits, level D)
Authors: Arjun Paradkar and Reva Paradkar (independent; no university affiliation).  Written 2026-10-05.  Tag `prereg-phase-6` (annotated, unsigned: no GPG key).
Status of the Fiete-lab collaboration: not arranged by this session; the lab's own grid-cell RNN configuration and data are not on this machine.  Everything here uses our own implementation and public data.  Coordination of data use, authorship and timelines with the lab remains Arjun's.

## Disclosed before registration
- N7 gate (level-D routes on planted populations with exact ground truth, numerics/test_n7_bio.py, docs/record.md): response route PASS (3 modules, spacing <= 1.7%, orientation <= 1.2 deg, hex R2 ~0.85, fresh seeds); connectivity route: module count exact on 10/10 fresh-seed networks, torus (b1 = 2) in 13/15 modules on each of two fresh-seed sets (FAIL by the all-modules criterion), random wiring 0 bars, ring 1 bar.  A shuffle-null bar rule was tried and rejected.  Any statement about connectivity-route recovery must carry the 87% module-level rate.
- Own path-integrating RNN (bench/bio/gridrnn.py): pilot seeds 900-902 only.  Attempt 1 (256 units, sigma bug 0.26 m, Gaussian target) 6000 steps: no grid structure.  Attempt 2 (DoG target, sigma still 0.26 m) 40,000 steps both conditions: path-integration error 0.079 m, units are boundary / band cells, 0 grid-like units, 1 module (viewed ratemaps).  Attempt 3 (sigma 0.12 m, 512 units) is running at the time of writing.
- The Gardner et al. 2022 dataset (figshare 16764508, CC0) was inspected for structure only (keys, durations, neuron counts: Q 97+66, R day1 166+168+149, S 140 neurons per module; open-field arena 1.5 m); no analysis was run on it before this registration.

## Confirmatory RNN runs (H6.1-H6.4) are conditional
They are launched only if the attempt-3 conformal pilot shows grid-like units (>= 10% of units with hex-template R2 >= 0.7).  Otherwise H6.1-H6.4 are recorded as NOT TESTABLE: no grid code emerged in our RNN (a statement about our RNN and training budget, not about the framework), and the planted-population N7 results remain the only level-D validation of the lattice and connectivity routes.  If launched: conformal x10 seeds, no-conformal x10, memorizer x5, untrained x10; 40,000 steps; checkpoints every 500 steps; thresholds: H6.1 module count equals the per-cell classical analysis and spacing within 10% / orientation within 7 deg for >= 80% of seeds (conformal); H6.2 connectivity route gives the same module count and b1 = 2 in >= 2/3 of modules, plus |Spearman| >= 0.3 between within-module weight and spatial-phase distance; H6.3 mean hex-template R2 >= 0.7; H6.4 structure absent (mean R2 < 0.3) in >= 90% of memorizer and untrained runs, present (>= 0.5) in >= 80% of conformal and >= 50% of no-conformal runs, and its onset within 20% of training length of the path-integration onset in >= 60% of conformal seeds.  Results for the conformal condition are partly by construction (the loss acts on the metric the instrument reads).

## H6.5 Gardner et al. 2022 recordings (real rodent; no loss imposed by anyone)  — the decisive biological test
Files: rat Q (modules 1-2), rat R day 1 (modules 1-3), rat S (module 1); open-field windows rat Q 27826-31223 s, rat R day 1 7457-14778 s (first valid interval), rat S 9939-12363 s.  Unit of analysis = one animal-day file (3) for module recovery and one module (6) for torus recovery.  No numbers are tuned: all parameters are in the frozen code.
**H6.5a blind module recovery.**  Pool every neuron of a file, ignore the authors' module labels, run the response route (ratemaps 40x40 bins over 1.5 m, autocorrelogram lag 20 bins, k-means on symmetry profiles, silhouette rule).  PASS on a file if the number of recovered modules equals the authors' count AND the adjusted Rand index with the authors' labels >= 0.8.  Overall PASS if >= 2 of 3 files.
**H6.5b torus.**  Per module: open-field spike counts in 0.1 s bins, Gaussian smoothing sigma 2 bins, z-score, PCA to 6 dimensions, density filter + farthest-first subsample to 800 points, Rips H1/H2 (ripser).  Null: each neuron's binned series circularly shifted independently (20 shuffles, same pipeline).  A torus is CALLED if at least two H1 bars exceed the maximum H1 lifetime over the 20 null shuffles (permutation p <= 1/21 per bar).  Also reported: whether the longest H2 bar exceeds the null maximum.  Overall PASS if the torus is called in >= 4 of the 6 modules.  A pass is shown against shuffle nulls only; it does not show that the torus is the 2-torus of the lattice group unless the module's lattice is also recovered (H6.5a).
**H6.5c ring (head direction)**: NOT TESTED in this session: no head-direction recordings small enough to download were identified (DANDI 000056 / 000939 are 200-700 GB).
**H6.6** (kappa on recorded populations flags out-of-distribution behavioural states): NOT TESTED.

## Not done in Phase 6 (stated up front)
Insect head-direction connectome re-expression in v2 (the fly data are outside this repo), the lab's own RNN configuration, level-D coupling to the lab's data.

## Frozen artefacts
| file | sha256 |
|---|---|
| `bench/bio/gridlib.py` | `f45722b215b27d7d6180af7a70b321d4331a18605043a92d1f88eb618ab74565` |
| `bench/bio/gardner.py` | `7b6822513dc653225cbe64274e493a2303f460a9cad24d82cc678ad69a6f6c7d` |
| `bench/bio/gridrnn.py` | `2c214c648c25d52cece7712a377af53e1507187f162df45bccdc5d480c57a592` |
| `bench/bio/rnn_analysis.py` | `3437ff1cef54a307a893938918e4ee081ad92208dfba5a2be8908e59f4ebc2e0` |
| `numerics/test_n7_bio.py` | `34d1809af45143ea4f8f6d99ab35b720b35c428764f348c9875dd728e4b50bf8` |
