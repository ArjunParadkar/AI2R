# FINDINGS — Run 2, Block A (depth–scale bipersistence of class fibers, GW distortion)

Pre-registration: `SPEC2.md` §A. Written by the Block-A agent 2026-09-20 and saved verbatim by the main session (the agent's harness could not write report files).

Block A fails by the letter on all three positive criteria, the kill test fires on both systems, and I could not write `FINDINGS_A.md`. The harness blocked that Write ("Subagents should return findings as text, not write report files") and I did not work around it. The findings are in this reply for you to save; the other deliverables are on disk. The script runs end to end in 129 s.

## Pass/fail by the letter (SPEC2 §A)

| criterion | measured | verdict |
|---|---|---|
| U_clean − U_scr > 0.4, ResNet-18, (a) Euclidean | 0.000 − 0.000 = **0.000** | NOT MET |
| U_clean − U_scr > 0.4, Z113, (a) Euclidean | 0.000 − 0.000 = **0.000** | NOT MET |
| E[GW]_scr > 2 × E[GW]_clean at ResNet penultimate | 0.0406 vs 0.0419, ratio **0.970** | NOT MET |
| Null: label-permuted U within 0.05 of 0 (clean, both systems) | 0.000 and 0.000 (20 seeds, max 0.000) | MET, but vacuously: the real-label U is 0.000 as well |
| Null: feature-shuffled GW ratio ∈ [0.8, 1.25] | 0.0519 / 0.0549 = **0.944** | MET |
| **Pass (A)** | 0 of 3 positive criteria | **FAIL** |

Secondary, reported only: under the robust-reach scale, U = 0.000 for all four models under both metrics.

## Kill test

It fires on both systems. Under (a), d* = L for every class in both the clean and the scrambled model, so "ambient fibers do not connect".

- **ResNet:** 10/10 classes never connect, in both the clean and the scrambled model.
- **Z113 clean:** 103/113 never connect; the other 10 connect only at `resid_post`, which is still d* = L.
- **Z113 scrambled:** 113/113 never connect.

Scored under (b) pullback, with the circularity caveat, U_clean − U_scr is 0.000 on ResNet and 0.000 on Z113. The criterion is NOT MET there too, so the verdict is FAIL, not PARTIAL. U is 0 under (b) because nothing connects before layer L, and d* = L is assigned both to "connects at L" and to "never".

Descriptively, at the last layer under (b):

- **Z113:** 88/113 clean fibers are connected (mean β0 1.2) against 0/113 scrambled (mean β0 33.8 of about 34 points). The scrambled model's train accuracy is 99.6%. Under Euclidean the same layer reads mean β0 4.1 vs 33.8.
- **ResNet clean (10/10 connected):** this is an artefact and not evidence.
  - Its penultimate pullback k-NN graph has 2 components (1050 + 450 points).
  - The library's cross-component fill (2 × the largest finite distance) becomes the largest MST edge.
  - ε then equals the largest finite distance, so any fiber inside one component connects trivially.
  - POST-HOC and never scored: the same rule on finite edges gives 0/10 connected, mean β0 82.8.
- **ResNet scrambled:** 0/10 connected.

## Surprises and contradictions of the brief's framing

1. **The Euclidean statistic cannot fire, and labels have nothing to do with it.** Distances concentrate: the median nearest-neighbour distance is 0.6–0.9 of the largest MST edge at most layers, while ε is 0.5 of it.
   - At ε the full cloud is 1459–1487 isolated points of 1500 on clean ResNet, and 3830 of 3830 at Z113 `embed`.
   - The scale at which a fiber becomes one piece, over the scale at which the whole cloud does, is 0.92–0.99 at every clean-ResNet layer. β0 = 1 needs that ratio ≤ 0.5.
2. **GW to the input fiber does not track memorisation.**
   - On ResNet the real GW (0.042 / 0.041 at the penultimate) sits only slightly below feature-shuffled activations (0.055 / 0.052). At layer2–4 of the clean net it is within 3% of the shuffled value.
   - On Z113 the direction is reversed. Scrambled/clean = 0.30, 0.18, 0.18, 0.23 across `embed`, `attn`, `resid_mid`, `resid_post`; for example clean 0.087 vs scrambled 0.016 at `attn`.
   - The Z113 input fiber is a regular simplex, so GW there measures distance from equidistant. It is the generaliser that builds structure.
3. **Z113 has three distinct clouds, not four.** `resid_mid` equals `attn` plus the constant "=" embedding, so every number is identical at those two layers.
4. **The feature-shuffle null is outside its band at other ResNet layers.** Ratios are 0.94, 0.75, 1.40, 0.93 at layer1–4.
   - Layer3 is the only layer where the real ratio exceeds 2 (2.12).
   - Its shuffled ratio is 1.40 and its scrambled-net cloud is outlier-dominated.
5. **A pattern in the A2 arrays (seen after the fact, no null, not claimed):** the scale at which a fiber becomes one piece, in units of the layer's median distance, falls with depth in the clean nets and barely moves in the scrambled ones.
   - Clean ResNet, class 0: 1.38 → 1.03 → 0.92 → 0.67 → 0.46.
   - Scrambled ResNet, class 0: 1.13 → 1.18 → 1.64 → 1.03 → 0.92.
   - Z113 clean: 0.82 → 0.67 → 0.31; Z113 scrambled: 0.92 → 0.87 → 0.92.

## What is claimable

- The β0-at-MST-reach statistic does not separate a generaliser from a memoriser on either system, and the mechanism above is measured. This is a negative result.
- U = 1 − E[d*/L] has a blind spot at the last layer by construction.
- GW to the input fiber does not measure memorisation: about 1× on ResNet and reversed on Z113.
- The last-layer Z113 fiber contrast (88/113 vs 0/113 under pullback, mean β0 4.1 vs 33.8 under Euclidean) is consistent with run 1. It is one seed pair, and (b) is circular.

## Timings

- **Whole script:** 129–148 s across three end-to-end runs; per-model times below are from the last (129 s) run.
- **Z113:** about 44 s per model; the `embed` layer alone takes about 26 s because of the 113×256 Jacobians.
- **ResNet:** about 18–22 s per model.
- **GW:** all solves together took under 10 s. Settings were max_iter 1000 and tolerance 1e-9; no reduction of iterations or point counts was needed.
- **Tests:** 7 pass in 2.3 s.
- **Peak RSS:** 3.9 GB in the final version; the first version broke the 6 GB rule (see Deviations 8).

## Deviations and choices

1. **Z113 "20 classes (every 6th)":** stride 6 from 0 gives only 19 classes, so the 20th is 6·19 mod 113 = 1.
2. **Full cloud and probes:**
   - ResNet uses the 1,500 labelled training probes only. Run 1 also mixed in 1,000 held-out points; SPEC2's probe definition does not, so they are left out.
   - PCA to 256 dims is fitted on those 1,500 points. Variance kept: clean 0.87 / 0.78 / 0.79, scrambled 0.88 / 0.75 / 0.60 at layer2 / 3 / 4.
   - Z113 uses all 3,830 training pairs; fibers have 21–60 points.
3. **Rips edge:** an edge exists iff d ≤ ε. β0 is computed exactly from single-linkage merge heights and tested against brute-force connected components.
4. **Metric (b):**
   - Distances are all-pairs Dijkstra on the k=15 graph, re-ranked and weighted by g.
   - ResNet layer1–4 use secant weights with logits as outputs.
   - Disconnected pairs are put at 2 × the largest finite distance (the library convention).
   - The Z113 `embed` Jacobian runs through the whole network from the concatenated token + positional embeddings of positions 0 and 1, with the hint and "=" rows held constant.
   - For `attn`, ψ(u) = ψ_mid(u + e_"=").
   - Each ψ was asserted to reproduce the model's logits.
5. **GW:**
   - Both sides use Euclidean distances.
   - All fiber points are used; every fiber is under the 300-point cap.
   - POT's default start is the independent coupling, with a single start.
   - Values are local optima.
   - Solver check: a cloud against a permuted copy of itself returns 1e-16.
6. **Label-permutation null:** seed 0 is the scored number; 19 further seeds are reported alongside.
7. **Diagnostics added after the first pass (unscored, no threshold touched):**
   - full-cloud β0 at ε;
   - the fiber/cloud connection-scale ratio;
   - GW at the independent coupling;
   - the POST-HOC finite-edge variant;
   - feature-shuffled GW at all layers.
8. **Memory:**
   - The first version peaked at 12.7 GB RSS, breaking the 6 GB rule, though nothing was OOM-killed.
   - The cause is the library's `pullback_metric_lengths`, which gathers J[i] and J[j] for all edges at once.
   - I replaced it inside the new package with an edge-chunked equivalent, tested equal to the library.
   - Results are bit-identical and the existing library file is untouched.
9. **RIVET:** not installed.
10. **Test timings file:** `tests/conftest.py` rewrites `results/test_timings.json` on every pytest session. I backed it up and restored it around each run of the new tests, so run 1's record is unchanged.

Files are in `/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight`:
- `analysis/run2_blockA.py` (`python analysis/run2_blockA.py figure` re-renders only the figure)
- `whiteboxtopo/depthprofile/__init__.py`
- `tests/test_run2_depthprofile.py`
- `results/run2/blockA.json`
- `results/run2/blockA_bipersistence.npz` (Euclidean and pullback β0(class, layer, 40 ε) arrays for all four models)
- `figures/run2_bipersistence.png` (300 dpi, Greys)
- `logs/run2_blockA.log`
