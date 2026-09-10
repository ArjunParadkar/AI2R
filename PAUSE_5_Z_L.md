# PAUSE 5 — Blocks Z (topology beyond PH) and L (release gates) — FINAL 2026-09-10 (all of Z1–Z5 and L1–L6, L8–L10; L7 on the HPCC)

_Naming: every run in this document is a horse from the Tracen roster (`experiments/HORSES.md`); the handbook codes below are the pre-registration labels and appear here only as such._

_Handbook §13 format. Z1–Z3 run in `wb-gpu-queue` after C1–C3; Z4/Z5 in `wb-cpu-queue` after D1/M1. Their sections are filled
when the CSVs land. The L section below is final for the laptop gates; L7 is an HPCC job._

## Block L — Library release gates (L1–L10)

### 1. Plain English
Every gate that can be met on the laptop is met and is checked by a file in the repository, not by a sentence in a report.
The one gate that needs a cluster (L7, Gemma-2-9B end-to-end) has its script written and its "does not scale" list
pre-declared as a field of the timing file so that whatever fails on the cluster is recorded, not smoothed. The end-to-end
run on GPT-2 small (L6) takes 8.5 minutes on the RTX 5070 laptop and reproduces the F11 audit pattern on a 1 000-token
probe with 8 renamings per kind: cloud-level persistence summaries and LayerNorm-folded singular values are invariant
(CV ≤ 1e-5); per-unit activation and attention entropy are label artifacts that become invariant after gauge alignment;
per-head output norms remain artifacts under GL(d_head) changes of basis even after alignment (0.06); raw `W_fc` singular
values are covariant under the ln_2 scaling (0.26) and invariant once folded. The SAE residual at layer 6 is spectrally
structured (137 outliers of 768 on 20 000 tokens, structured fraction 0.34) and, as in K3, not topologically structured
(tc_ratio 0.09) — the two readings disagree exactly as PAUSE_3 reported.

### 2. Pre-registered thresholds and outcomes
| Gate | Requirement | Evidence | Met |
|---|---|---|---|
| L1 | every public function has a VALIDATION.md entry | `VALIDATION.md` sprint table (symmetry, invariants, nulls, graphs, discovery, topology, darkmatter) + v1/v2 tables; anything absent is not in `__all__` | yes |
| L2 | every structure detector < 0.02 structured fraction / invariant verdict on noise or anchors | `tests/test_release_gates.py` (18 tests): MP outlier fraction < 0.02 over 20 noise draws; `residual_structure` on noise → fraction < 0.02, ≤ 1 outlier, `neither`; `symmetry.hidden` on 8 random nets → no false dead unit, unexplained ≤ 2; random graph inside its rewired-Q null; tree → 0 H1; noise blob → β₁ = 0 (Rips and Hilbert); same-distribution BiasTop ≤ 1/10 rejections; homogeneous gluing cover inside its null; anchor verdicts invariant / artifact / covariant (k = 1) | yes |
| L3 | every detector recovers planted structure of known size within 10 % | same file: spikes k ∈ {3, 5, 12} exact; planted rank {5, 10} exact; ≥ 90 % of a planted fiber (3 duplicates + 2 dead) identified, dead set exact; k ∈ {4, 10} blocks exact count, ARI > 0.9; k ∈ {3, 12} planted cycles exact; circle β₁ = 1, k ∈ {4, 6} clusters exact (β₀); Mapper k ∈ {3, 5} components exact; zigzag loop extent (3, 7) of 10 exact | yes |
| L4 | every "better" claim in docs links to its baseline row | grep of README / CLAIMS / VALIDATION / benchmarks / PAUSE reports for "better than / outperforms / beats / improves on / superior": the only hits are a negation ("no better than a random re-pairing") and three pre-registered *thresholds* in SPECs ("a baseline beats it by DeLong p < 0.05" in M, "changepoint beats linear by BIC > 10" in V) plus PAUSE_3's K4 ridge-vs-shuffle R² gap, each of which names its baseline row in the same sentence — the library itself makes no "better" claim | yes (recorded) |
| L5 | CLAIMS.md: claim → experiment → evidence file → null → failure mode → novelty status | `CLAIMS.md` §"Sprint modules" (6 entries) + 3 dead-claim entries (F4a-1/F5/F6, B2, K3/K5); every entry carries a NOVELTY_BOARD number | yes |
| L6 | end-to-end on GPT-2 small in one session, timing logged | `benchmarks/oguri_cap.py` → `benchmarks/results/L6_timing.json`: load 5 s, identity 22 s (max Δlogit 1.7–2.3e-3 fp32), audit 308 s (9 quantities × 4 kinds × 8), darkmatter 172 s, total **507 s**; `hidden` recorded not-applicable (ReLU-MLP only); gluing and legibility recorded not-covered with the reason | yes |
| L7 | end-to-end on Gemma-2-9B on HPCC, timing, what doesn't scale | `hpcc/kawakami_princess.py` written (bf16 identity floor, audit, K8-cache darkmatter, `does_not_scale` list) — **not run** (no local GPU holds 9B) | pending HPCC |
| L8 | pip-installable; CI green on 3.10–3.12 | `uv build` → `whitebox_topo-0.1.0-py3-none-any.whl` + sdist; fresh Python 3.10 venv installs the wheel with `[dev]` and imports `whiteboxtopo.__all__` (38 names); full suite in that venv after the `[topology]` fix: **138 passed, 4 skipped (dionysus), 0 failed** (106 s); CI matrix widened to 3.10 / 3.11 / 3.12 (`.github/workflows/ci.yml`) — CI itself runs on push (repo not yet pushed) | local yes; remote pending push |
| L9 | README pitch: "the auditor", never "opened the black box" | `README.md` now opens "whitebox-topo — the auditor"; the only occurrence of "open the black box" is the negation "We did **not** open the black box" | yes |
| L10 | NOVELTY_BOARD.md finalised, every claim re-searched within 14 days | `NOVELTY_BOARD.md`: 12 claims, searches 2026-09-07 (F, K) and 2026-09-09 (M, C, B, V, Z, D); next re-search due before release, ≤ 2026-09-23 | yes (dated) |

### 3. Kill-test outcomes
* L2/L3 are themselves the kills for the detectors; two gates failed on first writing and were **fixed in the test, not the
  detector**, with the reason recorded in the test file: (i) `symmetry.hidden` on random nets reports genuine dead units
  (random biases on the [−1, 2]³ probe region) — the anchor is therefore "no false positives + nothing unexplained", not
  "excess = 0"; (ii) a 40-point circle at scale 0.7 has a sampling gap wider than the scale in one of ten seeds — the
  planted loop uses 60 points. A third first-write failure was a counting error in the test (the infinite H0 bar is
  dropped from the diagram, so β₀ = finite bars + 1).
* L8 fresh-venv run (Python 3.10.20, wheel + `[dev]`): first attempt failed at collection with `No module named
  'conftest'` — caused by the `--import-mode=importlib` flag I added, not by the package. The rerun without it:
  **133 passed, 4 skipped (dionysus), 5 FAILED — `gudhi` and `kmapper` were imported by shipped `topology.graph_ph` /
  `topology.mapper` but declared in no extra.** A real L8 miss, fixed by adding a `[topology]` extra (gudhi, kmapper,
  networkx) and both packages to `[dev]`; third run: 138 passed, 4 skipped, 0 failed.
* L6 kill (flips vs F11): none of the nine verdicts differs from the F11 table (same quantities, 1 000 vs 2 000 tokens,
  8 vs 30 renamings).

### 4. What contradicts the plan's framing
* The handbook's L2 wording ("< 0.02 structured fraction on pure noise") fits spectral detectors; for `symmetry.hidden`
  the honest anchor is a *false-positive* criterion, because random nets are not "pure noise" in the sense of having no
  hidden symmetry — most have dead units, and the detector is right to say so (F3: 99.0 % of trained nets are G_incomplete).
* L4 is met vacuously: after PAUSE 1–4 the library claims nothing is better than anything. That is the intended outcome
  of the sprint's kill discipline, but it means L4 exercised nothing.
* L7 cannot be met locally; the handbook lists it under L, so PAUSE 5 is not final until the HPCC run.

### 5. Novelty
Board-level, see `NOVELTY_BOARD.md` (12 rows). New 2026-09-09 neighbours that changed a status: 2608.30487 Topo² (M:
TAKEN strengthened), 2608.10566 erasure counts not affine-invariant (B4: "rank is a concept dimension" TAKEN in the
negative), 2606.12138 SAE seed dependence (C: seed-consistency half TAKEN), 2601.12913 Barbiero et al. (F audit: a
theoretical framing to cite; the empirical audit remains OPEN).

### 6. Claimable / not claimable
* Claimable: "whitebox-topo v0.1.0 ships 12 detectors/instruments each gated on a noise anchor and a planted recovery
  within 10 %, an end-to-end audit of GPT-2 small in 8.5 minutes, and a dated novelty board" — every word has a file.
* Not claimable: anything about 9B models (L7 pending); "CI green" until the repository is pushed and the workflow runs;
  any of the Z results (below).

## Block Z — Z4/Z5 FINAL 2026-09-09; Z1–Z3 pending (`wb-gpu-queue` after C1–C3)

### 1. Plain English (Z4, Z5)
Z4 asked whether explanations are path-connected in the quotient of parameter space by the gauge group. Within one
network the question is vacuous — F4/F5 found K = N, so there are no two explanations in the same orbit to connect. The
fallback the SPEC pre-registered — take a valid circuit of network A, match B's units to A's by the Hungarian assignment,
interpolate linearly from A to B and ask whether A's circuit stays valid — gives a clear negative: of 380 pairs across 20
cells and both validity criteria, 11.8 % keep the circuit valid over ≥ 90 % of the path, 12.9 % have it valid at B, and
the median path is valid for only 28 % of its length, breaking first at a median 28 % of the way. The same algorithm
in a different network (F4a's "same algorithm, different weights") is not reachable along a straight line through
matched weights. Z5 confirms the covariance law on all 657 specimens: raw persistence summaries scale with exponent
exactly 2.000 under uniform per-network rescaling (R² = 1) and move under per-unit scalings (median CV 0.15 on the probe
cloud, 0.82 on the truth table), while the per-unit-standardised persistence is invariant on 100 % of specimens
(pre-registered ≥ 99 %).

### 2. Pre-registered thresholds and outcomes
| Item | Threshold | Outcome |
|---|---|---|
| Z4 within-orbit | pairs in the same orbit, different specimens | **0 pairs** (K = N everywhere) → vacuous, as the SPEC anticipated |
| Z4 cross-orbit fallback | fraction of pairs with a valid path (validity ≥ 0.9 of 50 points) | **0.118** (45/380) raw Hungarian; **0.095** after F4a's scale gauge-fix; **0.024** random-assignment null (Z4b, AMENDMENTS #1); valid at the endpoint 0.129; median fraction valid 0.28; median first break at 0.28 of the path; by criterion: `zero` 0.16, `mean` 0.08; best cells OR4_d2_w10 / OR4_d2_w6 (0.4 zero), worst THR3 / XORPAIR / EQPAIR / ONEHOT (0.0) — `results/Z4_paths.csv` |
| Z4 "π₀ of the valid set vs H₀ of the quotient" | to be reported | **not computed**: with no within-orbit pairs the quotient's H₀ is the orbit count itself (= N), and the valid set's π₀ along straight lines is what the table above measures pairwise; no component count was formed |
| Z5 TC exponent | k = 2, R² ≥ 0.99 | median component exponent **2.000** (IQR 2e-9), R² 1.000 on 657 specimens |
| Z5 H1 count, persistent entropy | k = 0 | **not separately reported**: `Z5_scaling_law.csv` carries the median over the 6 summary components (n_bars, TC, PE for H0/H1), not per-component exponents — a gap in the runner, recorded, not patched today |
| Z5 `ph_std_probe` | invariant on ≥ 99 % of specimens | **100 %** (657/657) |

### 3. Kill-test outcomes
* Z4 had no calibrated null at first writing; **Z4b (run 12:30, `results/Z4b_summary.json`) adds it**: random unit
  assignment in the same gauge-fixed frame gives 2.4 % valid paths against 9.5 % for Hungarian matching (paired Wilcoxon
  p = 7e-9; Hungarian better on 54 % of pairs, the null better on 32 %). Matching is real but weak: nine of ten matched
  straight lines still break, at a median 20 % of the way. The scale gauge-fix is verified exact (validity unchanged on all
  380 pairs). A same-circuit-different-seed control is still not run (no within-orbit pairs exist to build it from).
* Z5 kill is F9/F10's own: the anchor `weight_product` is invariant on every specimen in the same table.

### 4. What contradicts the plan's framing
* Z4's premise (there are explanations to connect within an orbit) was already dead after PAUSE 1; the block ran its
  fallback and produced a negative that agrees with F4a-1 (orbits do not merge explanations across seeds) — the same
  finding seen a third way, not new evidence.
* At first writing I expected F4a's scale gauge-fix to raise the valid-path fraction. It did not (0.118 → 0.095): the
  breaks are not a scale artifact of the straight line, they are the path leaving the region where A's circuit computes
  the function. Stated as a wrong expectation, not smoothed.

### 5. Novelty
NOVELTY_BOARD #12 (OPEN; nearest 2505.23681 mode connectivity via parameter-space symmetry) and #11 (Z1/Z2 pending).

### 6. Claimable / not claimable
* Claimable: the Z5 covariance law (exponent 2 on 657 nets; standardisation makes persistence gauge-invariant) — an
  instrument statement backing `topology.normalized_persistence`.
* Claimable (weak, calibrated): "after exact gauge-fixing, unit-matched linear paths between two networks' weights keep a
  circuit valid 4× more often than randomly matched ones (9.5 % vs 2.4 %, p = 7e-9), but 90 % of matched paths still
  break" — a negative for straight-line connectivity of explanations across seeds, consistent with F4a-1.
* Not claimable: any statement about the quotient's π₀ / H₀, or about non-linear (mode-connecting) paths, which were not
  searched for.

## Block Z — Agnes Tachyon (Z3), Seiun Sky (Z1), Manhattan Cafe (Z2) FINAL 2026-09-10

### 1. Plain English (Z1, Z2)
Seiun Sky ran the zigzag on 18 models (12 real-label, 6 random-label; 20 checkpoints each) under Z AMENDMENTS #2/#3 — normalised
clouds, the final cloud's most-loops scale, a simplex budget that lowered the subsample on 8 of 18 models (5 to 100 points, 3 to 50).
The pre-registered question was whether the per-checkpoint "film" (diagram per checkpoint + greedy bar matching) is a valid tracker:
it is not, on the pre-registered criterion. The film reports about three times as many births and deaths as the zigzag sees
(median 314 vs 106 events on real-label models); the fraction of film events the zigzag does not recognise has median 0.39 on
real-label models (7/12 above the 0.2 bar; bootstrap CI of the median 0.00–0.65) and 0.49 on random-label ones — with no
difference between the two (Mann-Whitney p 0.70). The zigzag's own H1 is mostly short-lived: a median of 21 % of bars on real-label
models last ≥ 3 positions (47 % on random-label). Three of the 18 rows are degenerate (the budget cut them to 50 points and no H1
bar survived at any scale — events 0, fraction 0), and two MLP rows have *more* zigzag events than film events (the one-sided
"mislabelled" statistic reads 0 there). Manhattan Cafe's two-parameter summary (scale × "checkpoint at which the point
stabilises") fails its pre-registration: the best two-parameter correlate of discovery cost is the H1 argmax scale (ρ 0.41) against
0.32 for the one-parameter total persistence — a gain of 0.09 whose bootstrap CI (−0.11, 0.27) crosses zero; for circuit size the
gain is 0.05 (CI −0.51, 0.47). Three of 20 bifiltrations are empty (transformer points never "stabilise" under the 10 %-of-diameter
rule: `frac_stable_by_half` median 0.00 on transformers vs 0.19 on MLPs).

### 1b. Plain English (Z3)
Graph persistent homology of the 130 attribution graphs (density 0.1) adds nothing to the V4 logistic: ΔBIC for β1 finite,
β1 essential and total H1 persistence over (Q_excess, tw/N, sparsity) is +1.0, −3.0, −2.8 against a pre-registered > 10. The
reason is structural, not statistical: an MLP attribution graph is layered (edges only between consecutive layers), so its clique
complex has no triangles — every H1 class is essential, β1 finite = 0 and total persistence = 0 on all 73 MLPs, and β1 essential is
just the cycle rank E − V + components (median 830, up to 25 282), a function of the edge count the density already fixes.
Transformers have triangles (attention mixes positions) and finite bars (median 6, up to 3 823) but they do not move the fit.

### 2. Pre-registered thresholds and outcomes
| Item | Threshold | Outcome |
|---|---|---|
| Z3 ΔBIC of β1 over (Q_excess, tw/N, sparsity) | > 10 | β1_finite **+0.98**, β1_essential **−2.98**, total_persistence_H1 **−2.80** (n = 130; `Z3_deltabic.json`) — **fail** |
| Z3 features | reported | MLP (73): β1_finite 0, TP_H1 0 on every graph; β1_essential median 830; transformer (57): β1_finite median 6 [0, 3 823], β1_essential median 11, TP_H1 median 0.37 — `Z3_graphph.csv` |
| Z1 zigzag vs film | mislabelled fraction > 0.2 → film invalid | **0.39 median (real-label, n 12; 7/12 > 0.2; CI 0.00–0.65), 0.49 random-label (5/6 > 0.2)** → film invalid on the criterion; real vs random p 0.70; film/zigzag event ratio ≈ 3; 3/18 rows degenerate (50-point budget, no H1); `Z1_zigzag.csv`, `Z1_summary.json` |
| Z1 settings actually run | 150 points, absolute median scale (SPEC) | normalised clouds; scale quantile median 0.07 (0.005–0.16); n_sub 150 on 10, 100 on 5, 50 on 3 models (AMENDMENTS #2, #3) |
| Z2 two-parameter summary | \|ρ_2p\| − \|ρ_1p\| > 0.1, bootstrap CI > 0 | cost: best 2p = argmax_H1_scale ρ 0.41 vs 1p 0.32, gain **0.09, CI (−0.11, 0.27)** — **fail**; size: gain 0.05, CI (−0.51, 0.47) — **fail**; 3/20 empty bifiltrations; `Z2_multiparam.csv`, `Z2_summary.json` |

### 3. Kill-test outcomes
* Z1's null is the random-label set: it is *not* separated from the real-label models (p 0.70) — whatever the zigzag/film
  disagreement measures, it is not learning. Z1's pre-registered criterion is met (film invalid) but the statistic is one-sided
  (film events the zigzag lacks) and 3/18 rows are empty after the budget cut; the finding is "the film over-counts events", not
  a certified event count.
* Z2: none beyond the bootstrap; the 10 %-of-diameter stabilisation rule produces empty bifiltrations on transformers.
* Z3: none pre-registered beyond the ΔBIC; the layered-graph observation is the kill — for MLPs the feature is determined by
  the edge count. Z1's first attempt (pre-registered absolute median scale, 150 points) was killed by the kernel at 19.5 GB on
its first model: the final cloud's median distance exceeds the early checkpoints' diameter, so early unions are complete
complexes (Z AMENDMENTS #2: per-checkpoint unit-median normalisation, β1-maximising scale of the final cloud, 2.5 M simplex
budget, the same 150 probe points at every checkpoint). The pre-registered setting is not reportable on this hardware.

### 4. What contradicts the plan's framing
* "Graph PH of attribution graphs" was written as if the graphs were generic; MLP attribution graphs are bipartite-layered and
  their H1 is a counting statistic. The instrument (`topology.graph_ph`) is correct; the object makes it redundant.
* Z1's pre-registered scale rule was wrong for clouds that grow during training; recorded, amended before any row was written.

### 5. Novelty
NOVELTY_BOARD #11 (zigzag/multiparameter, FLAGGED against 2603.03037) — Z3 falls under #5's phase-diagram row as a rejected
feature; nothing to position.

### 6. Claimable / not claimable
* Claimable (negative, instrument-level): "graph-PH features of attribution graphs add no information over modularity, treewidth
  and sparsity for circuit-discovery success (ΔBIC ≤ 1 on 130 models); for layered MLP graphs they reduce to the cycle rank".
* Claimable (instrument-level, narrow): "per-checkpoint diagram matching over-counts topological events by ≈ 3× relative to
  zigzag persistence on the same clouds (18 models)" — an argument for `topology.zigzag` over the v1 film, positioned against
  2603.03037 (same tool, neural responses).
* Not claimable: any training-dynamics statement from Z1 (random-label models look the same); any two-parameter advantage (Z2
  fails); the pre-registered absolute-scale Z1 (not run).


