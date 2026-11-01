# Block V — Legibility: what makes a network explainable (pre-registration)

_Naming: every run in this document is a horse from the Tracen roster (`experiments/HORSES.md`); the handbook codes below are the pre-registration labels and appear here only as such._

Status: SPEC written 2026-09-07 before any Block-V run. Novelty: phase diagram **OPEN** (Golechha
et al. qualitative); V5 drift **FLAGGED** (Filan et al. clusterability during training). Prior
in-house: E4 (2026-08-30, 87 models): no structural measure predicts discovery success within an
architecture (ρ(Q, S) pooled −0.37 is an architecture offset; partial ρ 0.03); no knee; the fixed
pipeline's cost/N is 1–2 by construction. Block V re-runs with the handbook's larger, regularised
ensemble, saved checkpoints, the rewired null, and the changepoint test; E4 is the prior.

## V1 — Ensemble  [5070]
Tasks: modular addition (p = 113, a + b mod p, `=` token), sorting (length 8, digits 0–7),
copying (length 8). Architectures: 2- and 3-layer MLPs (widths 64, 128, 256, 512) and 2-, 3-,
4-layer transformers (d = 128, 4 heads, d_mlp 256). Regularisation grid: L1 ∈ {0, 1e-4, 1e-3} ×
weight decay ∈ {0, 0.1, 1} × dropout ∈ {0, 0.2} = 18 arms; to stay within one GPU-day the grid is
crossed with (task × architecture) as a **fractional design**: every arm appears with every task
on a rotating subset of 2 architectures (Latin-square assignment, seed-fixed), giving
3 tasks × 18 arms × 2 archs = 108 models; plus 10 clusterability-loss models (Golechha recipe,
λ ∈ {5, 40}, 5 (task, arch) pairs), 6 random-label, 6 untrained → **130 models** (≥ 60). Training:
AdamW, lr 1e-3, batch 256, up to 20 000 steps or eval acc ≥ 0.99; checkpoints every 5 % of the
step budget (20 per model) for V5/Z1/Z2. `results/V1_models.csv`.

## V2 — Structure  [5070]
S4 attribution graphs (EAP, mean corruption, 512-example batch) at edge densities
{0.01, 0.05, 0.1}: Q (Leiden), treewidth bound (min-fill), sparsity; degree-preserving rewired null
for Q (20 rewirings) → Q_excess. `results/V2_structure.csv`.

## V3 — Discovery cost  [5070]
Frozen pipeline (`discovery.find_circuit`, F ≥ 0.9): wall time, cost (forward passes), size,
faithfulness, 3-seed Jaccard stability, on 3 disjoint 256-example batches. `results/V3_discovery.csv`.

## V4 — Phase diagram  [LAPTOP]
Success = faithfulness ≥ 0.9 reached with |C|/N ≤ 0.5 (a circuit is "found" only if it compresses;
E4 kill 3: untrained models succeed trivially with the full set — the accuracy guard also applies).
Logistic regression of success on (Q_excess, tw/N, sparsity) with and without a changepoint in
Q_excess (2-piece); **pre-registered: changepoint model beats linear by BIC > 10**; also the
continuous cost (1 − |C|/N) vs Q_excess with a piecewise-linear knee, same BIC test. Within-
architecture partial correlations reported (the E4 confound). Figures `fig/V4_phase_Q.png`,
`fig/V4_phase_tw.png`, 2-D (Q, tw) heatmap.

## V5 — Drift  [5070]
V2 at every saved checkpoint (density 0.1) + discovery cost at 5 checkpoints; Q_excess and cost vs
training fraction; random-label trajectory as null; direction of drift (sign of the slope of
Q_excess vs fraction, bootstrap CI over models) and knee location relative to the loss curve
(fraction at which loss reaches 50 % of its total drop).

## V6 — v1's loss-invisible reorganisation, certified  [5070]
SV (2-Wasserstein between consecutive-checkpoint diagrams, PCA-20 clouds of the penultimate
layer) on the V1 checkpoints; feature-shuffle null on each peak (peak = SV above the 95th
percentile of its own trajectory); `check_invariance` of SV under the model's gauge (MLP: unit
perm/scaling on the cloud columns; transformer: MLP-unit perm on the MLP-hidden cloud).
Survives iff ≥ 50 % of peaks exceed the shuffle null AND SV is invariant; otherwise retired.

## V7 — Pythia drift  [HPCC]  `hpcc/mejiro_ardan.py`.

## V8 — Ship legibility_score  [LAPTOP]
Calibrate on V1 with 20 % held out (stratified by architecture); held-out AUC for "discovery will
succeed". Ships only if held-out AUC ≥ 0.75 AND within-architecture AUC ≥ 0.65 for every
architecture; otherwise `legibility_score` keeps raising `NotCalibratedError` with the new numbers.
