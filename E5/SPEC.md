# E5 — Cross-substrate legibility (pre-registration)

> **NOT FOR EXTERNAL CLAIMS until cleared with the Fiete lab.** Every number,
> figure and sentence in E5 is a preliminary result for discussion with the
> owner's mentors. Nothing here may be cited, posted or shipped.

Status: SPEC written 2026-08-30 **before** any E5 analysis code, any RNN
training, or any measurement on the connectome. `results/` is empty.
Amendments are appended at the bottom, numbered and dated, never edited in
place. Rule-5 pause points were lifted by the owner for E5; judgment calls
that would otherwise have been questions are marked **[JC]**.

## Question

Do biological circuits occupy the region of E4's legibility phase diagram
where circuit discovery succeeded, and do task-matched artificial networks
land in the same region as the biological circuit? Same instruments as E4
(`E4/e4lib`: Leiden modularity with the 7-point density sweep, weight-shuffle
Q null, min-fill treewidth bound, effective sparsity), so that biology and
artificial nets sit on the same axes. E4's own result governs the framing:
in E4, Q did **not** predict discovery success within architecture (the
pooled relation was an architecture offset and N was the strongest
predictor), so "the legible region" is defined here *empirically* from E4's
outcomes, not from a theory of Q.

## Data

**(a) Fly head-direction connectome** (hemibrain; the delivered 439 × 439
matrix at `/home/god/MIT Fiete Lab/InvProj1/data/raw/weight_matrix.csv`,
labels in `neuron_labels.csv`; read-only, nothing under that tree is
modified). Convention (settled in that project's README, 2026-08-27):
`W[i,j]` = presynaptic i → postsynaptic j, sign = presynaptic transmitter,
Dale's law exact (rows single-signed), every Delta7 row negative.
**Subcircuit used: the 148 neurons whose label starts with `EPG(PB08)`,
`PEG(PB07)`, `PEN_a(PB06a)`, `PEN_b(PB06b)` or `Delta7(PB15)`** (46 + 18 +
20 + 22 + 42), i.e. the compass ring-attractor core (Turner-Evans et al.
2017/2020, Hulse et al. 2021). **[JC]** EPGt (4) and the ER/ExR ring
neurons (inputs to the ring) and GLNO are excluded: the question is about the
recurrent attractor circuit, and the Fiete-lab project's own Tasks 1–6 used
exactly this 148-neuron 4-type core. The README warns that this truncation
drops 67 % of total |weight| of the 439-neuron matrix; E5 also reports the
same measures on the full 439 as a sensitivity row, but the pre-registered
object is the 148-neuron core.

There is no forward pass for a connectome. **The "attribution graph" of the
connectome is its signed weighted adjacency `W` itself** (the E4 measures
use |E| symmetrised, so the sign enters only through the kill attempts and
the block analysis). This is stated, not argued: the synapse-count matrix is
the only causal graph the data give.

**(b) Task-matched RNNs.** Head-direction integration (continuous-attractor
task, Cueva & Wei 2018 / Sorscher et al. 2019 style, 1-D version):
* Input per step: angular velocity ω_t (radians/step, Ornstein–Uhlenbeck,
  std ≈ 0.15 rad/step, clipped to ±π/4), plus at t = 0 only a 2-D cue
  (cos θ₀, sin θ₀) (zero afterwards), so the network must *integrate*.
* Target: heading θ_t = θ₀ + Σ ω, decoded by a **population of K = 32
  heading cells** with a softmax read-out over 32 equal angular bins. The
  task metric is therefore identical in form to E4's: mean log-probability
  of the correct class over batch × time (`e4lib.tasks.metric` is reused
  verbatim), so E4's discovery pipeline runs unchanged.
* Architecture: **vanilla RNN, tanh, N = 148 hidden units** (matched to the
  connectome's N so the size confound found in E4 is controlled by
  construction), h_{t+1} = tanh(W_rec h_t + W_in x_t + b); read-out linear
  on h_t. T = 100 steps, batch 64, Adam lr 1e-3, 3000 steps or until
  held-out bin accuracy ≥ 0.9 at the final 50 steps (checked every 250
  steps), weight decay 1e-5. **10 seeds.** [JC] Vanilla RNN because its
  recurrent matrix is a single signed N × N weight graph directly comparable
  to `W`; a GRU has three. A GRU arm (3 seeds, N = 148, graph = the
  candidate-state matrix W_hn) is run as a secondary architecture check.
* **Size arms** (kill: size confound): vanilla RNN, 3 seeds each at N = 64
  and N = 256.
* **Controls**: `untrained` (3 seeds, init only), `shuffled_target` (3
  seeds: targets permuted across the batch *and* time so no integrator
  exists; same budget), plus a **planted ring attractor** — a hand-built
  N = 148 weight matrix W_ij = J₀ + J₁ cos(θ_i − θ_j) − J_inh (θ evenly
  spaced; the classical bump-attractor profile) — measured but not trained;
  it is the "legible by construction" anchor for biology, as `copy` was for
  E4.
* Compute: CPU only (torch, 4 threads); no GPU use.

## Graphs measured per network

1. **Weight graph** (connectome and RNNs): E = signed recurrent matrix,
   E[u,v] = weight u→v (connectome: `W`; RNN: `W_rec[v,u]` transposed to the
   u→v convention). Directly comparable across substrates.
2. **Functional graph** (RNNs only): E4's edge attribution patching on the
   trained task: score(u→v) = Σ_{batch,t} Δin_v(u,t)·∂m/∂in_v(t), with Δ
   from replacing unit u's activation by its mean over the measure batch
   (mean over batch and time), in_v = pre-activation of v at t+1, m the
   task metric. Nodes = hidden units; embed/read-out not nodes (as E4).
   Measure batch: 256 held-out sequences.

Per graph, exactly E4's profile: Q_d for d ∈ {1, 2, 5, 10, 20, 50, 100} %
(top-d % of **existing** nonzero undirected edges by |weight|, Leiden
`ModularityVertexPartition`, 10 restarts, seed 0), pre-registered summary
**Q₁₀**; Q₁₀ shuffle null (20 weight permutations on the same edge slots)
and Q₁₀ excess; treewidth bound tw₁₀ (min-fill) raw and /N; Q_AUC; effective
sparsity 1 − PR/N. **[JC] Node importance for sparsity**: E4 used mean
|activation|; the connectome has none, so for *weight graphs* s_i = total
weighted degree Σ_j |E_ij| + |E_ji| (used for connectome **and** RNN weight
graphs, same rule), and for functional graphs s_i = mean |activation| as in
E4. Both are reported for RNNs. Edge sparsity (E4's `edge_sparsity`) is also
computed. **Density note**: E4's threshold is relative to existing edges;
the connectome core is sparse (fraction nonzero to be reported) while
`W_rec` is dense, so top-10 % keeps different *absolute* edge counts. A
**matched-edge-count** Q (RNN thresholded to the connectome's top-10 % edge
count, and vice-versa the connectome at the RNN's) is reported as a
secondary column so that any placement difference cannot hide in the
threshold.

Discovery outcome (S, F, Jaccard, cost) is run with E4's
`discover_circuit` unchanged for every RNN (3 discovery seeds, 256-sequence
discovery batches, learned-guard: final-50-step accuracy ≥ 0.5 as E4's
amendment 2). It cannot be run on the connectome (no forward pass): the
connectome gets structural coordinates only, and that asymmetry is stated
in FINDINGS.

## Phase-diagram coordinates and "region"

Coordinates: x = (Q₁₀, tw₁₀/N, sparsity), standardised by the mean and SD of
E4's 87-model ensemble (`E4/results/e4_models.csv`). Distances are
**Mahalanobis with the pooled covariance Σ of the E4 ensemble** in these
three coordinates.

**E4 success cluster** 𝒮: the E4 `learned == True` models in the **top
tercile of S** (S = 1 − |C|/N at F ≥ 0.9, mean of 3 seeds). Failure cluster
ℱ: bottom tercile of learned models. (Binary success was reached by 1/87
models in E4, so the binary label cannot define a cluster.)

## Pre-registered hypotheses and criteria

* **H-E5a.** The fly HD core (weight graph) lies in the region of E4's
  (Q₁₀, tw₁₀/N, sparsity) space where E4 discovery succeeded. **Pass**
  requires both: (i) d_M(connectome, 𝒮) ≤ the 95th percentile of the
  leave-one-out d_M of 𝒮's own members to 𝒮 (i.e. the connectome is not an
  outlier of the success cluster), and (ii) d_M(connectome, 𝒮) <
  d_M(connectome, ℱ). Secondary: k = 5 nearest E4 neighbours (Mahalanobis)
  of the connectome — the mean S of those neighbours vs the ensemble mean S
  of learned models (reported, no threshold). Because E4 found no
  within-architecture Q→S relation, a pass here means "biology sits among
  the models that happened to compress well", not "biology is legible
  because it is modular"; FINDINGS will say which.
* **H-E5b.** Task-matched HD RNNs (N = 148, weight graph) land in the same
  region as the connectome. **Same region** is defined numerically: the
  Mahalanobis distance between the 10-seed RNN centroid and the connectome
  point, d_M(RNN̄, fly), is **less than the median pairwise d_M among E4
  learned models of a single architecture** (the median over the three
  architectures' within-architecture medians — "closer than one E4
  architecture's own scatter"). Secondary (pre-registered nearest-neighbour
  test): among the 87 E4 points plus the connectome, the connectome is
  among the 5 nearest neighbours of the RNN centroid. Pass = primary
  criterion; the secondary is reported.
* **H-E5c (functional vs weight placement, exploratory but pre-stated).**
  For each RNN, the functional (attribution) graph and the weight graph
  are compared in the same coordinates; they "agree" if d_M(weight,
  functional) for that seed is below the H-E5b threshold. Reported as the
  fraction of seeds that agree and the per-coordinate signed differences.
  Whether they agree is a result either way: the connectome can only be
  placed by its weight graph, so if weight and functional placements
  disagree for the RNNs, the biology–RNN comparison is a weight-graph
  comparison only.

## Kill attempts (all pre-registered)

1. **Degree-preserving rewiring null of the connectome**: 100 copies via
   `networkx.directed_edge_swap` on the directed signed weighted graph
   (each edge keeps its weight; in/out degrees preserved; 10·|E| swaps),
   then the full Q₁₀ / tw₁₀ pipeline on each copy → z-scores and
   empirical p for the observed Q₁₀ and tw₁₀/N. If |z(Q₁₀)| < 2 the
   connectome's modularity is what its degree sequence gives and is not
   reported as structure.
2. The **same rewiring null for each RNN weight graph** (100 copies per
   seed; on the top-10 % thresholded weighted graph, since the dense
   matrix has trivial degree sequence) → z per seed.
3. **Block-shuffled connectome** (shuffle neuron labels). Q and tw are
   label-invariant, so this null bears on *what the communities are*: the
   normalised mutual information NMI(Leiden partition at d = 10 %, cell-type
   labels) and NMI(partition, PB glomerulus/wedge labels parsed from the
   label suffix, e.g. `_R8`, `_L2R7`) versus 1000 label permutations.
   Also the type-block weight structure: W collapsed to the 5 × 5 type block
   matrix, signed, versus label-shuffled blocks (reported for the
   sign/expected-connectivity sanity check).
4. **Size / density confound**: (i) RNN size arms N ∈ {64, 148, 256}:
   Spearman of each coordinate against N across the 16 vanilla models; if
   the connectome-to-RNN distance at N = 148 is smaller than at 64/256 the
   match is at least partly N. (ii) An Erdős–Rényi graph with the
   connectome core's N and its top-10 % edge count, 100 draws → Q and tw of
   pure size+density; the connectome's z against it. (iii) The
   matched-edge-count columns above.
5. **Threshold robustness**: H-E5a/b re-evaluated with Q₅ and Q₂₀ in place
   of Q₁₀ (and the matching tw). A placement that flips with the threshold
   is reported as threshold-dependent.
6. **Full-439 sensitivity**: the 439-neuron matrix's coordinates, to show
   how much the 148-core truncation moves biology.

## Deliverables

`results/e5_points.csv` (every network / graph / coordinate row, with the
E4 rows appended and a `substrate` column), `results/e5_sweep.csv`,
`results/e5_nulls.csv` (rewiring/ER/label nulls with z and p),
`results/e5_tests.csv` (H-E5a/b/c, thresholds, distances),
`results/e5_rnn_training.csv`; **one** 300-dpi main figure
`fig_e5_phase.png` (biology + artificial nets on E4's (Q₁₀, tw₁₀/N) and
(Q₁₀, sparsity) diagrams, E4 ensemble in grey coloured by S) plus
`fig_e5_support.png` (Q sweep per substrate, rewiring null histograms,
size arms, weight-vs-functional agreement). `FINDINGS.md` with the banner.
`e5lib/`: `connectome.py` (`load_connectome`), `hdtask.py` (task),
`rnn.py` (E4-interface RNN), `profile.py` (thin adapter
`profile_adjacency(E, importance)` that calls `e4lib.graph` — the measures
are **not** forked), `nulls.py`. `tests/test_e5.py`: connectome loads with
the expected block signs; a planted ring attractor gets high Q; the rewiring
null runs and preserves degrees.

## Amendments

1. (2026-08-30, after learning-rate/curriculum probes on seed 0 — logs kept in
   the scratchpad, no E5 result computed yet — and before the pre-registered
   run) **Task and RNN step changed so the task is learnable in the CPU
   budget.** The pre-registered recipe (plain tanh step, σ = 0.15 rad/step,
   K = 32 bins, lr 1e-3/3e-3, T = 100 from the start) sat at chance (loss
   ≈ ln 32) for 2500 steps in every probe. Changes: (i) **leaky step**
   h_{t+1} = (1−α) h_t + α tanh(W_rec h_t + W_in x_t + b), α = 0.5 — the
   standard discretised-CTRNN form used by Cueva & Wei 2018 / Sorscher et
   al. 2019; still a single N × N recurrent matrix, so the weight graph is
   unchanged in kind; (ii) **σ = 0.08 rad/step** (total drift over 100 steps
   ≈ 3.5 rad instead of a full lap); (iii) **K = 16 heading bins** (22.5°,
   comparable to fly HD precision) — the learned guard stays E4's "exact-bin
   accuracy ≥ 0.5"; ±1-bin accuracy is reported alongside; (iv) **T
   curriculum** 20 (400 steps) → 40 (400) → 70 (400) → 100, lr 2e-3 during
   the curriculum then linear decay to 2e-4 by step 4000; early stop at
   tail accuracy ≥ 0.9 checked every 250 steps after the curriculum; budget
   4000 steps. The GRU arm gets the same task and schedule (no leak; its
   gates already integrate). The shuffled-target control uses the same
   schedule. Metric, discovery pipeline and every measure are unchanged.

2. (2026-08-30, post-run, informational — no result changed.) `run.log`
   shows two run starts: 12:46:55 (killed by a machine reboot after training
   the 10 main N=148 seeds, at ~13:03–13:14) and 14:16:36 (resumed).
   `run_e5.py`'s `train_one` checks for an existing checkpoint per model id
   and, if found, loads it and sets `steps=-1` instead of retraining — the
   10 resumed `trained|rnn|N148|s*` rows in `results/e5_rnn_training.csv`
   carry `steps=-1` and `train_time≈0`. Their `acc_tail`/`acc_tol1`/`acc_all`
   are re-derived deterministically after reload (task eval batch is seeded
   999 regardless of resume) and are bit-for-bit identical to the pre-reboot
   values logged at 12:50–13:03 (e.g. seed 1: acc_tail=0.774 both times) —
   the resume did not corrupt or re-randomize any of the 10 main-arm models,
   and no measurement in FINDINGS depends on a value computed differently
   before vs. after the reboot. All non-resumed arms (GRU/size/shuffled/
   untrained) trained fresh in the second run only. Recorded because SPEC
   did not anticipate a mid-run reboot; the checkpoint-resume behavior,
   while correct here, is a code path that wasn't pre-registered.
