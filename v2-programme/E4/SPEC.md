# E4 — Legibility phase diagram (pre-registration)

Status: SPEC written 2026-08-30 **before** any model in this experiment was
trained or any analysis code was written. `results/` is empty. Changes after
the first run go in "Amendments" at the bottom, numbered and dated, never
edited in place. Rule-5 pause points were lifted by the owner for E4; the
judgment calls that would otherwise have been questions are marked **[JC]**.

## Question

Which structural parameters of a trained network predict whether a *fixed*
circuit-discovery pipeline succeeds, and at what cost? Framing: Adolfi,
Vilas & Wareham, "The Computational Complexity of Circuit Discovery for Inner
Interpretability" (ICLR 2025, arXiv 2410.08025) prove that circuit-discovery
queries are NP-hard, mostly fixed-parameter *in*tractable in the model/circuit
features they examine, and inapproximable. Golechha, Chaudhary, Velja, Abate &
Schoots, "Studying Cross-cluster Modularity in Neural Networks" (arXiv
2502.02470, 2025; supersedes the withdrawn 2409.15747 "Training Neural
Networks for Modularity aids Interpretability", Golechha, Cope & Schoots
2024) show *qualitatively* that a clusterability loss yields smaller, more
disjoint circuits. Neither gives a phase diagram of discovery outcome versus
structural parameters. That diagram, and the shape of the Q→success
relation, is the only thing E4 claims. This is **empirical
parameterized-complexity evidence, not a theorem**.

## Pre-registered hypotheses

* **H-E4 (primary).** Modularity Q of the attribution graph and discovery
  success S are positively related **with a knee**, not linearly.
  Pass requires *both*:
  1. Spearman ρ(Q₁₀, S) > 0 with a 95 % bootstrap CI (2000 resamples over
     models) excluding 0; and
  2. a continuous one-breakpoint segmented linear fit S = a + b₁Q + b₂(Q−κ)₊
     (κ on a grid of the 10th–90th Q-percentiles, 41 points; 4 parameters)
     beats the straight line (2 parameters) by **ΔBIC = BIC_lin − BIC_seg ≥ 2**
     on the observed data, **and** the bootstrap distribution of ΔBIC
     (2000 resamples) has ≥ 50 % of its mass above 0, **and** a
     Q-label-permutation null (2000 permutations of Q across models) gives
     P(ΔBIC_null ≥ ΔBIC_obs) < 0.05. The knee κ is reported with its
     bootstrap 95 % CI. ΔBIC ≥ 2 and the CI of the slope difference b₂ are
     the shape claim; ρ > 0 is the direction claim.
  Partial outcome: ρ > 0 but no knee ⇒ "monotone, not knee" (H-E4 fails on
  shape, passes on direction). ρ ≤ 0 ⇒ H-E4 fails outright.
* **H-E4b (secondary).** Treewidth bound tw₁₀/N is *negatively* related to S
  (Spearman CI excludes 0). Adolfi et al.'s FPT-style intuition is that
  bounded treewidth is what makes subgraph search tractable; this is the
  direct test of that intuition on discovery *cost* as well: Spearman
  ρ(tw₁₀/N, cost) > 0.
* **H-E4c (secondary).** The Q→S relation survives controlling for
  effective sparsity and model size: partial Spearman ρ(Q₁₀, S | sparsity,
  N_nodes) > 0, CI excludes 0. If it does not, the honest reading is that Q
  is a proxy for sparsity and the FPT-conjecture is **not** stated.

## Ensemble (target ≥ 60 models; planned 84)

Architectures (3):
* `mlp`: one-hot input → 2 hidden ReLU layers of width 64 → logits.
* `tf2`, `tf4`: 2- and 4-layer pre-LN transformers, d_model 32, 4 heads
  (d_head 8), MLP width 64, ReLU, learned positional embeddings, full
  (non-causal) attention, one read-out position per output slot.

Tasks (3), all with tokens in a small vocabulary:
* `modadd`: (a, b) ↦ (a+b) mod p, p = 23, all 529 pairs; train on a random
  70 % split, evaluate on the rest. Transformer input `[a, b, =]`, read at `=`.
* `sort`: sequence of L = 4 digits in 0..7 ↦ sorted sequence (4 output
  slots of 8 classes). MLP: 32-dim one-hot ↦ 4×8 logits. Transformer: read
  slot i at position i.
* `copy`: same format, target = input (identity). Deliberately the easiest
  task; it anchors the "legible by construction" corner.

Regularisation arms (8) applied to every (arch, task) cell, one seed each
(seed = 0; **[JC]** one seed per cell, breadth over depth, because the
hypothesis is about a between-model relation and 72 trained models already
exceed the ≥ 60 target):
* `none`; `wd0.1` (AdamW weight decay 0.1); `l1_1e-4`, `l1_1e-3` (L1 on all
  weights); `drop0.1`, `drop0.3` (dropout on hidden activations); `clus5`,
  `clus40` — Golechha et al.'s clusterability loss, **re-implemented from
  their description, not ours**: for each MLP weight matrix W (MLP in/out
  matrices, the matrices they regularise), neurons on both axes are split
  into k = 4 fixed contiguous equal-size clusters and
  C(W) = Σ_{i,j: cluster(i)=cluster(j)} W²_ij / Σ_{i,j} W²_ij;
  L = L_CE + λ·mean_W(1 − C(W)), λ ∈ {5, 40}. (1 − C is −C up to a
  constant; λ = 40 is their typical value.)
* Training: AdamW lr 1e-3, batch 256, up to 4000 steps or until held-out
  accuracy ≥ 0.99 for `copy`/`sort` and ≥ 0.95 for `modadd` (checked every
  200 steps). Models that end below 2× chance held-out accuracy are kept in
  the ensemble but flagged `learned = False` (see success definition).

Controls (12):
* `randlabel` (3 archs, `modadd` with a fixed random permutation of the
  answer per input pair, trained on all pairs to memorise; seed 0).
* `untrained` (3 archs × 3 tasks, init only, no training).
* `untrained_block` (3 archs, `modadd`): init only, but every weight matrix
  masked to a k = 4 block-diagonal pattern (within-cluster entries kept,
  cross-cluster set to 0) — **high Q by construction, no function**. This is
  the "does high Q succeed trivially?" kill control.

Everything runs at one fixed batch of 256 held-out inputs (`measure batch`)
for the graph measures and three independent 256-input batches for the
three discovery seeds.

## Measurements per model

Nodes: every hidden neuron (MLP) or every attention head and every MLP
neuron (transformer). Embedding and unembedding are *not* nodes (they would
be hubs in every model and swamp Q); they enter as the source/sink of
attribution only.

**Edge attribution scores** (edge attribution patching, Syed, Rager &
Conmy 2023, "Attribution patching outperforms automated circuit discovery"):
for an edge u→v, score(u→v) = Σ_batch,positions Δin_v(u) · ∂m/∂in_v, where
Δin_v(u) is the change in v's input caused by replacing u's output with its
**mean over the measure batch** (mean ablation is the corruption everywhere in
E4), m is the task metric (below), and ∂m/∂in_v is the gradient at v's own
input copy (each head reads its own copy of the residual stream so gradients
are head-specific; each MLP neuron's input is its pre-activation, so
Δin_v(u) = w_vu·Δout_u). Edges exist from every node to every downstream
node that reads it (MLP: consecutive layers; transformer: all upstream nodes
via the residual stream). The attribution graph has weight |score|.

**Metric m.** Mean over the batch (and output slots) of the log-probability
of the correct class. m_clean = full model; m_corr = all nodes mean-ablated.

(a) **Modularity Q.** For a density d ∈ {1, 2, 5, 10, 20, 50, 100} % keep
the top-d % of edges by |score|, symmetrise, and run Leiden
(`leidenalg.ModularityVertexPartition`, weighted, 10 restarts, best Q, seed
fixed). Report the full sweep `Q_d`. **Pre-registered summary: Q₁₀** (d = 10 %).
Secondary: Q_AUC = mean over the sweep. Null-corrected Q₁₀ᵉˣ = Q₁₀ − mean
Q₁₀ of 20 graphs with the edge weights shuffled among the *same* edge slots
(preserves the layered skeleton, destroys the placement of strong edges).
**[JC]** d = 10 % rather than an absolute threshold because absolute scores
are not comparable across tasks/architectures.

(b) **Treewidth bound** tw₁₀ = `networkx.algorithms.approximation.treewidth_min_fill_in`
on the (unweighted, symmetrised) top-10 % graph; reported raw and as tw₁₀/N.
Also on the sweep. Min-fill is an upper bound; exact treewidth is NP-hard
(Adolfi et al. use treewidth-type parameters in their FPT discussion).

(c) **Effective sparsity.** Node importance s_i = mean |output| of node i
over the measure batch (head: L2 norm of its output vector). Participation
ratio PR = (Σs)²/Σs²; `sparsity = 1 − PR/N`. Secondary: weight sparsity =
fraction of |w| < 1e-3·max|w| per matrix, averaged.

## Fixed discovery pipeline

Node-level circuits (a circuit = a set of kept nodes; all other nodes
mean-ablated with the measure-batch means; the same corruption as the graph).

**Faithfulness** F(C) = (m(C) − m_corr)/(m_clean − m_corr) on the discovery
batch, clipped to [0, 1.05] for reporting. Target F ≥ 0.9.

Pipeline (identical for every model; seed s ∈ {0,1,2} fixes the discovery
batch and the tie-breaking jitter):
1. Compute node attribution scores (node-level attribution patching, mean
   corruption) on the discovery batch.
2. Order nodes by |score| ascending. Single greedy pass: tentatively remove
   the node, evaluate F by an actual ablated forward pass, keep it removed if
   F ≥ 0.9, else restore. Re-compute the attribution ranking of the remaining
   nodes after every 16 accepted removals.
3. Second identical pass over the survivors (catches nodes whose
   redundancy partner was removed in pass 1).
4. Output: circuit C, F(C), |C|, and **cost = number of ablated forward
   passes** (the primary cost — GPU contention makes wall-clock unreliable;
   wall-clock is recorded as secondary).

**Success** (primary, continuous): S = 1 − |C|/N, the compression achieved
at F ≥ 0.9, averaged over the 3 seeds. **Stability**: mean pairwise Jaccard of
the three circuits. Binary success (secondary): S ≥ 0.75 and Jaccard ≥ 0.5.
Guard: a model with held-out accuracy < 2× chance (`learned = False`) has
m_clean − m_corr ≈ 0 and F is meaningless; S is set to 0 and stability NaN.
**The un-guarded S is also recorded** (`S_raw`) so the kill control can show
what the pipeline would have said.

## Kill attempts (all pre-registered)

1. **Partial correlation**: H-E4c above (Q | sparsity, N).
2. **Label-permutation null** on ρ(Q, S) and on ΔBIC (2000 perms).
3. **Untrained high-Q control**: does `untrained_block` "succeed" on S_raw?
   If it does, S is only meaningful under the learned-guard, and Q is not
   sufficient for success — stated as a limit on the conjecture.
4. **Sparsity-vs-Q horse race**: rank Spearman |ρ| of every structural
   measure (Q₁₀, Q₁₀ᵉˣ, Q_AUC, tw₁₀/N, sparsity, weight sparsity, N) against
   S and cost; the strongest predictor is reported whatever it is.
5. **Threshold robustness**: repeat H-E4 (i) at d = 5 % and 20 %. A knee that
   only exists at d = 10 % is not reported as a knee.
6. **Within-architecture**: ρ(Q, S) inside each architecture separately;
   if the pooled relation is driven by the arch-level offset (transformers
   vs MLP), say so.

## Deliverables

`results/e4_models.csv` (one row per model, all measures), `results/e4_sweep.csv`
(Q_d, tw_d per model per density), `results/e4_tests.csv` (every hypothesis
test with CIs), 300-dpi figures: `fig_phase_Q_tw.png` (S as colour over
(Q₁₀, tw₁₀/N)), `fig_phase_Q_sparsity.png`, `fig_knee.png` (S vs Q₁₀ with
linear + segmented fits and bootstrap knee band), `fig_cost.png` (cost vs
tw₁₀/N and Q₁₀), `fig_sweep.png` (Q_d sweep by arm). `FINDINGS.md`. Reusable
core in `e4lib/` (`modularity_q`, `treewidth_bound`, `sparsity`,
`discover_circuit`, `legibility_profile(model, task_batch)`), tests in
`tests/` (planted modular vs non-modular graphs order correctly; treewidth
bound on a known graph; the pipeline recovers a planted circuit).

## Conjecture protocol

Only if H-E4 passes *and* H-E4c passes *and* kill 3 does not show trivial
success is the FPT conjecture (modularity as the parameter) written into
FINDINGS, flagged "for JVHM". Otherwise FINDINGS says the data are not clean
enough to state it and why.

## Amendments

(none yet)

1. (2026-08-30, after a 4-cell smoke run — `results/run_smoke.log` retained —
   and before the pre-registered run) **`modadd` is trained and evaluated on
   its full domain (all 529 pairs)**, as E1 trained on the full truth table.
   The 70 % split needs grokking-length training: tf2 reached 0.107 held-out
   accuracy in the 4000-step budget while the full-domain model reaches
   ≥ 0.99 in ≤ 600 steps for all three architectures. Consequence made
   explicit: `modadd` and `randlabel` now differ *only* in label structure
   (same inputs, same regime), which makes `randlabel` a cleaner control.
   `sort`/`copy` keep the 70/30 held-out split.
2. (same time) **Learned-guard threshold raised from 2× chance to accuracy
   ≥ 0.5.** The block-masked untrained tf2 scored 0.088 vs the 0.087 bar by
   chance on 159 eval inputs — the pre-registered bar was too loose to do
   its job. 0.5 is far above chance (≤ 0.125) for every task and far below
   every trained model's accuracy in the smoke run. `S_raw` is still
   recorded un-guarded, so kill attempt 3 is unaffected.
