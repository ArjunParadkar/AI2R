# E4 — Legibility phase diagram: FINDINGS

**Status: COMPLETE (2026-08-30).** 87 models (SPEC said "planned 84"; the
grid arithmetic is 72 trained + 3 random-label + 9 untrained + 3
block-masked = 87), one run, no cell failed. Total compute 27 min wall
(training + profiling + 3 discovery seeds each) on a shared RTX 5070 Laptop.
Everything is in `results/` (`e4_models.csv`, `e4_sweep.csv`,
`e4_circuits.csv`, `e4_tests.csv`, `e4_by_arm.csv`, `e4_posthoc.txt`, five
300-dpi figures). Tests: `tests/test_e4.py` 7/7 green.

## Verdict

| Hypothesis | Pre-registered criterion | Result |
|---|---|---|
| **H-E4 (i)** direction: ρ(Q₁₀, S) > 0, CI excludes 0 | Spearman, 2000-resample bootstrap | **FAILS — the sign is wrong.** ρ = −0.37 [−0.55, −0.17], perm p = 0.0005 (all 87). Learned-only ρ = −0.37 [−0.58, −0.14]; trained-only ρ = −0.46 [−0.65, −0.25]. |
| **H-E4 (ii)** shape: knee beats line, ΔBIC ≥ 2, ≥ 50 % bootstrap mass > 0, perm p < 0.05 | segmented vs linear | **FAILS.** ΔBIC = −2.6 (line wins); bootstrap mass above 0 = 44 %; perm p = 0.06; knee CI [0.16, 0.46] spans the whole transformer range; slope change b₂ CI [−20, 31]. At d = 5 % ΔBIC = −6.0, at d = 20 % −3.5 (kill 5). |
| **H-E4b** treewidth/N ↔ S negative; treewidth/N ↔ cost positive | Spearman CIs | **FAILS both.** ρ(tw₁₀/N, S) = −0.11 [−0.35, 0.13]; ρ(tw₁₀/N, cost/N) = +0.17 [−0.03, 0.36]. Raw ρ(tw₁₀, cost) = 0.67 is entirely model size: cost/N is 1.0–2.1 for every learned model by construction of a greedy pass (two degenerate-gap controls exit early) (see below). |
| **H-E4c** Q survives control for sparsity and N | partial Spearman | **FAILS.** partial ρ(Q₁₀, S | sparsity, N) = −0.22 [−0.43, 0.09] (all), +0.02 [−0.24, 0.31] (learned). |

**Modularity of the attribution graph does not predict circuit-discovery
success in this ensemble, and the pooled relation that exists is negative
and is an architecture offset.** The FPT conjecture is **not stated** (SPEC
"Conjecture protocol": all three gates failed). Nothing here is for JVHM.

## What the data actually show

1. **The pooled Q→S relation is an architecture confound.** MLPs sit at
   Q₁₀ ≈ 0.44 and S ≈ 0.19; tf2 at Q₁₀ ≈ 0.25, S ≈ 0.40; tf4 at Q₁₀ ≈ 0.19,
   S ≈ 0.36 (`fig_phase_Q_tw.png`: three disjoint clouds). Within
   architecture (kill 6) every ρ(Q₁₀, S) CI crosses 0: mlp +0.32 [−0.12,
   0.68], tf2 −0.38 [−0.70, 0.04], tf4 +0.21 [−0.20, 0.55]. Post-hoc
   (`e4_posthoc.txt`, not pre-registered): partial ρ(Q₁₀, S | arch dummies)
   = +0.03 [−0.26, 0.33]. Q's "effect" is that a 2-hidden-layer MLP's
   attribution graph is a single bipartite layer pair, which Leiden cuts
   into high-Q communities regardless of what was learned.
2. **Attribution graphs are *less* modular than their own shuffle null.**
   Q₁₀ − Q₁₀(shuffled weights on the same edge slots) is negative for 90 %
   of models (87 % of learned ones): the strong edges concentrate on a few
   hub nodes, which *lowers* modularity relative to spreading the same
   weights at random. Trained transformers are indistinguishable from
   untrained ones on the whole Q sweep (`fig_sweep.png`); only MLPs trained
   with the clusterability loss and the block-masked control exceed the
   null.
3. **Golechha et al.'s qualitative claim replicates; their mechanism does
   not run through Q.** Their clusterability loss (re-implemented from
   arXiv 2502.02470, Sec. 3; λ = 40 is their value) gives smaller faithful
   circuits in every architecture: tf2 |C| 86 → 37 nodes (S 0.37 → 0.73),
   tf4 174 → 98 (S 0.36 → 0.64), mlp 103 → 83 (S 0.19 → 0.35). But in the
   transformers the attribution-graph Q₁₀ does *not* move (0.25 → 0.22,
   0.18 → 0.16, and Q-excess falls); only the MLP's Q rises (0.38 → 0.70).
   So "clusterability loss ⇒ smaller circuits" holds, and "smaller circuits
   because the attribution graph became modular" does not. The loss acts on
   weights; the attribution graph measured on a 256-input batch sees the
   task-relevant hub structure, not the weight blocks. Stability (Jaccard)
   does not improve with the loss (tf2 0.53 → 0.47, tf4 0.53 → 0.39).
4. **Q predicts stability and achieved faithfulness in the pool (ρ(Q₁₀,
   Jaccard) = +0.43 [0.19, 0.64]; ρ(Q₁₀, F) = +0.39 [0.20, 0.57]) — and
   those too are the architecture offset**: post-hoc partial ρ | arch =
   −0.16 [−0.37, 0.14] and −0.21 [−0.44, 0.06]; within tf4 the sign flips
   (ρ(Q₁₀, Jaccard) = −0.51 [−0.71, −0.12]).
5. **Discovery cost is not a function of structure in this pipeline.** A
   greedy prune touches each node once per pass (two passes, early exit
   for the second), so cost/N ∈ [1.0, 2.1] for all 71 learned models; the only
   thing tw or Q could predict is the small variance from re-scoring
   frequency, and neither does (partial ρ(tw/N, cost/N | arch) = −0.13
   [−0.38, 0.10]). Adolfi et al.'s complexity results are about *optimal*
   circuits; a fixed-budget greedy heuristic has, by design, linear cost, so
   the parameterized-complexity question cannot be read off it. The
   quantity that *would* carry it — cost of an exact/branch-and-bound
   search to the minimum faithful circuit — was not in the pre-registered
   pipeline and is not computed here.
6. **Untrained models "succeed" trivially without the guard (kill 3
   confirmed).** S_raw = 0.93 (untrained) and 0.96 (block-masked, Q₁₀ 0.52)
   versus 0.38 for trained models: when m_clean − m_corr ≈ 0, the
   faithfulness ratio is met by almost any subset, so the pipeline reports
   a tiny "circuit". Compression at F ≥ 0.9 is only meaningful when the
   model computes something (guard: accuracy ≥ 0.5, amendment 2). This is
   a limit on any legibility score built from a faithfulness ratio.
7. **Binary success (S ≥ 0.75 and Jaccard ≥ 0.5) is reached by 1/87
   models** (tf2 / modadd / L1 1e-3). Minimal faithful circuits at F ≥ 0.9
   are large (median |C|/N = 0.63) and seed-unstable (median Jaccard 0.56;
   transformers 0.49). Non-identifiability (E1) shows up here as instability
   across discovery batches.
8. **Strongest predictors of the discovery outcome** (horse race, all
   models, `e4_tests.csv`): for S — N (+0.32, bigger models compress
   more), Q₁₀ (−0.37), sparsity (−0.28), all architecture-confounded; for
   Jaccard — N (−0.51), Q₁₀ (+0.43); for F — sparsity (+0.50), N (−0.40).
   Weakest: every treewidth measure (|ρ| ≤ 0.28 for anything but raw cost).
   Effective sparsity beats Q for F but not for S.

## Kill attempts (SPEC list)

1. Partial correlation Q | sparsity, N: relation gone (H-E4c row).
2. Label-permutation nulls: ρ(Q₁₀, S) survives (p = 0.0005) — but with the
   wrong sign, and the knee does not (p = 0.06, ΔBIC negative).
3. Untrained high-Q control: trivial success on S_raw (0.96). Guard needed.
4. Horse race: Q is not the strongest predictor of anything once N is in
   the race; N itself is strongest for S and Jaccard.
5. Threshold robustness: no knee at d = 5 % or 20 %; ρ(Q_d, S) negative at
   every density.
6. Within-architecture: no CI excludes 0; the pooled effect is the offset.

## Judgment calls (all in SPEC as [JC] or amendments)

* One seed per (arch, task, arm) cell; breadth over depth.
* `modadd` trained on its full domain (amendment 1; 70/30 split needs
  grokking-length training). `sort`/`copy` keep held-out evaluation.
* Learned guard raised to accuracy ≥ 0.5 (amendment 2; the pre-registered
  2× chance passed a block-masked untrained model by luck).
* Q summary at 10 % edge density (relative threshold); nodes exclude
  embed/unembed; the shuffle null keeps the layered skeleton.
* Faithfulness = gap-normalised mean correct-class log-prob, mean ablation
  everywhere; cost = ablated forward passes, not wall-clock (GPU shared).
* Four "trained" models did not learn (both L1 1e-3 on modadd, mlp clus40
  on modadd, mlp L1 1e-3 on copy) and are kept as flagged rows with S = 0.

## Relation to the literature (honest)

* Adolfi, Vilas & Wareham (ICLR 2025): they prove circuit discovery is
  NP-hard, fixed-parameter intractable in the parameters they examine, and
  inapproximable. E4 provides **no empirical support** for modularity of the
  attribution graph as a tractability parameter; its treewidth bound
  predicts neither success nor cost. This is one heuristic pipeline on
  87 small models — a negative for the specific conjecture, not a theorem
  in either direction.
* Golechha, Chaudhary, Velja, Abate & Schoots (arXiv 2502.02470): their
  "clusterability loss ⇒ smaller circuits" replicates in our ensemble with
  a different circuit-discovery method (node-level EAP + greedy prune at
  F ≥ 0.9). Our addition is that the effect is not mediated by the
  modularity of the attribution graph in transformers.
* Syed, Rager & Conmy (2023): edge/node attribution patching as used.

## What survives for Stage 2 (`whitebox.legibility`)

`e4lib.legibility_profile` is a working instrument (Q sweep, shuffle null,
treewidth bound, sparsity, discovery outcome) and is what E5 needs to place
biological circuits on the same axes. But there is **no calibrated
`legibility_score`** to ship: no structural measure predicts discovery
success within architecture, so a score would be predicting the
architecture label. Scope statement for the module: measures only, no
prediction, until an ensemble with within-architecture variation in
outcome exists (more seeds, wider regularisation range, an exact-search
cost metric).

Files: `results/e4_models.csv` (87 rows), `e4_sweep.csv` (7 densities ×
87), `e4_circuits.csv` (261 circuits), `e4_tests.csv`, `e4_by_arm.csv`,
`e4_posthoc.txt`, `fig_phase_Q_tw.png`, `fig_phase_Q_sparsity.png`,
`fig_knee.png`, `fig_cost.png`, `fig_sweep.png`, `run.log`.
