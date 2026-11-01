# Block Z — Topology beyond persistent homology (pre-registration)

_Naming: every run in this document is a horse from the Tracen roster (`experiments/HORSES.md`); the handbook codes below are the pre-registration labels and appear here only as such._

Status: SPEC written 2026-09-07 before any Block-Z run. Each item carries its own novelty flag.

## Z1 — Zigzag persistence across training  [5070]  FLAGGED (search "zigzag persistence neural network training")
On V1 checkpoints (20 per model; 40 models: every arm's first architecture on modular addition
plus the 6 random-label): activation clouds of the penultimate layer on a fixed 512-example probe,
PCA-20 fitted on the final checkpoint, **subsampled to 150 points** (S5: dense clouds blow up the
union complex), Rips scale = median pairwise distance of the final cloud. `topology.zigzag`
(union-of-consecutive). Compare to the per-checkpoint film (v1 diagrams + greedy bar matching by
birth/death proximity): count the film's births+deaths that the zigzag joins into one bar
("mislabelled" events) and the zigzag bars the film splits. Pre-registered: mislabelled fraction >
0.2 means the film is not a valid tracker (report either way). Random-label models as null.

## Z2 — Multiparameter persistence  [5070]  FLAGGED
Scale × training-fraction bifiltration on 20 V1 models (fraction = checkpoint index; each point
exists from the checkpoint at which its activation pattern stabilises — defined as the first
checkpoint after which the point's PCA coordinates move < 10 % of the cloud diameter);
`topology.hilbert_function` on a 20 × 20 grid; summary = Hilbert mass H1, argmax location. Test:
Spearman correlation of the 2-parameter summary with V3 discovery cost vs the 1-parameter total
persistence at the final checkpoint; **pre-registered: |ρ_2p| − |ρ_1p| > 0.1 with bootstrap CI > 0**.

## Z3 — Graph PH of attribution graphs  [5070]  FLAGGED
`topology.graph_ph` on the V2 graphs (density 0.1): β1 finite / essential, total persistence; added
as features to the V4 logistic; pre-registered: ΔBIC > 10 for adding β1 over (Q_excess, tw, sparsity).

## Z4 — Quotient-space connectivity beyond H0  [LAPTOP]  OPEN
On F specimens: for every pooled cell, pairs of explanations in the same orbit (there are none
unless K < N — if F4 finds K = N everywhere, Z4 reports "no within-orbit pairs exist; the
question is vacuous" and instead tests **cross-orbit** path connectivity: linear interpolation in
parameter space between two specimens' networks (after Hungarian unit matching) with the
explanation's validity checked at 50 points; fraction of pairs with a valid path (validity ≥ 0.9
of the path)). Report π0 of the valid set vs H0 of the quotient.

## Z5 — Persistence under scaling: the covariance law  [LAPTOP]
From F10: per statistic, the fitted exponent k under uniform scaling (R² ≥ 0.99 expected for TC:
k = 2; H1 bar count: k = 0; persistent entropy: k = 0) and the per-unit-scaling CV (no single
exponent). Proposal: `topology.normalized_persistence` (per-unit standardised cloud; E2's exact
repair); verified via `check_invariance` on every F10 specimen (`ph_std_probe` column) —
pre-registered: invariant on ≥ 99 % of specimens.
