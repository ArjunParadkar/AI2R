# Block D — Attribution (anchor only) — pre-registration

_Naming: every run in this document is a horse from the Tracen roster (`experiments/HORSES.md`); the handbook codes below are the pre-registration labels and appear here only as such._

Status: SPEC written 2026-09-07 before any run. Novelty: **TAKEN** — Implementation Invariance
axiom (Sundararajan, Taly & Yan 2017); axiom satisfaction per method tabulated in the July-2026
survey. Path methods pass; LRP/DeepLIFT and neuron-level methods fail. No claim; D1 demonstrates
that the auditor reproduces the known result.

## D1 — Known-answer anchor  [LAPTOP (CPU suffices for the MLP specimens); GPT-2 variant [5070]]
Specimens: 5 seeds × every (function, depth, width) cell of Block F (760 nets), on the truth table
and on a 256-point probe cloud. Methods: integrated gradients (input level, 64-step Riemann
midpoint, zero baseline), gradient×input (input level), and neuron-level attributions (per hidden
unit a_u ∂logit/∂a_u, and per-unit mean ablation effect). Through `check_invariance` (100 perm /
100 scale / 100 combined).
**Pre-registered expectations:** IG and gradient×input at input level → `invariant` (CV < 0.01,
float floor); neuron-level attribution *values* → `artifact` under permutation (they are indexed by
unit) and — as E2 measured — their *sorted multiset* invariant under scaling (a_u ∂/∂a_u is
scale-free). Report per-method verdict table `results/D1_audit.csv` and the fraction of specimens
agreeing. GPT-2 variant: IG on token embeddings for the next-token loss vs MLP-neuron
attributions (F11 already carries the latter); n = 20 renamings.
