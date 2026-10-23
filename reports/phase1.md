# Phase 1 report: estimators and planted-answer validation

WhiteBox E&E. Started and completed 2026-10-23. **Gate: met for every estimator Phase 2 uses; formation (K14) is blocked; the unsupervised proposers are not gated** (§5).

Planted systems are estimator tests with known answers. **Nothing in this report is evidence for a claim in the paper** (master prompt rule 0.1.1).

## 1. What was built

| Component | Files | Notes |
|---|---|---|
| System protocol, residual MLP systems, realiser patches (norm-matched option) | `whitebox_ee/systems.py` | Planted nets and small trained nets |
| HF causal-LM system (sites = layer × token role, hook patching) | `whitebox_ee/llm.py` | Tested (`tests/test_llm.py`) |
| Probe set, nodes, members, patch runs | `whitebox_ee/probe.py` | |
| Proposers: probe (ridge, block-write), sparse dictionary, ISA, pullback | `whitebox_ee/propose.py` | Only the probe proposer is gated |
| Condition 1 (quotient) and decodability | `whitebox_ee/filter.py` | |
| Condition 2 (direct-effect parents, sufficiency, stability, observational minimal sets) | `whitebox_ee/parents.py` | |
| Graph, merging, F_G counterfactual model, Condition 3 (IIA) | `whitebox_ee/graph.py` | |
| Discovery fixed point | `whitebox_ee/discover.py` | |
| Activity, active derivation, generality | `whitebox_ee/generality.py` | |
| Depth (λ, μ, γ), memorization verdict | `whitebox_ee/depth.py` | |
| Moves classifier | `whitebox_ee/moves.py` | |
| Creation protocol | `whitebox_ee/create.py` | |
| Formation | `whitebox_ee/formation.py` | |
| Baselines (softmax, entropy, margin, CCGP and per-input CCGP, Hutchinson sharpness) | `whitebox_ee/baselines.py` | |
| Statistics (bootstrap, Holm, AUROC with grouped CIs) | `whitebox_ee/stats.py` | |
| Planted systems V1–V5, formation sequence | `validation/planted.py` | |
| Harness and summariser | `validation/run_phase1.py`, `validation/summarise_phase1.py` | |
| Tests: unit, property (hypothesis), regression | `tests/` | 41 pass |

## 2. Design findings on planted systems (development seeds 0–9 only)

Each was found on a planted answer, fixed before the validation seeds, and written into `theory/core.tex` §5 (M1–M6) and `prereg/PHASE_1.md`.

1. **Observational parents cannot represent rule-plus-exceptions mechanisms (M1).** In V2, minimal sufficiency makes the output a lookup on the inputs, the sum node is never on any derivation, and every input reads as memorized. Parents are now direct interventional effects (others held at clean values) against a within-fibre-swap null, still required to be ε_D-sufficient.
2. **Derivations need edge activity (M2), normalised by available variation.** A threshold-crossing edge (s_0 → c_0) changes its child on only ~50% of partner swaps by construction; normalising by the fraction of partners whose child value differs gives such edges full activity while keeping rare exception routes inactive.
3. **"Each input its own leaf" needs the training range (M3).** An exception node whose value on x is shared by fewer than n_min training inputs is a leaf on x.
4. **Realiser estimation is the main source of spurious structure (M4).** Probes decoded from the full state find approximate readouts of variables before they are computed and readouts of upstream variables' functions; in residual streams the canonical realiser is decoded from the block's write; coarsenings and (separable) composites are not separate nodes; merged members are refitted to the node's partition and only the primary site is patched.
5. **The causal-use test is circular on an incomplete graph (risk R5 of Phase 0).** Resolved by the fixed point of M5 (derivation failures first and earliest-first, suspension and reactivation, causal use on effective pairs until stable, the full test last).
6. **Random-realiser nulls must be norm-matched (M6)**, and hand-built networks must be error-correcting (indicator margins), otherwise random perturbations of a concept-swap's norm change 80% of outputs and every realiser looks inert.
7. **Calibrations:** κ_max 0.25 → 0.6; Condition 1(i) per behaviour.
8. **Unsupervised proposers are not yet usable.** Top-k SAEs and ICA/ISA do not recover factorial categorical codes on V4 (centring, non-identifiability with few distinct points, discrete-source ICA). They are not gated; the blind Vector-HaSH anchor (Phase 6) and K12 (unsupervised principles) wait on them.
9. **Limitations found:** circularly encoded variables (V1 Fourier) defeat linear-decoder realisers; recovery needs N ≳ 1000 probe inputs on V1 (none at N ≤ 500).

## 3. Propositions 2.7.1–2.7.3 (proofs in `proofs/`, unchecked by a second reader)

- **2.7.1 (CCGP is the depth-one case).** False as stated: perfect depth-one grounding with CCGP = 1/4, and CCGP = 1 for a causally inert variable (P7, explicit constructions). True in the aligned form: with the decoder fixed to the system's readout and causal use added, readout-aligned CCGP = 1 iff the depth-one candidate passes exact quotient and causal-use tests. Prediction carried to K2: the CCGP–depth disagreement analysis.
- **2.7.2 (grounded ⇒ correct on lower inputs).** Proved under (A1) grounding on the input, (A2) value locality below the cut, (A3) lower for the cut; correctness additionally needs the target to factor through the principles (shortcut failure mode) (P8).
- **2.7.3 (memorized ⇒ no sideways generalization).** False for memorization at τ (private compositional node in a large suite); true for leaf-level memorization: every new input is diagonal (P9).

## 4. Validation results (seeds 100–104, frozen at tag `prereg-phase-1`, commit bc6036a)

All cells ran on the laptop CPU at the tagged code. One later commit is performance-only: patch runs return only the states they need, results are identical, and the regression tests pass. Raw results are in `runs/phase1/val/*.json`. The verdicts below are computed mechanically by `validation/verdicts_phase1.py`.

| Cell | Criterion | Result | Verdict |
|---|---|---|---|
| P1a V1-onehot | node precision & recall >= 0.9 on 5/5 | P 1.00-1.00, R 1.00-1.00 | PASS |
| P1b V1-onehot | edge F1 >= 0.9 on 5/5 | 1.00-1.00 | PASS |
| P1c V1-onehot | decoy represented-not-used 5/5 | 5/5 | PASS |
| P1e V1-onehot | generality MAE vs oracle <= 0.05 on 5/5 | 0.005 (max) | PASS |
| P1a V1-noise05 | node precision & recall >= 0.9 on 5/5 | P 1.00-1.00, R 1.00-1.00 | PASS |
| P1b V1-noise05 | edge F1 >= 0.9 on 5/5 | 1.00-1.00 | PASS |
| P1c V1-noise05 | decoy represented-not-used 5/5 | 5/5 | PASS |
| P1e V1-noise05 | generality MAE vs oracle <= 0.05 on 5/5 | 0.007 (max) | PASS |
| P1d V1 | wrong hypotheses accepted <= 10% (pooled) | 0/50 | PASS |
| P1f NULL | FP <= 0.02 every seed; mean accepted <= 1 | max FP 0.000, mean accepted 0.00 (20 cells) | PASS |
| P2 V2 | AUROC(-mu) >= 0.9 on 5/5 and lower CI > 0.8 on 4/5 | pt 1.000-1.000, lo 1.000-1.000 | PASS |
| P3a V3-comp | move accuracy >= 0.9 (withheld and seen) 5/5 | 1.00 | PASS |
| P3b V3-comp | predicted-correctness accuracy >= 0.9 5/5 | 1.00 | PASS |
| P3a V3-lookup | move accuracy >= 0.9 (withheld and seen) 5/5 | 1.00 | PASS |
| P3b V3-lookup | predicted-correctness accuracy >= 0.9 5/5 | 1.00 | PASS |
| P4 V4 | AUROC >= 0.95 and scaffold recall >= 0.9, 5/5 | AUROC 1.000, recall 1.00 | PASS |
| P5a V5-valid | exactly one valid node N <- {A,B}, 4/5 | 5/5 | PASS |
| P5b V5-decoy | zero valid nodes, 4/5 | 5/5 | PASS |
| P5b V5-leaf | zero valid nodes and N 'new leaf (used, not derived)', 4/5 | 5/5 | PASS |
| P5b V5-relabel | zero valid nodes, 4/5 | 5/5 | PASS |
| P5b V5-generic | zero valid nodes and N 'exposure-generic', 4/5 | 5/5 | PASS |
| P6a FORM | grounding event 4+-1 and sum principle 3+-1, 4/5 | 5/5 | PASS |
| P6b FORM | |Lambda_t - (1 - memo_t)| <= 0.2 every checkpoint, 4/5 | 0/5 (max deviations 0.58, 0.74, 0.88, 0.50, 0.83) | FAIL |

**Reported, no verdict:**
- **V1 noise10** (stress): exact recovery on 5/5 seeds.
- **V1 fourier** (stress): node recall 0.20–0.40 and edge F1 0.05–0.25. Circular codes defeat linear-decoder realisers (limitation).
- **SENS** (V1 one-hot): nothing recovered at N = 250 and N = 500 on any seed; exact at N = 1000 on 5/5. Recovery has a sample-size floor of about 1000 probe inputs on V1, set by held-out sufficiency over ~100-tuple parent tables. Phase 2 uses N = 1500.
- **Oracle graphs:** the graph built with oracle realisers equals the planted spec on every V1 seed (edge F1 1.00).
- **P6b, FORM:** in the fully memorized checkpoints, Λ_0.25 is 0.01–0.88 where the planted answer is 0; at checkpoint 3 (80% memorized) it is 0.62–0.71 where the planted answer is 0.2. This is the answer-funnel failure declared in the prereg before the run (§5).

## 5. Gate (self-gate by the letter of prereg/PHASE_1.md §4)

| Estimator | Criteria | Verdict | May be used on real models |
|---|---|---|---|
| Probe proposer, quotient, parents, causal use, graph and merging | P1a–P1f | PASS | Yes |
| Generality | P1e, P4 | PASS | Yes |
| Depth and memorization verdict | P2 PASS; P6b FAIL (implicates the verdict in the answer-funnel regime) | PASS for P2; boundary stated | Yes, for depth (λ, μ, γ) in K2. Every memorization verdict on a real model must carry the answer-funnel diagnostic, and K4/K9 memorization claims are limited accordingly. |
| Moves classifier | P3a, P3b | PASS | Yes |
| Creation protocol | P5a, P5b | PASS | Yes |
| Formation | P6a PASS, P6b FAIL | **FAIL** | **No** (K14 blocked until fixed and re-validated on seeds 105–109) |
| Unsupervised proposers (SAE, ISA, pullback) | not gated | not validated | **No** (blind Vector-HaSH anchor and K12 unsupervised arm blocked) |

**Phase 2 dependency.** Phase 2 needs P1a–P1f, P2 and P3, and all of them pass, so Phase 2 may proceed under its own pre-registration.

**What the answer-funnel failure means.** A structural verdict built on reuse cannot tell a looked-up answer from a computed one when both pass through a shared value node. Inside a hand-built network the rule and the lookup are both tables from (a, b); only the table's content differs, and generality does not see content. This is a property of the definition, not a bug in the estimator. For K2 it means depth can be inflated on inputs whose answer is retrieved rather than computed. Phase 2 therefore pre-registers a diagnostic for it.

## 6. Compute

- Planted work ran on the laptop CPU only: about 15 CPU core-hours over development and validation, and 0 GPU-hours.
- An LLM development run (Qwen2.5-1.5B, development model) used about 1 laptop GPU-hour.
- Pinnacle: two env-build jobs (≈ 1.5 core-h).
- Details are in `reports/compute_ledger.md`. This is far inside the Phase 1 envelope (200 A100-h, 2,000 CPU core-h).

## 7. What passed, what failed, what was cut, what is next

- **Passed:** the probe-proposer pipeline (quotient, parents, causal use, merging), generality, depth and the memorization verdict (within the answer-funnel boundary), moves, and creation. All of these passed on 5/5 validation seeds, with zero false positives on the untrained and shuffled nulls.
- **Failed:** formation (P6b), from the answer-funnel limitation of the structural memorization verdict.
- **Not validated:** the unsupervised proposers.
- **Cut or replaced** (`docs/cut.md`): observational minimal-sufficiency parents, replaced by M1.
- **Next:** Phase 2 (K1, K2). Freeze `prereg/PHASE_2.md` with the answer-funnel diagnostic, tag it, run the cells on Pinnacle when access returns (laptop for the models that fit), and self-gate on K2.

## 8. Amendments 1 and 2 (re-validation on new seeds)

The §5 gate above stands for the original estimator. When the realisers were made accurate enough for LLMs (amendment 1, principal-subspace realisers), the estimator changed and had to be re-validated. Every result stays in `docs/record.md`.

**Amendment 1** (tag `prereg-phase-1-amend-1`, seeds 105–109): **P2 FAILED** (V2 seed 105: AUROC(−μ) 0.778, lower CI 0.668). With accurate realisers the override node of V2 is legitimately reused, so memorized inputs were protected only by the value-level leaf rule, which fails under the answer funnel.

**Amendment 2** (tag `prereg-phase-1-amend-2`, commit 95a65da, seeds 110–114). It adds the M9 step-level leaf rule and dual-mode (delta and plain) realisers, removes the coarsening rule, and gates on V2D/FORMD. Results so far (`runs/phase1/val_a2/VERDICTS_partial.md`):

| Criterion | Result | Verdict |
|---|---|---|
| P1a–P1e (V1 one-hot, noise 0.5) | P = R = 1.00, edge F1 1.00, decoys 5/5, generality MAE ≤ 0.009, wrong hypotheses 0/50 | PASS |
| **P1f** (untrained and shuffled nulls, V1 and V4) | max FP **0.025** (seed 114, untrained V4: 4/160), mean accepted 0.80 | **FAIL** |
| P3a, P3b (V3 compositional and lookup) | 1.00 | PASS |
| P4 (V4 scaffold) | AUROC 1.000, recall 1.00 | PASS |
| P5a, P5b (V5) | 5/5 each | PASS |
| P2 (V2D), P6a/P6b (FORMD) | long cells running on comp06 (jobs 1364993/1364994) | PENDING (reported when done; cannot change the gate below) |

**Gate (by the letter of prereg/PHASE_1.md §4).** The amended discovery estimator fails P1f. It may not be used on a real model. **The Phase 2 dependency (P1a–P1f, P2, P3) is not met**, so Phase 2 does not run under `prereg-phase-2`. The Phase 2 outputs produced before this verdict are discarded unanalysed (`docs/record.md`). The original, validated estimator (§5) is the one with too-weak realisers for LLMs (development finding, amendment 2 "Change 3"), so it is not a working fallback for K1/K2. The next step is a new development cycle (amendment 3, new seeds, tagged before its run).

**Post-hoc diagnosis (a hypothesis for that cycle, not tested).** On every validation seed, the untrained copy of V4 yields accepted nodes g0, g1, g2 (pos mod 3, 4, 5). These are coarsenings of a single input. In an untrained copy any partition of pos can be read linearly from a random embedding of pos. Amendment 2 removed the coarsening rule (M4 ii), which rejected such readouts before; the original and amendment-1 validations accepted 0 nodes on this null. On seeds 110–113 the three nodes give FP 0.019, just under the threshold; on seed 114 one more, unlabelled node tips it to 0.025. The FP rate is per candidate, so this margin depends on the candidate count; the next cycle should fix the cause, not the margin. The amendment-2 development run (dev_a3, job 1364799, final code) already showed 3 and 4 accepted nodes on this null for development seeds 0 and 1. Those files were not inspected before the tag, and this is recorded as a process failure: from now on, every development null is checked before a tag.
