# Pre-registration, Phase 1: planted-answer validation of the estimators

WhiteBox E&E. Written 2026-10-23, before any validation seed was run. It is frozen by the annotated tag `prereg-phase-1`; the commit hash and the SHA-256 of this file are recorded in `docs/record.md`. Thresholds and criteria below never change after the tag (master prompt rule 0.1.2).

**What this is and is not.** Phase 1 tests estimators on planted systems with known answers (master prompt §5). These are estimator tests only. They are **never evidence for a claim in the paper** (rule 0.1.1). The only consequence of a result here is whether an estimator may be used on a real model.

## 1. Estimator definition under test

The estimators are those in `whitebox_ee/` at the tagged commit. This includes the Phase 1 amendments M1–M8 (`theory/core.pdf` §5):
- direct-effect parents;
- edge activity;
- the leaf rule;
- canonical realisers;
- the discovery fixed point;
- norm-matched nulls, with κ_max calibrated to 0.6;
- output realisers (M7): a candidate equivalent to a behaviour's output partition is that output's realiser;
- interventional identification of derivation maps (M8).

Thresholds are the `Thresholds()` defaults in `whitebox_ee/filter.py`:
- ε_Q 0.10
- κ_max 0.6
- h_min 0.5 bit
- ε_D 0.05
- k_max 8
- J_min 0.8 (reported)
- τ_IIA 0.90
- Δ_IIA 0.20
- ε_merge 0.05
- α 0.5
- τ ∈ {0.25, 0.5}
- ε_M 0.10
- n_min 5
- θ_dir 0.05

Candidates come from the **probe proposer**: ridge-decoder realisers of labelled variables, decoded from block writes. A realiser is proposed only if it decodes its label with 2-fold held-out nearest-centroid accuracy ≥ 0.9. Its label set is:
- every planted variable, including planted non-concepts and decoys;
- 5 wrong hypotheses per system: sums of two random planted variables modulo the smaller cardinality.

**Gated estimators.** These are the estimators Phase 2 needs and Phases 3–7 use:
- probe proposer;
- quotient filter (Condition 1);
- parents (Condition 2);
- causal-use filter (Condition 3);
- graph and merging;
- generality;
- depth and memorization verdict;
- moves classifier;
- creation protocol;
- formation.

**Not gated in Phase 1 (development, reported only):** the unsupervised proposers (sparse dictionary, ISA, pullback). They failed development checks: they do not recover factorial categorical codes on V4. They may not be used on any real model until a later amendment pre-registers and passes their own planted tests.

## 2. Systems, seeds, sample sizes

- **Validation seeds:** 100, 101, 102, 103, 104. They are disjoint from the development seeds 0–9 used to build the estimators.
- **Unit of analysis:** the planted network instance (seed). Each seed draws a fresh random rotation Q, encodings, distractors, memorized subsets, withheld sets and tables.

| ID | System | N (probe set) | Variants |
|---|---|---|---|
| V1 | Hand-built 3-digit addition: digits → column sums → carries → result digits; decoy written but never read | 2000 | onehot (gated), noise05 (gated), noise10 (stress, reported), fourier column sums (stress, reported) |
| V2 | Rule (a+b mod 11 via a sum node) plus a planted memorized subset (20% of pairs, one detector per pair, override chain) | 2000 | — |
| V3 | Sideways: C=(A+B) mod 5 with 5 withheld (A,B) combinations | 1500 | compositional (table correct on withheld) and lookup (wrong on withheld) |
| V4 | Vector-HaSH-style: position → 3 grid-module phases → path-integrated phases → conjunctive place units storing 6 content items. Every tested variable (items, path-integrated phases, conjunctive code) is read through two coarsened behaviours (bits), so none is equivalent to an output (M7) | 2000 | — |
| V5 | Creation: before network (A, B) and after network. A, B and the new problem's yN are each read through two coarsened behaviours (mod 2 and ≥ 3) | 1500 | valid (new N=(A+B) mod 5 read by the new behaviour), decoy (N written, never read), leaf (new behaviour via a node from raw inputs), relabel (N an invertible relabelling of A), generic (control exposure also contains N) |
| FORM | Formation: 7 "checkpoints" of V2 (p=7) with memorized fraction 1, 1, 1, 0.8, 0.1, 0.1, 0.1; labels exclude the per-pair detector variables m_j (single-input fibres that cannot pass h_min) | 1500 | — |
| NULL | V1 and V4 with (i) the untrained copy (same shapes, random weights) and (ii) shuffled labels (each label permuted across inputs). Nulls run without the decodability pre-filter, so that the filters themselves are tested | 2000 | — |
| SENS | V1 onehot at N ∈ {250, 500, 1000} | — | descriptive |

## 3. Pass criteria (frozen)

A gated estimator passes Phase 1 only if every criterion that involves it passes. "k/5 seeds" means the criterion holds on at least k of the 5 validation seeds. Nodes are matched to planted variables by normalised VI ≤ 0.1.

| Cell | Criterion | Threshold | Estimators involved |
|---|---|---|---|
| P1a | V1 onehot and noise05: node precision and recall against the planted concepts not read directly by a behaviour, {s_0, s_1, s_2, c_0, c_1} (under M7, r_0–r_2 and c_2 are output realisers) | both ≥ 0.9, on 5/5 seeds per variant | proposer, quotient, parents, causal use, graph |
| P1b | V1 onehot and noise05: edge F1 against the planted parent sets, with each output's parents being the parents of the variable it reads (M7) | ≥ 0.9, on 5/5 seeds | parents, graph |
| P1c | V1: decoy classified "represented, not used" at its site (all variants) | 5/5 seeds per gated variant | quotient |
| P1d | V1: wrong hypotheses accepted as nodes (pooled over seeds and gated variants) | ≤ 10% of hypothesis × seed cells | quotient, causal use |
| P1e | V1: mean absolute difference between estimated generality and generality computed with the oracle (true) realisers, over matched nodes (Spearman ρ reported; with 5 nodes and near-tied column sums it is dominated by tie-breaking) | ≤ 0.05, on 5/5 seeds per gated variant | generality |
| P1f | NULL: untrained copy and shuffled labels, V1 and V4: accepted concept nodes / candidates | ≤ 0.02 on every seed, and mean accepted nodes ≤ 1 | quotient, causal use |
| P2 | V2: AUROC of −μ (τ = 0.25) for planted-memorized vs rule inputs, CI resampling distinct input pairs | point estimate ≥ 0.9 on 5/5 seeds, and lower 95% CI > 0.8 on 4/5 | generality, depth, memorization verdict |
| P3a | V3: move classification. Withheld inputs → sideways; seen inputs → lower | accuracy ≥ 0.9 for each, on 5/5 seeds per variant | moves |
| P3b | V3: predicted correctness of sideways inputs (interchange-assembled value vs reference derivation) against actual correctness | accuracy ≥ 0.9, on 5/5 seeds per variant | moves |
| P4 | V4: AUROC of generality, scaffold (6 phase nodes) vs content and conjunctive (7 nodes); and scaffold recall | AUROC ≥ 0.95 and recall ≥ 0.9, on 5/5 seeds | generality (the blind version waits for the unsupervised proposer) |
| P5a | V5 valid: exactly one valid new node, labelled N, with parents labelled {A, B} | 4/5 seeds | creation |
| P5b | V5 decoy, leaf, relabel: zero valid new nodes. Leaf: N reported as "new leaf"; generic: N reported "exposure-generic" | 4/5 seeds per variant | creation |
| P6a | FORM: grounding event (largest rise of Λ_0.25) at checkpoint 4 ± 1, and the sum node first a principle at checkpoint 3 ± 1 | 4/5 seeds | formation, depth |
| P6b | FORM: Λ_0.25 tracks the planted grounded fraction, \|Λ_t − (1 − memorized fraction_t)\| ≤ 0.2 at every checkpoint | 4/5 seeds | formation, depth (memorization verdict) |

**Reported only (no verdict):**
- V1 noise10 and fourier stress variants;
- SENS (recovery against N);
- runtime and forward rows;
- parent stability (Jaccard);
- the oracle-realiser graph's agreement with the planted spec.

## 4. Consequences

- **Pass.** An estimator that passes may be used on real models in later phases, within the pre-registration of that phase.
- **Fail.** An estimator that fails may not be used on a real model. A fix is a new development cycle. Its re-validation uses new seeds 105–109 under a logged amendment `prereg/PHASE_1_amend_N.md`, tagged before the run. The failed result stays in `docs/record.md`.
- **Phase 2 dependency.** Phase 2 (K1, K2) requires P1a–P1f, P2 and P3 to pass.
- **Later dependencies.** P4–P6 gate Phases 3, 6 and 7. P6 gates formation (K14) only.
- **Known risk, declared before the run (development seed 1).** In the fully memorized checkpoints of FORM, the override chain computes partial answers. These are nodes whose value is the stored label, shared by more than n_min memorized inputs. They are in range and widely reused, so they ground memorized inputs, and Λ is about 0.8 where the planted answer is 0. P6b is expected to fail for this reason. If it does, the memorization verdict carries a stated validity boundary: it detects memorization only when memorized answers do not funnel through a shared answer-valued node that is not output-equivalent. Every later memorization claim (K4, K9) must then report the answer-funnel diagnostic. P2 tests the regime where the boundary does not bind.

## 5. Compute

All cells run on the laptop CPU, about 0.5–2 CPU core-hours in total. No GPU is used. This is well inside the approved Phase 1 envelope (200 A100-h, 2,000 CPU core-h).
