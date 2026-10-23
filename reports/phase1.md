# Phase 1 report: estimators and planted-answer validation

WhiteBox E&E. Started 2026-10-23. **Status: validation pending** (this file is completed after the frozen validation seeds run; sections marked PENDING).

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

## 4. Validation results (seeds 100–104) — PENDING

## 5. Gate — PENDING

## 6. Compute

Laptop CPU only for planted work; see `reports/compute_ledger.md`.
