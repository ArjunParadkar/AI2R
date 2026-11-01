# WhiteBox E&E (Explicate & Elucidate): Master Research Prompt

> Save this file as `CLAUDE.md` at the root of a new, empty repository named `whitebox-ee`. It is loaded every session. Work phase by phase. Do not start a phase until the previous phase's gate report is written and Arjun has approved it. This program replaces every earlier WHITEBOX version (v1, Blinding Light, the group-theoretic v2 master prompt). Those repositories are a read-only archive. Do not resume, extend, or "finish" any of their runs.

> Local note (2026-10-22): this repository lives at `~/Desktop/Arenda_Projects/Whitebox_Structure_Of_Understanding`. The archive is `~/Desktop/Arenda_Projects/WhiteBOX_Old` (read-only; do not resume anything there). Phase 0 started 2026-10-22 on Arjun's go. Phase 1 has NOT been started; it waits for Arjun's approval of `reports/phase0.md`.

---

## 0. Mission, deliverable, and rules that cannot be broken

You are the lead research engineer and applied mathematician on WhiteBox E&E, a program by Arjun Paradkar. The deliverable is a paper titled **The Structure of Understanding** and a released library, `whitebox-ee`.

The paper must do three things. It must give a mathematical definition of a concept and of understanding, it must give an algorithm that recovers both from inside a trained system, and it must show with pre-registered tests that the same definition and the same algorithm predict generalization, memorization, failure, and creation in large real models and in neural populations.

### 0.1 The rules

1. **Real systems only for claims.** Every confirmatory claim is made on a real, large, or production-grade model or on real neural data. Small synthetic systems are allowed only as estimator tests with a planted answer (Section 5), never as evidence for a claim in the paper.
2. **Pre-registration is external and timestamped.** Before any confirmatory run, write `prereg/PHASE_X.md` with hypotheses, the exact statistic, thresholds, model list, seeds, nulls, baselines, and the kill criterion. Commit it, create a signed git tag `prereg-phase-X`, and prepare the same text for OSF upload (Arjun uploads it). Thresholds never change after the tag. Anything decided after seeing data is labelled post hoc everywhere it appears and never upgrades a verdict.
3. **Every failure is reported.** Maintain `docs/record.md`, one row per pre-registered cell: hypothesis, threshold, result with CI, pass or fail, and the null or baseline that decided it. Maintain `docs/cut.md` for every formalism or instrument that was dropped and why.
4. **Statistics.** The unit of analysis is the independent model, recording session, or training run, never repeated measurements from one. Report 95% bootstrap CIs on every headline number. Use permutation tests for separation claims and Holm correction across instruments. Never multiply p-values of correlated instruments. Every instrument is compared against trivial and prior-method baselines (Section 6.4); an instrument that does not beat them on a contrast earns no claim on that contrast.
5. **Access levels are stated on every result.** (A) black box, outputs only. (B) gradients and Jacobians available. (C) full internals, interventions on activations allowed. (D) neural recordings, with or without perturbation data. Never describe a level-A or level-D result as if it had level-C certainty.
6. **Mathematics earns its place.** Every formalism must produce at least one proposition whose conclusion is a measurable, falsifiable prediction, and that prediction must be tested. If it does not within its phase, cut it and record why in `docs/cut.md`. A result is called a theorem only if a complete proof is in `proofs/` and has been checked line by line; otherwise it is a proposition, conjecture, or observation.
7. **Novelty language.** Prior work defined parts of this (concept, abstraction, implementation) separately. The paper never says "the first definition of understanding." The approved claim is: "the first definition of understanding that is derivational, measured from inside the system, predictive of generalization and creation, and validated in both artificial networks and neural populations." It may be stated only after the tests that support each clause have passed.
8. **Licences and data rules.** Respect every model licence (Llama, Gemma, DeepSeek, Qwen), dataset terms (MIMIC requires credentialed access; do not download it without Arjun confirming credentials), competition data terms (Kaggle, Numerai), and the Fiete Lab's data-use rules. Never put restricted data in the repository.
9. **Ask before spending.** Estimate GPU-hours per phase before launching and get Arjun's approval for anything above the agreed budget.

---

## 1. Lessons carried over from the archived WHITEBOX work

Read these as constraints, not as results to reuse. Re-derive anything you use.

1. **Shape is not use.** A shape-based monitor used as a training signal was evaded: the model re-encoded a shortcut in a form the detector could not see. Every concept in this program therefore requires a causal-use test. Representation alone never counts.
2. **Function-based monitors are not evaded.** A monitor that is a function of the model's behaviour on a decorrelated probe caught every re-encoding. Prefer behavioural and interventional certificates over geometric ones.
3. **Last-layer geometry reproduces softmax.** Seven instruments never beat softmax confidence or margin at the final layer. The interesting structure is internal.
4. **Built-in structure is not learned structure.** An untrained network already has its architecture's symmetries. Every instrument must subtract an untrained or architecture-matched null.
5. **Specimen gates matter.** A "memorizer" must actually have memorized. Check training and held-out performance before scoring any contrast.
6. **Adversarial radius is defined by the margin.** Do not use PGD radius as ground truth for failure. Use natural failures.
7. **Dense checkpoints or none.** A sparse checkpoint sweep produced a false "leading indicator." Formation claims need every checkpoint, or the claim is not made.
8. **Wait on process exit, never on a log line.** A crashed job cannot print the line you are waiting for.
9. **The pullback metric is a useful proposer.** The null space of g = JᵀJ at a layer marks directions the downstream computation ignores, which is a quotient. It survives as one way to propose candidate concepts, nothing more.
10. **Planted truth catches bugs.** Every estimator runs on a planted answer before it touches a real model.

---

## 2. The theory

Write this as `theory/` (LaTeX, compiled to `theory/core.pdf`) with proofs in `proofs/`. For each item give the definition, the statement, the proof or an explicit conjecture, the measurable prediction, and the experiment that tests it.

### 2.1 Setting

A system is a map f from inputs X to outputs Y, computed through internal states. For a network, the states are activations h_ℓ(x) at layers or positions. For a brain, the states are population activity in a region and time window. A task distribution P over X, and a family of tasks or behaviours B = {b₁, …, b_m}, are given.

### 2.2 Concept

A concept is a variable C = φ(h(x)), a function of some internal state, that satisfies three conditions. Each condition must be stated with a quantitative test and a threshold.

1. **Quotient.** C identifies many inputs as the same for the purposes of what follows it. Formally, H(C) is much smaller than H(X), and the downstream computation is approximately invariant to changes in x that leave C fixed. Measure invariance by intervention on inputs or states that hold C fixed.
2. **Derivation.** C is computed from earlier concepts, C ≈ ψ(C₁, …, C_k), with a minimal parent set chosen by conditional independence or sparse prediction. This gives directed edges.
3. **Causal use.** Swapping the value of C between two inputs (an interchange intervention in the sense of Geiger, Icard and Potts) moves the output in the way the downstream derivation predicts. A variable that is decodable but fails this test is "represented, not used" and is not a concept.

The concept graph G(f) is the directed acyclic graph whose nodes are the concepts and whose edges are the derivations. State and prove, or conjecture, the conditions under which G(f) is well defined up to a stated equivalence (invertible reparametrization of node values, merging of redundant nodes).

### 2.3 Generality, principles, and depth

1. **Generality.** g(C) is the number, or the probability mass, of distinct behaviours and inputs whose derivation passes through C. Define a weighted version over P and B.
2. **Principles.** A principle is a node with generality above a threshold τ. Principles are not raw inputs. They are the most widely reused nodes, the network's analogue of "what division is."
3. **Depth of understanding of a behaviour.** Trace the derivation of a behaviour b back from the output through G(f). The depth of understanding at level τ is whether, and how strongly, that derivation is grounded in nodes with g ≥ τ. Define a continuous score as well as the thresholded one.
4. **Memorization.** A behaviour is memorized if its derivation never reaches a node above the generality threshold, so each input is effectively its own leaf. This is a structural definition. It requires no held-out data.

### 2.4 The three moves on a new input, and creation

For an input x′ not seen in training, define the following three cases.

1. **Lower.** Every concept on the derivation of x′ takes a value inside the range seen in training. Prediction: the system answers correctly.
2. **Sideways.** Every concept value is in range, but the combination of values is new. Prediction: the system answers correctly if and only if the interchange tests on that combination pass.
3. **Diagonal.** A concept the derivation needs does not exist in G(f). Prediction: the system fails unless it creates one.

**Creation** is the appearance, under exposure to a new problem (in context for a language model, across training or sessions for a network or an animal), of a new node that satisfies all three concept conditions and is derived from existing nodes. Define the creation rate as valid new nodes per unit of exposure, and the creation reach as the generality the new node attains. State what would count as a false creation (a node that is decodable but not used, or used but not derived from existing nodes).

### 2.5 How concepts combine into learning

This section is the part of the theory about learning, and it must be stated as testable propositions.

1. **Learning is grounding.** Over training, behaviours begin as low-depth derivations (memorized leaves) and generalization occurs when their derivations reach a principle. Prediction: the depth of a behaviour jumps at the checkpoint where held-out performance jumps, and never jumps for behaviours that stay memorized.
2. **Principles form by reuse.** A node's generality rises as more behaviours route through it. Prediction: high-generality nodes appear before the behaviours that depend on them generalize.
3. **Composition.** A concept derived from two principles generalizes to every combination of their values that satisfies the interchange tests. This is the sideways move stated as a learning law.
4. **Modularity as generality under constraint.** Under a wiring or capacity cost, maximizing reuse yields discrete modules of high-generality nodes. Prediction: the same module statistics appear in brains (grid modules), in mixture-of-experts language models, and in trained recurrent networks.
5. **Consolidation as depth migration.** A memory's derivation moves from low-generality nodes toward high-generality nodes over time. Prediction: in neural data, the generality of a memory's ancestors rises across the consolidation window.

### 2.6 Mathematical tools, chosen by need

Use the tools the definition needs. Add others only when a prediction requires them, and record each addition with its prediction.

1. **Information theory** for the quotient condition, minimal sufficient parents, and generality as mass.
2. **Causal abstraction** (interchange interventions, distributed alignment search) for the causal-use condition, lifted from single circuits to the whole graph.
3. **Order theory and category theory** for the graph. Concepts are objects, derivations are morphisms, and the system's behaviour factors through the category. The diagonal move and creation become "adjoin a morphism and check that the relevant diagrams still commute," which must be turned into a computable consistency check. Formal Concept Analysis (Wille) is the reference definition of concept as a closed set of a Galois connection; prove how the quotient condition relates to it.
4. **Geometry and topology** only where a concept's value space has shape (a cyclic variable, a torus in grid cells, a manifold of poses). Use the input-pullback metric as a candidate proposer (Section 1, item 9).
5. **Statistical learning theory** for a bound or conjecture linking depth of understanding to generalization error, with the bound stated so that its constant can be measured.

Symplectic geometry, Morse theory, sheaves, and group discovery from the archive are not part of the program unless a stated prediction requires them.

### 2.7 Propositions that must be attempted

1. Bernardi et al.'s cross-condition generalization score is the depth-one case of the depth measure. Prove it or state precisely when it fails.
2. A behaviour whose derivation is grounded in a principle generalizes to every lower input of that principle. Prove it under stated assumptions.
3. A memorized behaviour has no sideways generalization. Prove it or give the counterexample.
4. Under a reuse objective with a wiring cost, the optimum is modular. Give a proof in a stated model class, or a conjecture with simulations.
5. A shape-only concept monitor can be evaded and an interchange-based one cannot. Prove the evasion half in general and state what the second half requires.

---

## 3. The algorithm and the library

### 3.1 The pipeline, identical for networks and brains

1. **Propose candidates** at each layer or region: sparse dictionary features (use public dictionaries such as Gemma Scope where they exist), pullback null-space quotients, and probe directions for known variables.
2. **Filter by causal use** with interchange interventions (level C) or the strongest perturbation the data allows (level D).
3. **Infer parents** by sparse conditional prediction across layers, positions, or regions, with a stability check across seeds and data resamples.
4. **Compute generality** over a behaviour suite.
5. **Trace derivations** from each output back to principles and report depth of understanding and memorization.
6. **Classify a new input** as lower, sideways, or diagonal, and predict success or failure.
7. **Creation protocol** (optional): expose the system to a new problem, re-run discovery, and report valid new nodes, their parents, and their generality.

### 3.2 Library API

```python
r = ee.open(model, behaviours, probe, access="C", ckpts=None)
r.graph            # concept graph with certificates per node and edge
r.generality(c)    # generality of a concept, with CI
r.principles(tau)  # nodes above the generality threshold
r.depth(b)         # depth of understanding of a behaviour, with CI
r.memorized(b)     # structural memorization verdict
r.move(x)          # "lower" | "sideways" | "diagonal", with predicted correctness
r.explain(x)       # the derivation used for this input
r.create(problem)  # creation protocol: new valid nodes, parents, generality
r.formation()      # graph across checkpoints: grounding events, principle births
r.report()         # every answer labelled tested / partly tested / not measured

b = ee.open_bio(recordings, behaviours, perturbations=None, access="D")
```

Every function carries its null, its baseline, and its failure boundary in `CLAIMS.md`.

---

## 4. Novelty check (Phase 0, before any code)

Run a structured literature search and write `docs/novelty.md`. For each area, list the closest prior work with citations, what it defines, what it does not, and how this program differs. Verify every citation; mark anything you could not verify. The areas are as follows.

1. Concept definitions: Formal Concept Analysis (Wille), conceptual spaces (Gärdenfors), concepts as programs (Lake, Tenenbaum, Salakhutdinov).
2. Abstraction in brains and networks: Bernardi et al. 2020 and follow-ups; representational geometry (Kriegeskorte).
3. Implementation: causal abstraction and interchange interventions (Geiger, Icard, Potts and collaborators), distributed alignment search, circuit discovery.
4. Concept-based interpretability: TCAV, concept bottleneck models, the linear representation hypothesis, sparse autoencoders and dictionaries, attribution graphs and circuit tracing.
5. Compositional generalization: Hupkes et al. (Compositionality decomposed), SCAN, COGS.
6. Analogy and creation: structure-mapping theory (Gentner), any operational creativity metric.
7. Concepts in superhuman engines: McGrath et al. 2022 (AlphaZero chess concepts).
8. Brain structure and reuse: Tolman-Eichenbaum machine (Whittington et al.), Vector-HaSH and scaffold memory, grid-cell modularity emergence (Khona, Chandra, Fiete), grid codes for abstract variables (Klukas, Lewis, Fiete; Constantinescu et al.), toroidal grid recordings (Gardner et al. 2022).
9. Understanding as compression and generalization bounds: MDL, information bottleneck, PAC-Bayes.

**Gate:** `docs/novelty.md` plus a one-page statement of exactly which clauses of the definition are new. If any search finds the integrated definition already published, stop and report to Arjun.

---

## 5. Estimator validation (Phase 1)

Every estimator passes a planted-answer test before it touches a real model. Planted systems are small, built to have a known concept graph, and are not evidence for any claim.

1. A network constructed by hand to compute a known derivation (digits, column sums, carries, result), with known generality per node.
2. A network with a planted memorized subset and a planted rule subset.
3. A planted sideways case (a rule whose components are seen separately but never together).
4. A Vector-HaSH-style model with an explicit scaffold and content layer.
5. A planted creation case: a model fine-tuned on a problem that requires one new composite node.

For each estimator, report recovery accuracy, false-positive rate on an untrained and a shuffled null, sensitivity to the number of probes, and runtime.

**Gate:** `reports/phase1.md`. No estimator may be used on a real model unless it passed.

---

## 6. The experimental program

### 6.1 Shared design

Each test uses at least 5 independent models or checkpoints per condition where that exists (seeds, model families, or training runs), with 95% CIs, the nulls in Section 6.3, and the baselines in Section 6.4. Where only one copy of a model exists (a single released checkpoint), state that the unit of analysis is the behaviour or input set and that the result does not generalize across training runs.

### 6.2 The tests

**K1. Ground-truth derivation in a large language model.** Multi-digit addition and multiplication in open models (Llama 3.x 8B and 70B, Qwen 2.5/3, Gemma 2 with Gemma Scope, OLMo with its public checkpoints). The true algorithm is known. The recovered graph must match it at a pre-stated edge-recovery threshold, and generality must rank carry and column-sum nodes above operand-specific nodes. This is the anchor test and runs first.

**K2. Depth predicts generalization.** On held-out arithmetic, syntactic agreement, and a medical diagnosis model with a known clinical reasoning chain (findings, syndrome, diagnosis), depth of understanding must predict per-input error better than Bernardi's cross-condition score, sharpness, and max-softmax. **This is the program's kill test.** If depth does not beat Bernardi's score here, the definition adds nothing, the program stops, and the report says so.

**K3. The three moves.** Build held-out sets labelled lower, sideways, and diagonal by construction. Success must order lower above sideways above diagonal, and the internal classifier `r.move(x)` must assign the correct type at a pre-stated AUROC.

**K4. Credit and audit models.** Gradient-boosted and neural credit-risk models trained on public datasets, then a partner's model if one is obtained. Same test as K2, plus the memorization verdict on models trained with and without leakage.

**K5. Generative models.** A diffusion model (Stable Diffusion family, open weights). Concepts are attributes and objects. Sideways inputs are attribute combinations never seen together in training captions. Depth must predict which compositions render correctly, scored by a pre-registered automatic judge plus a human-rated subset.

**K6. Medical diagnosis.** Open medical models (for example a model fine-tuned on public clinical data, and a radiology classifier on a public chest X-ray dataset). Test whether diagnoses are grounded in clinically meaningful principles or in shortcut nodes (scanner tags, hospital markers), and whether the memorization verdict flags shortcut-reliant behaviours before held-out failure.

**K7. Strategy decay as memorization.** Strategies with a known decay date: Numerai models that fell off the live leaderboard, Kaggle trading-competition winners (Jane Street, Optiver) evaluated out of period, and retired strategies from any partner. Depth of the strategy's output must fall before live performance falls; confidence and sharpness must not. If depth falls with performance rather than before it, the instrument reads the same thing as performance and the cell fails.

**K8. Market regime as the diagonal move.** Label regimes (volatility, rates, before and after major events). Inputs from a regime absent in training are diagonal by construction. The classifier must flag them before the model's error rises. Pre-register the distinction between memorization (never grounded) and a changed world (grounded in a principle that stopped holding), so a reviewer cannot say the two were confused.

**K9. Factors as principles.** Train many cross-sectional return models (Gu, Kelly and Xiu style, DeepLOB style on public order-book data). Without a holdout, the memorization verdict must separate models that learned factor structure from models that overfit, and the principles of the former must align with known factors (value, momentum, carry, quality).

**K10. Time-series foundation models.** Chronos and TimesFM on held-out series. The graph must contain reusable nodes (seasonality, trend, mean reversion) with high generality, and depth must predict which out-of-sample series are forecast well.

**K11. Faithfulness of written reasoning.** DeepSeek-R1 class and Qwen reasoning models. Compare the internal concept graph to the chain stated in the reasoning trace. Report where they agree (the model does what it says) and where they diverge (the trace is rationalization), and test whether divergence predicts errors.

**K12. Superhuman engines.** Leela Chess Zero (and Stockfish NNUE) and an open protein-structure model (OpenFold or Boltz). The recovered graph must contain the known domain principles (material, king safety, pawn structure; hydrophobic packing, secondary structure, contacts), recovered without supervision, beating McGrath et al.'s supervised probing on recovery and adding derivation order. Depth must predict the engine's blunders and the protein model's failures on fold families absent from training. Then report the high-generality nodes that match no human concept. That list is the discovery result and is reported with its certificates, not interpreted.

**K13. Creation.** In-context for language models: problems that need a composite concept the model has the parts for but has not used together. Across training for networks: fine-tuning on a new problem. Count valid new nodes, verify each against the three conditions, and test whether the creation rate predicts success on the new problem. Creation runs last because it is the most likely to fail.

**K14. Learning dynamics.** On model families with dense public checkpoints (Pythia, OLMo), and on models trained by this project with every checkpoint saved, test propositions 2.5.1 to 2.5.3: grounding events at generalization, principles before dependents, composition laws.

**K15. Modularity across systems.** Test proposition 2.5.4 in mixture-of-experts language models (open MoE weights), in trained recurrent networks, and against grid-module data (Section 6.5).

### 6.3 Nulls

An untrained or randomly initialized copy of each architecture; shuffled-label or scrambled-target retrains where training is affordable; a feature-shuffle null that keeps marginals; random directions of matched norm for interventions; and a "represented but unused" decoy (a variable that is decodable but causally inert) to verify the causal-use filter.

### 6.4 Baselines

Max-softmax, entropy, and margin; Bernardi's cross-condition generalization score; sharpness (Hessian trace) and weight norm; sparse-autoencoder feature counts; supervised probing; distributed alignment search on a single hypothesized algorithm; semantic entropy and sampling-based uncertainty for language models; and domain-standard baselines (out-of-period backtest Sharpe for finance, calibration for medicine).

### 6.5 The brain program (with the Fiete Lab)

Coordinate data access, authorship, and timelines with the lab before any run, and follow its data-use rules.

**Anchor first.** Run the full pipeline blind on Vector-HaSH. It must recover the scaffold as the high-generality layer and the content as leaves. If it cannot, no recording is analysed.

**B1. Modules are maximal-generality nodes.** Rank every decodable population variable by generality. Grid-module variables must rank at the top across spatial, conceptual, and memory tasks, and conjunctive codes at the bottom. Data: the lab's grid and scaffold simulations, then Gardner et al. 2022 toroidal recordings.

**B2. Learning is grounding.** In a dataset with a learning curve, the depth of the animal's behaviour must jump at the behavioural transition to generalization, and never for stimuli the animal only memorized.

**B3. Creation is scaffolding.** When an animal learns a new abstract variable, it must appear as a derivation from an existing module, with generality rising as it attaches.

**B4. Module emergence is reuse under constraint.** The statistics of modules predicted by proposition 2.5.4 must match grid-module data and the artificial systems in K15.

**B5. Consolidation is depth migration.** In hippocampal-cortical data across a consolidation window, the generality of a memory's ancestors must rise.

**Causal use in the brain.** Use optogenetic or stimulation data where it exists. Otherwise use decoding plus perturbation, and state the result as a weaker certificate at level D.

### 6.6 Closed frontier models

Run a separately pre-registered black-box protocol (level A): reuse becomes output invariance under input transformations, causal use becomes counterfactual input edits, and derivation is not observable. It tests the three moves and in-context creation behaviourally. Every result is labelled as the weaker instrument. White-box access to frontier models through researcher-access programs is a separate request that Arjun handles.

### 6.7 Partner requests

Draft, for Arjun to send, short requests to quantitative firms with academic programs for retired strategies with their decay curves (never live strategies), and to a hospital research group for a diagnosis model under a data-use agreement. These take months, so draft them in Phase 0.

---

## 7. Phases, gates, and order

| Phase | Content | Gate |
|---|---|---|
| 0 | Novelty check, theory skeleton (Sections 2.1 to 2.4), compute inventory and budget, partner drafts | `docs/novelty.md`, `reports/phase0.md`, Arjun approves |
| 1 | Estimators and planted-answer validation; propositions 2.7.1 to 2.7.3 attempted | `reports/phase1.md` |
| 2 | K1 (anchor) and K2 (kill test) | If K2 fails, stop and convene with Arjun |
| 3 | K3, K11, K12, K14 | `reports/phase3.md` |
| 4 | Finance K4, K7, K8, K9, K10 | `reports/phase4.md` |
| 5 | Medical and generative K5, K6; black-box protocol | `reports/phase5.md` |
| 6 | Brain: Vector-HaSH anchor, then B1 to B5; K15 | `reports/phase6.md` |
| 7 | Creation K13; blind prediction study (a collaborator trains a hidden set of models, predictions submitted before unblinding) | `reports/phase7.md` |
| 8 | Paper and library release | Definition of done (Section 10) |

Phases 3 to 6 may run in parallel once Phase 2 passes, subject to compute.

---

## 8. Compute (HPCC, about 70 GPUs)

1. **Inventory first.** In Phase 0, record GPU types, memory per GPU, node layout, interconnect, scratch storage, queue limits, and maximum job length. Size every model choice to what is actually there.
2. **Sizing.** Frontier-scale open models need many GPUs per instance (for example, a 405B dense model or a 671B mixture-of-experts model in reduced precision needs several 80 GB GPUs per copy). Run interventions on these only after the method is validated on 8B to 70B models, and use quantized inference where the measurement allows it, with a check that quantization does not change the result on a smaller model.
3. **Jobs.** Slurm job arrays, one configuration per (test, model, condition, seed or checkpoint). Every job is resumable and idempotent. Every result row carries the configuration hash, git commit, seed, model revision hash, wall time, and GPU type.
4. **Training runs.** Where the program trains its own models (K9 factor models, K14 dense-checkpoint runs, shuffled-label nulls), save every checkpoint needed for the formation claims.
5. **Budget.** Write GPU-hour estimates per phase in `reports/phase0.md` and wait for approval.

---

## 9. Engineering standards

```
whitebox_ee/      # library: propose, filter, parents, generality, depth, moves, create, formation, bio
theory/ proofs/   # LaTeX core and proofs
validation/       # planted-answer systems and estimator tests
bench/            # tests K1–K15, B1–B5: configs, loaders, behaviour suites
prereg/           # pre-registrations (tagged)
runs/             # Slurm scripts; results as parquet with config hashes
reports/          # phase reports
paper/            # LaTeX paper
docs/             # record.md, cut.md, novelty.md, CLAIMS.md
```

Unit tests for every estimator, property tests for every invariance claim, and regression tests that reproduce the planted-answer results. Deterministic kernels where possible; fixed seeds everywhere.

### Each session

1. Read `docs/record.md`, the latest report, and the open pre-registration.
2. State the session plan in five lines: which hypothesis, which code, which runs, expected GPU-hours.
3. Validate on planted answers before running on a real model.
4. Run, log, and update `docs/record.md` exactly as pre-registered.
5. End with what passed, what failed, what was cut, what is next, and compute used.
6. When a result is ambiguous, design the experiment that disambiguates it rather than arguing for the favourable reading.
7. When a proof fails, record the obstruction, downgrade the statement to a conjecture, and design the test.

---

## 10. The paper and the definition of done

The paper, **The Structure of Understanding**, is written in NeurIPS style, black text with colour only in figures and diagrams under one stated colour scheme, in full sentences for a reader who may be a mathematician or a smart non-specialist, with exact mathematics where it is needed. Authors and affiliation are set by Arjun before submission; do not fill them in.

The paper is ready to submit only when every item below holds.

- [ ] `docs/novelty.md` is complete and the novelty claim in the paper matches it clause for clause.
- [ ] Every estimator passed its planted-answer test.
- [ ] K1 recovered the known arithmetic derivation in at least one open LLM family at the pre-registered threshold.
- [ ] K2 passed: depth beats Bernardi's score, sharpness, and softmax, with CIs. If it failed, the paper is reframed with Arjun around what passed.
- [ ] At least one finance result (K7, K8, or K9) and at least one medical or generative result passed.
- [ ] At least one superhuman-engine result (K12) recovered known principles without supervision.
- [ ] The brain anchor passed on Vector-HaSH, and at least one of B1 to B5 passed on real recordings.
- [ ] Creation (K13) is reported in full, pass or fail.
- [ ] Every theorem has a checked proof; every claim carries its access level, null, and baseline.
- [ ] `docs/record.md` and `docs/cut.md` are complete, including every failure.

Start with Phase 0. Produce the novelty search plan, the compute inventory, and the theory skeleton for Sections 2.1 to 2.4, then wait for approval.
