# WHITEBOX v2 — Research Program: Measurement Theory of Interpretability

(Program brief as given 2026-08-30. Verbatim; this is the pre-registration of the
programme itself. v1 = everything done on WHITEBOX before this — see WHITEBOX_V1.md.)

## CONTEXT (read first, this determines how you work)

You previously built whitebox-topo v1 at /home/god/whitebox-lib (Phases 0–5 complete, committed). The critical history: v1's validation batteries killed most of the original paper's claims — the 2,900× TC collapse was an artifact of models that never fit the noise, the memorization signal flips sign across architecture families (cross-family AUC 0.00), Sub-H4 correspondence hypotheses were refuted on real data, P4 died under a label-permutation null, P1 didn't replicate. What survived: a per-family memorization signal (LOSO AUC 0.98 CNN / 1.00 LM, converged checkpoints only) and the honest validation methodology itself.

The lesson that governs everything below: every claimed signal must survive (a) a proper null (shuffle/permutation/random-matrix), (b) calibration with uncertainty, (c) cross-family testing, (d) an attempt to kill it. We assume our own results are artifacts until controls fail to destroy them. This discipline is the lab's actual competitive advantage — the published papers we build on mostly did NOT run these controls.

We did deep literature review. The black box decomposes into seven core opacities:

1. Superposition — features smeared across shared dimensions; no clean unit of meaning.
2. Dark matter — SAE/transcoder decompositions leave 15–20%+ unexplained variance that may contain structured computation (Engels/Smith/Tegmark 2024).
3. Non-identifiability — multiple incompatible circuit explanations pass all validity criteria (Méloux et al. 2025); Hydra effect redundancy.
4. Computational intractability — circuit discovery NP-hard, global faithfulness Σ₂ᵖ-complete (Adolfi et al. ICLR 2025); inapproximable.
5. QK gap — attention-pattern formation largely unexplained; circuit tracing freezes attention.
6. Per-prompt vs global — explanations don't generalize; models use multiple mechanisms per task.
7. Alien concepts — possible inexpressibility in human ontology (conjecture level).

Novelty verdicts from our searches (respect these — do not claim what's taken):

* Parameter-symmetry theory is mature (2025 survey "Symmetry in Neural Network Parameter Spaces": fibers, hidden symmetries per Grigsby, completeness theorems). Méloux's explanation-multiplicity is separate literature. The bridge between them is unclaimed — that's our E1.
* Sheaf-of-explanations framework is TAKEN (arXiv 2603.17141, March 2026, category-theoretic with gluing counterexamples; also "A Gauge Theory of Superposition" — sheaf atlas with holonomy obstructions on Llama 3.2 3B). We cite both. The empirical Čech computation over prompt space with real numbers appears open — that's our E3.
* Modularity-aids-interpretability qualitative version is TAKEN (Golechha et al. 2024/2025: clusterability loss, smaller circuits, reduced search space). Open: parameterized-complexity phase diagram (E4) and cross-substrate biological comparison (E5).
* RMT null on SAE residuals: unverified novelty — run a targeted search before writing up E6.

Cite generously: Méloux et al., Adolfi et al., Grigsby et al., the symmetry survey, Golechha et al., Engels et al., arXiv 2603.17141, the gauge-superposition paper, Rucco 2026, RTD (Barannikov ICML 2022), Anthropic circuit tracing. We build on; we do not reinvent or overclaim.

## STAGE 1 — EXPERIMENTS

Work in a new repo whitebox-v2 (fresh, imports v1 as a dependency where useful). Each experiment gets its own directory with SPEC.md (pre-registered hypotheses and pass/fail criteria written BEFORE running), run_*.py, results/ (CSVs + 300dpi figures), and FINDINGS.md (honest, includes negatives).

### E1 — Gauge orbits vs circuit non-identifiability [FLAGSHIP, do first]

Question: Is the multiplicity of valid mechanistic explanations pure gauge (one mechanism seen through symmetry transformations) or genuine (multiple distinct mechanisms)?

Setup: Replicate Méloux-style enumeration. Train small MLPs (2–3 hidden layers, 4–10 neurons/layer, ReLU) on Boolean functions (start: 3-input functions — XOR3, majority, parity; then 4-input). Train to zero error, multiple seeds (≥20 per function).

Enumeration: For each trained network, exhaustively enumerate valid circuit explanations: all subnetworks (neuron subsets) that reproduce the function on all inputs when the complement is mean-ablated AND zero-ablated (record both; they may disagree — that disagreement is itself data). For each circuit, enumerate valid interpretations: consistent assignments of Boolean gates to neurons (a neuron implements gate g if its binarized activation matches g's truth table across all inputs).

Symmetry group: Compute the known symmetry group G acting on each network: (a) hidden-neuron permutations within layers, (b) positive scalings through ReLU (w→αw, w_out→w_out/α, α>0), (c) check for hidden symmetries via Grigsby's functional dimension — compute rank of the Jacobian of the realization map at the trained parameters; rank deficit beyond permutation+scaling count indicates hidden symmetries (report them, don't need to fully characterize).

The test: Define equivalence: two circuit explanations are gauge-equivalent if some g∈G maps one's parameter-implementation to the other's (for permutations: circuit A maps to circuit B under the neuron relabeling; for scalings: same support, rescaled weights). Partition all enumerated explanations into G-orbits. Primary metric: orbit count / explanation count. Ratio → 1/|explanations| means multiplicity is pure gauge (mechanism unique up to symmetry). Ratio near 1 means genuine multiplicity.

Pre-registered hypotheses (write in SPEC.md before running): H-E1a: ≥50% of explanation multiplicity is gauge-explained (orbit count ≤ half of explanation count). H-E1b: orbit count grows slower with network width than raw explanation count. Report whatever happens.

Controls: Random (untrained-but-function-matching via lookup layers, if constructible) baselines; verify orbit-partition code on hand-constructed networks with known symmetries; seeds.

Figures: orbit-count vs explanation-count across functions and widths (log scale); example of two gauge-equivalent circuits visualized; example of two genuinely-distinct circuits if found.

### E2 — Gauge-invariant observables [immediately after E1, same networks]

Question: Which interpretability quantities are constant on gauge orbits (and therefore meaningful), and which popular ones aren't?

Method: For each network from E1, generate ≥100 random orbit representatives (apply random permutations + random positive scalings; include hidden-symmetry directions if E1 found them). Compute across representatives, report coefficient of variation for each candidate quantity:

* Should be invariant: persistence diagrams of activation point clouds (β₀, β₁, TC — permutation-invariant by construction; test scaling behavior explicitly, since scalings change geometry — this is a real question, not a formality: report whether TC is scaling-invariant, scaling-covariant, or neither), singular value spectra of layer maps (permutation-invariant, NOT scaling-invariant — verify and report), products of weight matrices input→output (fully invariant — sanity check, CV should be ~0).
* Expected NOT invariant (the indictment list): individual neuron activation magnitudes, neuron-level attribution scores, raw weight magnitudes, "top neuron" identifications, per-neuron SAE-style feature assignments.

Deliverable: a two-column validated table — invariant vs non-invariant interpretability quantities, with measured CVs. Frame: "an interpretability claim is well-defined only if gauge-invariant; here is the empirical audit." This table is a core WHITEBOX v2 artifact.

### E6 — Dark-matter structure test [run in parallel with E1/E2 — no training needed]

Question: Is SAE reconstruction error noise, or does it contain structured computation?

Data: Public SAEs — Gemma Scope (Gemma-2-2b residual stream SAEs) and/or GPT-2-small SAEs from SAELens. Sample ≥50k tokens of natural text; compute error vectors e = x − SAE(x) per token per layer (3 layers: early/middle/late).

Test 1 (RMT null): Eigenvalue spectrum of the error covariance across tokens vs Marchenko–Pastur law (matched aspect ratio and variance). Count and report eigenvalues outside the MP bulk (use standard Tracy–Widom edge threshold). Structured fraction = variance in outlier eigenvalues / total.

Test 2 (topology with nulls): v1 pipeline on the error point clouds: TC, β₁ with the v1 shuffle null (feature-shuffled errors preserving marginals). Report TC/TC_shuffled ratios with bootstrap CIs.

Test 3 (predictive structure): Following Engels et al. (cite), linear-probe the error vectors for next-token information: train a linear map from e to the model's logit delta; report variance explained vs shuffled-e control.

Pre-registered: H-E6: >10 MP-outlier eigenvalues per layer and TC/TC_shuffled > 1.2 → dark matter is structured. Before writing this up, run a targeted novelty search: "random matrix theory Marchenko-Pastur null sparse autoencoder error residual" — flag results to me.

### E3 — Empirical explanation cohomology [after E1/E2]

Question: Can we measure the obstruction to a global explanation as actual Čech cohomology, with numbers?

Setup: GPT-2-small, IOI task (established multiple-mechanism case; cite the data-driven circuit discovery literature). Generate ≥2000 IOI prompts spanning templates/name-positions. Embed prompts (final-layer mean activation), cluster into k=8–15 overlapping clusters (soft clustering — fuzzy c-means or Gaussian mixture posteriors thresholded so clusters overlap; overlaps are required, they're where restriction maps live).

Sections: Per cluster, run attribution-patching-based circuit discovery (edge-level scores, threshold to a subgraph). The section over cluster U_i = its circuit C_i (as an edge set with weights).

Sheaf structure: Nerve of the cover (vertices = clusters, edges = nonempty pairwise overlaps, triangles = triple overlaps). Restriction: circuit C_i restricted to overlap U_ij = re-scored edge set on the overlap's prompts. Agreement on overlap: Jaccard similarity of thresholded edge sets ≥ τ (sweep τ). Čech: H⁰ dimension = number of connected consistent explanation components; compute the H¹ obstruction cocycle from pairwise disagreements that cannot be resolved by any global re-assignment (implement as the cycle-space computation on the disagreement-weighted nerve graph — document the exact construction in SPEC.md; where a design choice is ambiguous, make it, state it, and note the alternative).

Deliverables: obstruction number vs τ curve; the nerve visualized with disagreement weights; identification of WHICH prompt-cluster pairs block gluing (these are the model's distinct mechanisms, now located). Cite arXiv 2603.17141 and the gauge-superposition paper prominently as the frameworks; our contribution is the first concrete measurement on a real model. Sanity control: run the same pipeline on a task with a known single mechanism (e.g., simple induction) — obstruction should be ≈0. If it isn't, the method is broken; debug before believing IOI numbers.

### E4 — Legibility phase diagram [after E3]

Question: Which structural parameters of a trained network predict whether circuit discovery succeeds?

Ensemble: ≥60 small models spanning: MLPs and 2–4 layer transformers; trained on algorithmic tasks (modular addition, sorting, copying); varying L1/weight-decay/dropout; plus Golechha-style clusterability-loss models (cite them; reuse their loss); plus random-labeled and untrained controls.

Measure per model: (a) modularity Q of the thresholded attribution graph (Leiden, sweep threshold), (b) treewidth estimate (use a heuristic upper bound — min-fill — exact treewidth is itself NP-hard), (c) effective sparsity. Then run a fixed circuit-discovery pipeline (attribution patching + greedy pruning to minimal faithful circuit, fixed faithfulness threshold 0.9) and record: search time, discovered circuit size, faithfulness achieved, stability across 3 discovery seeds.

Deliverable: the phase diagram — discovery success/cost as a function of (Q, treewidth, sparsity). Pre-registered: H-E4: Q and discovery success are positively related with a threshold/knee, not linearly. Also report the Adolfi connection honestly: this is empirical parameterized-complexity evidence, not a theorem. If the data is clean, attempt the theorem statement (modularity as the parameter that makes discovery FPT) as a conjecture with the empirical support — flag for me to take to JVHM for the formal proof attempt.

### E5 — Cross-substrate legibility [LAST, and only alongside my lab obligations]

Same measurements as E4 — modularity Q, treewidth heuristic, the full legibility profile — computed on: (a) the fly head-direction connectome (public hemibrain data; EPG/PEG/PEN/Delta7 subcircuit — the exact system from my Fiete-lab project), (b) RNNs trained on head-direction integration tasks (train 10 seeds, standard continuous-attractor task). Question: do biological circuits occupy the legible region of E4's phase diagram, and do task-matched artificial networks land in the same region? Deliverable: one figure placing biology and artificial networks on the same phase diagram. This is a preliminary result for discussion with my mentors — mark everything from this experiment as NOT for external claims until I clear it with the lab.

## STAGE 2 — WHITEBOX v2 MODULES

Only after Stage 1 results are in. Each surviving finding becomes a calibrated module in whitebox-v2, held to v1's Phase-4 standard (nulls, CIs, per-family calibration, documented failure boundaries, no signal in the public API without a validation battery entry):

* whitebox.invariants — E2's validated gauge-invariant metrics only, each with its invariance test shipped as a unit test. Includes a check_invariance(metric_fn, model) utility that empirically audits ANY user-supplied metric by orbit sampling — this is the tool version of E2 and likely the most-used function in the library.
* whitebox.gluing — E3's pipeline as explanation_obstruction(model, prompts, task_fn) → obstruction score + the blocking cluster pairs. Honest docstring: framework due to [citations]; this is the measurement instrument.
* whitebox.legibility — E4's legibility_score(model) → predicted circuit-discovery tractability BEFORE burning compute, calibrated on the Stage-1 ensemble, with stated scope (small models; extrapolation flagged).
* whitebox.darkmatter — E6's residual_structure(sae, activations) → structured fraction with MP null + topology null.
* Carry over v1's surviving memorization_signal (per-family, converged-only) unchanged.

Each module: VALIDATION.md entry, benchmark battery, figures. Update the top-level AUDIT.md with the v2 findings including all negatives.

## STAGE 3 — LIBRARY RELEASE PREP

* Rename the import to whiteboxtopo (v1 note: whitebox collides with an existing PyPI distribution).
* Unified docs: README quickstart demonstrating the honest pitch — "instruments that tell you which parts of a model are explainable (legibility), whether your explanation is well-defined (invariants), whether it generalizes (gluing), and how much computation your decomposition missed (darkmatter)" — explicitly NOT "we opened the black box."
* Full test suite, CI, the validation batteries as shipped benchmarks, 300dpi figure regeneration scripts.
* A CLAIMS.md mapping every public signal to its evidence, its nulls, and its known failure modes. This file is the library's integrity backbone.

## OPERATING RULES

1. SPEC.md with pre-registered hypotheses BEFORE each experiment runs. Never retrofit.
2. Every positive result gets an attempted kill: null model, permutation test, or adversarial control. Report the kill attempts in FINDINGS.md.
3. Negatives are deliverables. Write them up with the same care.
4. Cite the real literature at every step; never describe a taken idea as ours.
5. Pause after E1, after E2+E6, after E3, and after E4 — print a findings summary and wait for my review before continuing.
6. Compute discipline: E1/E2 are CPU-scale; E6 needs one GPU for activation extraction; E3/E4 need the local GPU; nothing here needs the cluster yet. Flag if that changes.
7. If any result contradicts the plan's framing, the result wins. Rewrite the framing, not the data.

Start with E1: write E1/SPEC.md first, show it, then build the enumeration and orbit-partition code.

Process note: rule 5's pause points matter. Do not run all six experiments unsupervised — E1's orbit-partition definition has judgment calls in it; the SPEC is reviewed before compute is spent. Review at each pause, then release the next stage.
