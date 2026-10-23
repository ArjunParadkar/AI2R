# Phase 0 report: novelty, theory skeleton, compute, partners

WhiteBox E&E. Written 2026-10-22. Phase 0 started 2026-10-22 on Arjun's go and is complete as of this report.

**Gate status: APPROVED by Arjun on 2026-10-22** (see §10). Before approval no estimator code had been written, no GPU was used, and nothing was sent to anyone.

## 1. Deliverables

| Item | Path | State |
|---|---|---|
| Novelty search, nine areas plus the stop-condition search | `docs/novelty.md` | Done. 147 citations fully verified, 15 verified with one field UNVERIFIED, 2 UNVERIFIED (not relied on). |
| Statement of which clauses are new | `docs/novelty.md`, last section | Done |
| Theory skeleton, Sections 2.1–2.4 | `theory/core.tex` → `theory/core.pdf` (14 pp., compiled with latexmk/pdfLaTeX, TeX Live 2026) | Done |
| Proofs | `proofs/P1`–`P6` (`.tex`, included in the PDF appendix) | Written. Not yet checked line by line by a second reader. |
| Partner-request drafts | `docs/partners/quant_retired_strategies.md`, `docs/partners/hospital_diagnosis_model.md` | Drafts for Arjun; not sent |
| Record, cut list, claims | `docs/record.md`, `docs/cut.md`, `docs/CLAIMS.md` | Initialised |
| Compute inventory and budget | this report, Sections 4–5 | Done; awaiting approval |

## 2. Novelty verdict

**The integrated definition was not found published.** No work combines all four of these:
- a derivational graph of concepts;
- measurement from inside the system;
- prediction of generalization and of creation;
- validation in both networks and neural populations.

The gate's stop condition is not triggered.

Two works come close enough that the paper must cite them and set itself apart from each:

- **Beckmann and Queloz (2025), arXiv:2507.08017.** They give a qualitative three-tier account of understanding in LLMs, in which "principled" understanding means a compact circuit replaces memorized facts. This is the same idea as this programme's split between grounded and memorized behaviour. It is the main novelty risk.
- **Geiger et al. (JMLR 2025)** and the causal-abstraction line. They supply the causal-use condition, but they test a graph the experimenter provides and do not define understanding.

The following clauses were not found anywhere (details in `docs/novelty.md`):
- the quotient condition on internal realiser states;
- a global concept DAG with minimal parents, defined up to a stated equivalence;
- generality and principles;
- depth of understanding and structural memorization without held-out data;
- the three moves defined relative to the system's own graph;
- creation as a valid new derived node, with false-creation criteria;
- the integration of all of these.

The "predictive" and "validated" clauses of rule 0.1.7 are claims about results that do not exist yet.

**Recommendation for the paper's wording.** State the approved clause with the qualifier "quantitative, derivational". Cite the prior work clause by clause as tabulated.

## 3. Theory skeleton summary (`theory/core.pdf`)

### Setting (§1)

- A system is computed through sites ordered by a fixed linear extension.
- Behaviours are pairs (input distribution, readout) with weights.
- Realiser patches define interventions. They are linear subspaces or dictionary features, and the realiser class is restricted on purpose (see risk R3).
- Access levels A–D are attached to every quantity.
- Untrained-copy, feature-shuffle and random-realiser nulls are attached to every quantity.

### Concept (§2)

A concept is a candidate (site, realiser, value map) that passes three tests.

| Condition | Test | Proposed thresholds |
|---|---|---|
| Quotient | Within-fibre swap effect relative to the unconditioned swap effect (δ_Q); compression; nulls from random partitions and random realisers | δ_Q ≤ 0.10; H(C)/H(X) ≤ 0.25 |
| Derivation | Minimal ε-sufficient parent set from sparse conditional prediction; stability across resamples and seeds; interventional confirmation of each edge | ε_D = 0.05; Jaccard ≥ 0.8; edge interchange ≥ 0.9 |
| Causal use | Interchange intervention accuracy against the downstream derivation, with a graph-free matched-counterfactual variant to break circularity | IIA ≥ 0.90 and ≥ 0.20 above random realisers |

- A candidate that is decodable but fails causal use is recorded as "represented, not used".
- The concept graph has partitions (modulo null sets) as nodes and minimal parent sets as edges. Equivalence absorbs reparametrisation and merges redundant nodes.

### Generality, principles, depth (§3)

- **Generality** is the behaviour-weighted mass of inputs on whose active derivation a node lies, corrected against the untrained copy. A **principle** is a node with generality ≥ τ (two pre-registered levels, 0.25 and 0.5, plus full τ-curves).
- **Depth** has three forms:
  - thresholded: principles form a vertex cut of the input's active derivation;
  - structural: the bottleneck level λ(x) = min over paths of the max generality on the path;
  - interventional: the mediation share γ_τ(x).
- **Behaviour-level scores** are Λ_τ, λ̄ and γ̄_τ.
- **Structural memorization** means no active node of the derivation reaches τ. It is computed without held-out data.

### Moves and creation (§4)

- **Moves.** Lower, sideways and diagonal are defined from value ranges, parent-tuple ranges and coverage. All three are measured on the system's own graph.
- **Creation.** A valid new node is one that, under exposure:
  - passes all three conditions;
  - is not equivalent to any old node;
  - is derived from old nodes;
  - is absent under a matched control exposure.
- **Creation rate and reach** are defined.
- **False creation** has four types: decodable but unused; used but not derived (a new leaf); relabelling; exposure-generic.

### Status of each item

**Proved, not yet checked by a second reader.** These are propositions, not theorems.

| Proposition | Content |
|---|---|
| 2.5 | Exact derivation is refinement in the partition lattice. |
| 2.6 | The minimal parent set is unique under positivity of the candidates, via the intersection property with a deterministic target. |
| 2.8 | Derivation maps are unidentified on unseen parent combinations, and identified there by interchange under modularity. |
| 2.13 | Every statistic is invariant under invertible reparametrisation; for continuous value spaces, isometries only. |
| 2.14 | The graph is well defined in the exact case. |
| 2.17 | Relation to Formal Concept Analysis (below). |
| 3.5 | The bottleneck form of grounding, computable in one pass over the edges. |
| 3.8 | λ ≤ μ, so grounded, partly grounded and memorized are mutually exclusive when ε_M < 1/2. |

**Proposition 2.17 in full.** Let the downstream context have binary downstream readouts as attributes. Under an exact quotient, every FCA extent is a union of concept fibres. Under nominal scaling (attributes closed under complement), the coarsest exact quotient is exactly FCA's object-concept partition. FCA supplies neither derivation nor causal use.

**Conjectures, each with a named experiment:**

| Conjecture | Content | Tested in |
|---|---|---|
| 2.15 | Stability of the approximate graph | V1–V4, K1 |
| 2.18 | Approximate FCA relation | V1, V2 |
| 3.6 | Structural grounding implies interventional grounding | V1, K1 |
| 3.9 | Depth predicts per-input error better than CCGP, softmax and sharpness (**the K2 kill test**) | K2 |
| 3.10 | The memorization verdict predicts held-out failure | V2, K4, K9 |
| 4.3–4.5 | The three moves | V3, K3, K13 |
| 4.8 | Creation predicts success | V5, K13 |

### Obstruction recorded in the theory

Positivity fails whenever the candidate parents include a node together with a function of it, for example a node and its own parents. In that case minimal parent sets can be non-unique (C = C₁ ⊕ C₂). Until a canonical candidate rule is fixed in Phase 1, every quantity is reported as a range over all minimal parent sets.

## 4. Compute inventory

Source: Arjun's probe of the Arkansas HPCC "Pinnacle" (AHPCC) on 2026-10-22, quoted as given. No SSH, job submission or GPU use was made from this session.

**Software constraints**
- Compute nodes run CentOS 7 (glibc 2.17). This caps torch at 2.5.1 (cu124) and transformers below 5. Login nodes run Rocky 9.
- Jobs need `sbatch --export=NONE`.

**GPU partitions (public)**

| Partition | GPUs per node | Nodes | CPU / RAM |
|---|---|---|---|
| gpu06 / gpu72 | 1× Tesla V100-PCIE-32GB (driver 550.163.01) | c1613–1615, c1703–1705 | 32 cores, 192 GB |
| agpu06 / agpu72 | 1× A100 (40 vs 80 GB not yet confirmed) | about 13: c1905–1911, c2005–2009, c2106–2109 | 64-core AMD EPYC 7543, 1 TB |
| qgpu06 / qgpu72 | 4× A100 | 4: c1901–1904 | 1 TB |

**Other partitions**
- **CPU:** comp01/06/72 have 32-core Intel 6130 nodes with 192 GB. There is also himem. condo and pcon06 are owner nodes, not public.
- **Time limits:** \*06 partitions = 6 h; \*72 partitions = 3 days; cloud144 = 6 days, CPU only.

**Account, quotas and storage**
- Account jv002.
- QOS gpu: at least 24 cores per job, at most 384 cores per user.
- QOS comp: at least 32 cores per job, at most 320 per user.
- cloud: 44 cores.
- Scratch `/scrfs`: 10 TB and 10M files per user.

**Interconnect:** not recorded in today's probe. It must be confirmed before any multi-node job is planned. Until then, no plan assumes multi-node model parallelism.

**Discrepancy with the master prompt.** Section 8 assumes about 70 GPUs. The publicly visible total is about 6 V100 + 13 A100 + 16 A100 (qgpu) ≈ **35 GPUs**, half the assumed number. The budget below is sized to what is actually there.

**Share.** Account jv002 is shared with two other projects: the BPU simulator and OMEN/InvProj1. This programme plans on **at most one third of the account's GPU use**, about 11–12 GPUs if every public GPU were free. Public partitions are shared with all users, so the plan assumes about half of that is obtainable on average. That gives roughly **6 A100-equivalents sustained, ≈ 4,000 GPU-hours per month**.

**Consequences for model choice**

| Model size or task | Decision |
|---|---|
| 7–9B models (Llama 3.1 8B, Qwen 2.5/3 7–8B, Gemma 2 9B with Gemma Scope, OLMo 2 7B) | Run on single A100s (agpu). |
| V100s | They lack bfloat16; Gemma 2 and Llama are unreliable in fp16. V100s are reserved for models ≤ 3B (Pythia, OLMo-1B, Chronos, TimesFM), the planted systems, and non-LLM models (Lc0, credit models, RNNs, Vector-HaSH). |
| 70B models (Llama 3.1 70B, Qwen 2.5 72B) | Need a qgpu node (4× A100). On 4× 40 GB (160 GB), bf16 weights (≈ 140 GB) leave too little room for activation patching, so 70B would run in 8-bit. Master prompt §8.2 then requires a check, on an 8B model, that quantization does not change the result. On 4× 80 GB, bf16 fits. The 40/80 GB question must be settled before Phase 2. |
| 405B dense and 671B MoE models | **Not feasible on the public partitions.** They need several 80 GB GPUs per copy and multi-node parallelism over an unknown interconnect. They are removed from the white-box programme. They may appear only through the level-A black-box protocol (§6.6) or through a researcher-access programme that Arjun arranges. |
| Job length | Every job must checkpoint within 6 h so it can use \*06 backfill. The \*72 partitions are for training runs. |
| Storage | Full residual caches for an 8B model are about 50 MB per 200-token input, so 10k inputs would be ≈ 0.5 TB. Only sparse dictionary codes and per-pair patch results are stored, in parquet shards (few files, well under the 10M-file quota). |

## 5. GPU-hour budget per phase

Units are A100-hours. V100-hours count 0.5 for models that fit. All figures are planning estimates. Each phase's estimate is re-derived from its pre-registration and re-approved before launch (rule 0.1.9). Assumptions:
- Concept proposal is restricted to the top ~200 dictionary features per site.
- About 500 interchange pairs per behaviour.
- Nulls (untrained copy, random realisers) cost about as much as the real run again.
- 20% overhead for failures and re-runs.

| Phase | Content | GPU-h (A100-eq.) | CPU core-h | Main drivers |
|---|---|---|---|---|
| 0 | Novelty, theory, inventory | 0 | 0 | — |
| 1 | Planted-answer validation V1–V5 | 200 | 2,000 | V5 fine-tune; sensitivity sweeps |
| 2 | K1 anchor and K2 kill test | 2,800 | 5,000 | ≈ 600 on 7–9B models (4 families × ≈ 12 behaviours, with nulls); ≈ 1,500 for 70B on qgpu (K1 only, 2 models × 3 behaviours); ≈ 200 for sharpness/Hessian and baselines; 20% overhead |
| 3 | K3, K11, K12, K14 | 4,500 | 10,000 | K11 long reasoning traces (7–32B distilled reasoning models) ≈ 1,500; K14 Pythia/OLMo checkpoint sweeps ≈ 1,500 plus own dense-checkpoint training runs (5 seeds) ≈ 300; K12 Lc0 + Boltz/OpenFold ≈ 500; K3 ≈ 400 |
| 4 | Finance K4, K7–K10 | 1,000 | 20,000 | Gradient-boosted credit models are CPU; ≈ 200 cross-sectional return models ≈ 400; Chronos/TimesFM ≈ 300 |
| 5 | Medical, generative (K5, K6), black-box | 2,000 | 3,000 | Diffusion generations and interventions ≈ 1,500; chest X-ray classifiers ≈ 300. Black-box API costs are money, not GPU, and are budgeted separately when the protocol is pre-registered. |
| 6 | Brain: Vector-HaSH, B1–B5, K15 | 1,200 | 15,000 | MoE models (OLMoE on one GPU; Mixtral-class on qgpu) ≈ 800; RNNs and Vector-HaSH ≈ 300; recordings analysis on CPU |
| 7 | Creation K13; blind prediction study | 2,000 | 3,000 | In-context creation on 8B–70B and fine-tuning runs |
| 8 | Paper and release (reproduction runs) | 100 | 500 | — |
| **Total** | | **≈ 13,800** | **≈ 58,500** | About 3.5 months of the planned share if phases ran back to back; longer in practice because of the gates |

**Approval requested now:** only the Phase 1 envelope of **200 A100-h and 2,000 CPU core-h**. Phases 2–8 are shown for planning and will each be submitted with their pre-registration. If the 70B share of Phase 2 has to drop (for example if the A100s are 40 GB and the 8-bit check fails), K1 runs on 7–9B models only, and the paper's scale claim is narrowed to match.

## 6. Partner requests (master prompt §6.7)

Two drafts are in `docs/partners/`. **Neither has been sent.** Arjun chooses the recipients and sends them.

1. **Quantitative firms with academic programmes.** The request is for retired strategies with their decay curves, never live strategies, under NDA. It offers delayed or anonymised data, pre-registration, and review before publication.
2. **A hospital research group.** The request is for a diagnosis model under a data-use agreement. The model stays on the partner's infrastructure if they require it, and access is to weights or activations only. No patient data enters the repository. MIMIC-type data needs credentials that Arjun must confirm first (rule 0.1.8).

Both take months. The programme's finance and medical claims (K4, K6, K7) fall back to public models and data if no partner responds.

## 7. Risks

| # | Risk | Effect | Mitigation |
|---|---|---|---|
| R1 | Beckmann & Queloz (2025) is read as already covering the definition | Weakens the novelty claim | Differentiate clause by clause (`docs/novelty.md`). The quantitative graph, depth, memorization and creation are the separating clauses. |
| R2 | **K2 fails**: depth does not beat CCGP, softmax and sharpness | The programme stops (master prompt §6.2) | This is the designed kill test; the failure is reported in full |
| R3 | Sutter et al. (NeurIPS 2025): unrestricted alignment maps make causal abstraction vacuous | False "concepts" | Restrict the realiser class (linear subspaces, dictionary features) in pre-registration; random-realiser and untrained-copy nulls on every test |
| R4 | Non-unique parent sets wherever positivity fails (Remark 2.7) | Graph ambiguity | Report ranges over all minimal sets; fix a canonical rule on V1 in Phase 1 |
| R5 | Circularity between causal use (which needs the downstream graph) and the graph (which needs accepted concepts) | Estimator bias | Graph-free matched-counterfactual IIA on the first pass; check convergence on planted systems |
| R6 | Compute is half the assumed number of GPUs and shared three ways; A100 memory unconfirmed; interconnect unknown | Fewer or smaller models; slower phases | Budget sized to about 6 effective GPUs; 405B/671B dropped from white-box work; 70B gated on the 40/80 GB check |
| R7 | Cluster software is capped at torch 2.5.1 and transformers < 5 on glibc 2.17 | Some libraries may not install (current sparse-autoencoder tooling, flash-attention builds); newest model architectures unsupported | Pin an environment in Phase 1 and test it on the planted systems; prefer pure-PyTorch SDPA |
| R8 | Single released checkpoints are not independent models (master prompt §6.1) | Weak unit of analysis | Use model families, OLMo/Pythia seeds and checkpoints, and own training runs; state the unit of analysis on every result |
| R9 | Partners do not respond, or data-use terms forbid publication | K4/K6/K7 partner arms are lost | Public fallbacks: Kaggle and Numerai terms respected; public chest X-ray datasets; MIMIC only with confirmed credentials |
| R10 | Brain data access and authorship with the Fiete Lab are not yet agreed | Phase 6 is blocked | Arjun coordinates before any run (master prompt §6.5) |
| R11 | Signed `prereg-phase-X` tags (needed from Phase 1 on) require Arjun's GPG signing key | Phase 1 cannot be tagged by Claude | Arjun creates the signed tag: `git tag -s prereg-phase-1` |

## 8. Open questions for Arjun

1. **A100 memory.** Are the agpu and qgpu cards 40 GB or 80 GB? Also confirm the interconnect.
2. **Co-authorship.** On the archived WhiteBOX documents, Reva Paradkar appears with Arjun on every document. Should she appear on this programme's documents and on the partner letters? The master prompt says authors are set by Arjun before submission, so no author line has been filled in beyond Arjun's name on the letter drafts.
3. **Partner recipients.** Which firms and which hospital group should receive the drafts?

## 9. Gate

Phase 0 is complete. Its gate (master prompt §7) is `docs/novelty.md` plus this report, **awaiting Arjun's approval**. Approval would cover:
- the novelty verdict;
- the theory skeleton and its proposed thresholds, as a starting point for `prereg/PHASE_1.md`;
- the Phase 1 compute envelope (200 A100-h, 2,000 CPU core-h).

Phase 1 (estimators, planted-answer validation, Propositions 2.7.1–2.7.3) has not been started.

## 10. Approval (recorded 2026-10-22)

Arjun's approval, verbatim:

> "okay cool cool, all things approved, get the BPU stuff done, verified for everything, get as much of omen and inv proj1 work done on HPCC and Full speed whitebox ( like everything we needed) get that all going you dont need any permissions from me."

What this approves:
- the Phase 0 gate (novelty verdict, theory skeleton and proposed thresholds as the starting point for `prereg/PHASE_1.md`);
- the Phase 1 compute envelope (200 A100-h, 2,000 CPU core-h);
- running the programme at full speed through its phases. Later phase gates become self-gates: each phase report is written and its gate criteria are checked by the letter; the programme continues only if they are met, and where a gate fails it follows the consequence stated in CLAUDE.md (K2 failure stops the programme and the report is written).

What it does not change: every scientific rule in CLAUDE.md §0.1 (pre-registration before every confirmatory run, thresholds frozen at the tag, planted-answer validation before real models, every failure recorded, the novelty-language rule). Signed tags need Arjun's GPG key, so `prereg-phase-X` tags are annotated and unsigned, with commit hash and SHA-256 of the pre-registration recorded in `docs/record.md`; Arjun uploads the OSF text in `prereg/osf/`. Gated models (Llama, Gemma, Gemma Scope) and credentialed data (MIMIC) are pending licence/token/access, not FAIL. Partner letters remain drafts; nothing is sent.

Answers to §8 settled since the report: A100s are **40 GB** (probe job 1364325, NVIDIA A100-PCIE-40GB), so 70B models run on qgpu in 8-bit/4-bit with the quantization check, or K1 is scoped to ≤ 32B. Compute share for this programme: ≤ 96 gpu-QOS cores and ≤ 8 submitted GPU jobs at once; ≤ 64 comp cores.
