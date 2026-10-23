# Pre-registration, Phase 2: K1 (ground-truth derivation) and K2 (the kill test)

WhiteBox E&E. DRAFT — not frozen until the annotated tag `prereg-phase-2` exists; the commit hash and SHA-256 of this file at the tag are recorded in `docs/record.md`. Nothing confirmatory runs before the tag. Thresholds never change after it.

Status of dependencies: Phase 1 gate (`reports/phase1.md`) must show P1a–P1f, P2 and P3 passed for the estimators used here; if not, this pre-registration does not run.

## 0. Access level, unit of analysis, what is and is not claimed

- Access level (C): full internals, activation patching (realiser patches at residual-stream sites).
- Each released checkpoint is a single copy of a model. The unit of analysis **within a model** is the input (K2) or the node/edge (K1), and no result generalizes across training runs of that model. Across models the unit is the **model × task cell**; the cross-model summary treats cells as the units (family structure reported).
- Pending, not FAIL: Llama 3.x and Gemma 2 / Gemma Scope cells (pending licence/token); medical-chain arm of K2 (pending access: no public, ungated pretrained diagnosis model with a known clinical chain exists — `docs/data_sources.md`; a DDXPlus-trained model is run as an exploratory, non-claim cell, §5).

## 1. Models (ungated open weights, exact revisions resolved and logged at load)

| Family | Models | Precision |
|---|---|---|
| Qwen2.5 | Qwen/Qwen2.5-7B, -14B, -32B | bf16 (7B, 14B); 32B in 8-bit (bitsandbytes 0.42) with the quantization check (§4.4) |
| OLMo-2 | allenai/OLMo-2-1124-7B, -13B | bf16 |
| Pythia | EleutherAI/pythia-1.4b, -2.8b, -6.9b, -12b | bf16 |
| Mistral | mistralai/Mistral-7B-v0.1 | bf16 |

Development model (excluded from every confirmatory list): Qwen/Qwen2.5-1.5B, Qwen2.5-0.5B, pythia-70m/160m.

**Specimen gate** (master prompt lesson 1.5), applied per model × task before any K1/K2 statistic is computed: per-behaviour training-distribution accuracy ≥ 0.6 on every kept behaviour, and on the held-out set at least 30 errors and 30 correct inputs (K2 only). A cell failing the gate is recorded as "specimen gate failed" with its accuracies and contributes no K1/K2 statistic.

## 2. Tasks and behaviours

| Task | Format | Behaviours | Known derivation (K1 ground truth) |
|---|---|---|---|
| add3 | 3-digit + 3-digit, 3 fixed few-shot examples; digits one token each (format per tokenizer, `bench/tasks.py`); answers 4 digits, carry-out fixed to 1 (leading digit 1), all carry patterns of the lower columns equally often | out0, out1, out2 (result digits) | a_i, b_i → s_i = a_i + b_i; c_0 ← s_0; c_1 ← (s_1, c_0); r_0 ← s_0; r_i ← (s_i, c_{i−1}); out_j ← r_j |
| mul21 | 2-digit × 1-digit | result digits | partial products p_i = a_i·b; carries; result digits (as in `MultiplicationTask.labels`) |
| agree | subject–verb number agreement with attractors (Marvin & Linzen style templates, own grammar) | verb (2-way per example) | subj_num → verb |

Probe (training-distribution) set: n_train = 1500 per cell; held-out set: n_test = 1000 problems disjoint from the probe set. Seeds: 0 (all cells), plus seeds 1, 2 for the Qwen2.5-7B and OLMo-2-7B cells (resampling stability of the graph).

Sites: every second layer in the first two thirds of the network and every layer in the last third, × the task's roles. On the development model (Qwen2.5-1.5B, 2-digit addition) full-state patches showed that answer positions become causal only in a narrow late band (about L22–26 of 28), while operand positions stay causal until then. Candidates: the probe proposer (Phase 1-validated) with the task's known-algorithm labels and its wrong hypotheses (`w_i`, `q_i` for arithmetic; decoys for agreement).

## 3. K1: ground-truth derivation

**Hypothesis.** In open LLMs that can add, the concept graph recovered by the pipeline contains the known algorithm's intermediate variables and its derivation edges, and generality ranks the shared intermediate variables (column sums, carries) above operand-specific variables.

**Statistics (per model × task cell, add3 and mul21):**
- K1a node recall: fraction of the known intermediate variables (add3: s_0..s_2, c_0, c_1, r_0..r_2) recovered as concept nodes (label VI ≤ 0.1).
- K1b edge recovery: of the known edges whose two endpoints are both recovered (or are inputs), the fraction present in the recovered graph, counting an edge P → C as present if P is an ancestor of C within the recovered graph through at most one unlabelled intermediate node.
- K1c generality order: mean generality of recovered carry and column-sum nodes minus mean generality of operand-specific nodes (input nodes a_i, b_i, computed with the same on-path definition); positive = pass for that cell.
- K1d wrong hypotheses accepted: fraction of wrong-hypothesis labels matched by an accepted node.
- Nulls: untrained copy of each model ≤ 7B (accepted nodes, FP rate); random realisers inside every test.

**Thresholds (pre-registered).** K1 passes for a cell if K1a ≥ 0.6, K1b ≥ 0.7, K1c > 0 and K1d ≤ 0.2, and its untrained-copy null accepts ≤ 2% of candidates. **K1 (the anchor) passes** if at least one model family has at least one cell passing on add3 (master prompt §10: "in at least one open LLM family"); the result is reported for every cell.

## 4. K2: depth predicts per-input error (the kill test)

**Hypothesis (Conjecture 3.9).** On held-out inputs, the depth of understanding of the input, λ(x) (bottleneck grounding level, Prop. 3.5; per input the minimum over the task's behaviours), predicts whether the model answers correctly better than (i) Bernardi's CCGP (our pre-registered per-input adaptation, `whitebox_ee.baselines.ccgp_per_input`), (ii) sharpness (Hutchinson trace of the per-input loss Hessian over the last 2 decoder blocks, 32 probes, first 300 held-out inputs; the score is −trace), and (iii) max-softmax (minimum over behaviours of the restricted-class max probability).

**Baseline implementations (frozen):**
- **CCGP per input** (`whitebox_ee.baselines.ccgp_per_input`, called per variable in `bench/k12.py`). The variables are the known-algorithm variables (s_i, c_i, r_i; p_i and k_i for multiplication; subj_num, attr_num and match for agreement). Conditions are the task-variable combinations of each variable's own column: (a_i, b_i, c_{i−1}) for arithmetic and (subj_num, attr_num) for agreement. States are reduced to the top 64 principal components fitted on training rows. Sites are every 4th exposed layer at all roles; the best site is chosen on training rows only, with 3 random splits. The per-input score is the mean over variables of the signed decoder margin for the input's true value, from decoders that never saw the input's condition.
- **Sharpness:** Hutchinson, last 2 blocks, 32 probes, first 300 held-out inputs, model dtype.
- **Max-softmax, entropy and margin:** restricted to the task's class tokens, minimum over behaviours.

**Discovery settings (frozen):** `Thresholds()` defaults as validated in Phase 1, except `n_random_realisers` = 10 (cost; the 95th-percentile η is estimated from 10 norm-matched random realisers). n_pairs = 512, activity rows = 500, partners = 8.

**Analysis code (frozen):** `bench/analyse_k12.py` computes every K1 and K2 statistic and the decision rule.

**Statistic.** For each passing-gate cell and score S, AUROC(S → correct) on the held-out inputs. Primary contrast: ΔAUROC(λ, B) = AUROC(λ) − AUROC(B) for B ∈ {CCGP, −sharpness, max-softmax}, per cell with a paired bootstrap 95% CI over inputs (2000 resamples). Secondary: the same for γ_0.25 and for entropy and margin; per-behaviour AUROCs.

**Decision rule (the kill criterion).** Across the cells that pass the specimen gate (units = cells), compute the mean ΔAUROC(λ, B) with a 95% bootstrap CI over cells (resampling cells; families reported) for each of the three named baselines B; Holm-correct the three one-sided tests of mean ΔAUROC > 0 (p from the bootstrap). **K2 passes** if, for all three baselines, the mean ΔAUROC is positive with the Holm-adjusted lower confidence bound above 0, **and** depth beats CCGP (ΔAUROC CI lower bound > 0) in at least half of the cells individually. If fewer than 4 cells pass the specimen gate, K2 is "not decidable" (not pass) and Phase 2 reports so.

**If K2 fails, the programme stops** (master prompt §6.2): no Phase 3–8 confirmatory run starts, and `reports/phase2.md` reports the failure in full together with K1 and every secondary analysis labelled as such.

**Pre-registered disagreement analysis (Prop. P7).** For every cell: the share of held-out inputs where λ and CCGP disagree in rank half (above vs below their medians), split into (a) high CCGP / low depth and (b) low CCGP / high depth, with error rates in each; reported, no verdict.

### 4.3 Answer-funnel diagnostic (from Phase 1, P6b)

For every held-out input and behaviour, the node that attains λ (the bottleneck node) is recorded. A node is **answer-like** if it is a direct parent of the behaviour's output and, on the training probe set, H(out | C) ≤ 0.1·H(out). The diagnostic reports, per cell:
- the fraction of held-out inputs whose λ is attained at an answer-like node;
- AUROC(λ) recomputed on the inputs whose λ is not attained at an answer-like node.

This is secondary and carries no verdict. If more than half of a cell's inputs are funnel-grounded, the cell's K2 result is flagged "answer-funnel dominated" wherever it appears.

### 4.4 Quantization check (for the 32B 8-bit cells)

Qwen2.5-7B add3 seed 0 is run in bf16 and in 8-bit. The check passes if the two recovered graphs have node-label sets with Jaccard ≥ 0.8 and the two λ vectors on held-out inputs have Spearman ≥ 0.8. If it fails, the 32B cell is not run in 8-bit and the scale claim is narrowed to ≤ 14B.

## 5. Exploratory (labelled exploratory everywhere; never upgrade a verdict)

- DDXPlus-trained diagnosis model (MLP, our training; evidence → our pre-defined ICD-10 chapter-block layer → pathology): the same K2 statistic; reported as exploratory because the model is project-trained and the intermediate layer is our definition.
- Additional seeds, layer_step = 1 for one 7B cell, and Qwen2.5-72B in 4-bit if compute allows.

## 6. Compute estimate

Per cell (7B, one task): discovery 1.5–3 A100-h, K2 activity 0.5 A100-h, sharpness 1–2 A100-h. 14B ≈ 2×, 32B (8-bit) ≈ 5×. Cells: 10 models × 3 tasks + 4 extra seeds + 6 untrained nulls + 1 quantization cell ≈ 41 cells ≈ 220 A100-h ± 50%, plus 2,000 CPU core-h (CCGP SVMs). Within the Phase 0 planning figure (2,800 A100-h for Phase 2).
