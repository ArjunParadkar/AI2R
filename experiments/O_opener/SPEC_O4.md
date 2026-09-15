# O4.1 — T4 frontier open() (Bamboo Memory): pre-registration

_Written 2026-09-14, **before any O4.1 compute**, from WHITEBOX FINAL_SPEC §3 (the nine steps), §5 (T4 row), §6 (per-problem
criteria), §8.1 (running blocks as components), §8.2 (O4.1) and §12 (Operating Constitution). Thresholds below are frozen at this
commit; changes go to `AMENDMENTS.md` with reasons and timestamps (Constitution #1)._

Models (T4 row of §5): **meta-llama/Meta-Llama-3-8B** (32 layers, H = 32, KV = 8, d = 4096, d_mlp = 14336) and
**google/gemma-2-9b** (42 layers, H = 16, KV = 8, d = 3584, d_mlp = 14336, +Gemma Scope SAEs).

**This block composes existing jobs; it re-implements nothing.** Exactly one new computation is added — the P0 attribution
graph on the frontier model (`bamboo_memory.py --p0`). Everything else in the T4 row is read off jobs that are already
written and pre-registered elsewhere: F12 `hpcc/kitasan_black.py`, C4 `hpcc/sirius_symboli.py`, K8 `hpcc/biwa_hayahide.py`
(→ `experiments/K_darkmatter/smart_falcon.py`), L7 `hpcc/kawakami_princess.py`, and the EAP construction of V7
`hpcc/mejiro_ardan.py`. `bamboo_memory.py --assemble` is the §3 step-9 reporter: it reads those outputs and writes one
scorecard per model. It performs no model compute and it never raises on a missing input — a missing job is a
`not_reached` cell with the reason (Constitution #4: negatives are deliverables).

---

## Step 1 — Gauge (§3.1): F12's identity check

Instrument: `hpcc/kitasan_black.py` (F12), backend `whiteboxtopo.symmetry.llama`, which is the pre-registered exact
renaming set for this family (module docstring): `("mlp", l)` MLP hidden-unit permutation; `("kvhead", l)` KV-head
permutation with its query group; `("qhead", l)` query-head permutation inside a KV group; `("vo", l)` per-KV-head
invertible change of basis on V/O; `("qk", l)` per-RoPE-plane complex scalar `s·e^{iφ}` on Q and `(1/s)·e^{iφ}` on K;
`("ln1", l)`, `("ln2", l)` RMSNorm gain rescalings compensated downstream (Gemma-2's `(1 + w)` storage handled).
**Not** in the group and not sampled: MLP unit scalings (SiLU/GELU gates are not homogeneous) and the final-norm rescale
(tied embeddings). Sampling kinds: `perm`, `scale`, `rot`, `combined`; layers = the F12 audit layers.

`kitasan_black.py` writes `identity.json` = `max |Δ logit|` on the fixed probe for each kind, in the run dtype.

**bf16 floor, pre-registered rule.** §2.4's `CV < 1e-3` is an fp32 number (F11 measured 3–5e-3 on GPT-2 in fp32;
`kawakami_princess.py` already records this). A permutation renaming changes no value — it only re-orders the summations —
so **the `perm` deviation _is_ the floating-point floor of the run**. Frozen rule, evaluated by `--assemble`:

| quantity | rule |
|---|---|
| `float_floor` | `identity["perm"]` (max abs Δlogit under permutations only) |
| kind *k* is **exact at this precision** | `identity[k] <= 10 × float_floor` |
| kind *k* is **not exact in bf16** | `identity[k] > 10 × float_floor` → the kind is reported as approximate; the F12 rows carrying it are read as "approximate symmetry, residual reported" (§2.3), and the recommendation is a re-run at `--dtype float32` if memory allows |
| the gauge step **fails** | `max_k identity[k] > 0.5` (the `does_not_scale` threshold already used by L7) — the renamings are then not usable as a certification at this precision and every cell that depends on certification is downgraded to `partial` with that reason |

Reported, not claimed: the absolute deviations per kind. Claimed: only the exact/approximate classification above.

## Step 2 — Gate, **P0** (§3.2, §6.1): the one new computation

`bamboo_memory.py --p0 --model <M>`. Frozen construction, taken from `hpcc/mejiro_ardan.py` (V7) so that the T4 numbers
are comparable with the Pythia drift numbers, with one documented adaptation for sequential-residual blocks:

* **Probe.** The fixed OpenWebText-10k slice used by every K/V/F job (`experiments/K_darkmatter/eishin_flash.load_tokens`,
  BOS + 125-token sequences), first `ceil(n_tokens / seq_len)` sequences. Pre-registered `--n-tokens 2000` at T4 (16 × 125),
  identical to V7's probe, so anyone can verify the numbers.
* **Nodes.** Attention heads and the MLP of layers `mid-2 … mid+2` — Llama-3-8B: **14, 15, 16, 17, 18** (5 × 33 = 165 nodes);
  Gemma-2-9B: **19, 20, 21, 22, 23** (5 × 17 = 85 nodes).
* **Edges (EAP, mean corruption).** `E[u, v] = Σ_t (mean_u − a_u(t)) · dL/dx_in(v)(t)` with `L` = summed next-token
  log-probability, `a_u` the node's own contribution to the residual stream (head *h*: `z_h @ W_O[:, h-slice]ᵀ`; MLP: the
  MLP module output), `mean_u` its mean over the probe.
* **Adaptation (deviation from V7, stated because it is one).** Pythia's parallel residual lets V7 use one gradient per
  layer. Llama-3 and Gemma-2 are **sequential**: the MLP of layer *l* reads the residual *after* that layer's attention.
  Two read points are therefore taken per layer — the block input (read by the attention nodes) and the input of the
  pre-MLP norm (read by the MLP node). Consequences: within-layer `attn → MLP` edges exist and are scored, and readers in
  the same layer no longer share one gradient. Edges run strictly forward (`(l_u, u) → (l_v, v)` with `l_u < l_v`, plus
  `(l, head) → (l, mlp)`).
* **Gemma-2 sublayer norms, stated.** Gemma-2 applies `post_attention_layernorm` / `post_feedforward_layernorm` to the
  *sublayer output* before adding it to the residual. Contributions are measured pre-norm and rescaled per token by
  `(1 + w) / rms(sublayer_out)`. Because that factor is a property of the sum, the rescaled split is an **exact
  decomposition of the realised residual contribution** — checked on a randomly-initialised Gemma-2 config, where the
  node contributions sum to `h_{l+1} − h_l` to `2.8e-7` relative (the Llama branch checks to `3.9e-7`). It is
  first-order only as a **counterfactual**: mean-corrupting one node would also move `rms(sublayer_out)`. The run records
  `gemma_sublayer_norm_split: true` and the cell provenance carries this sentence; Llama-3 and GPT-2 record `false`.
* **Measures** (`whiteboxtopo.legibility.graph` via `whiteboxtopo.graphs`, no per-model tuning): symmetrise `|E| + |E|ᵀ`,
  threshold to edge **density 0.1**, then Leiden `Q` (10 restarts), min-fill **treewidth upper bound**, and effective
  sparsity `1 − PR/N` of the node-importance vector `Σ_j |E[i, j]|`. Densities 0.01 / 0.05 / 0.1 are all recorded;
  **0.1 is the pre-registered one** (AMENDMENTS #5d of O1.2).
* **Null:** degree-preserving rewiring, `whiteboxtopo.graphs.rewired_null(A, n=20)` — mean and sd of `Q` over 20 rewirings
  of the thresholded graph (§6.1's null, verbatim).
* **Verdict, frozen (AMENDMENTS #5d of O1.2):** `tractable` iff `treewidth_ub ≤ 8`; `marginal` iff `≤ 32`; else
  `intractable`. The verdict is **by the treewidth bound only** — it is the parameter of the FPT conjecture (§2.14). `Q` is
  reported against the rewiring null and is *not* part of the verdict: O1.2 found the old `Q > 0.3` clause mislabels sparse
  star-shaped attribution graphs.
* **Cell status:** `certified` iff the verdict is `tractable`; `failed` otherwise (the gate did not pass — §3.2: "if
  intractable: report and proceed with partial extraction only"). The status word records whether the model passed the
  gate, never whether the instrument ran.
* **Reference, not a prediction.** The V-block phase diagram is the reference placement (§8.4), but V8 found **no calibrated
  structure → tractability map** (`whiteboxtopo.open.GATE_RULE`, `legibility.profile.V8_CALIBRATION_ATTEMPT`). The verdict is
  therefore reported as a *placement on the pre-registered FPT rule*, not as a calibrated prediction of legibility. Stated
  in every P0 cell's provenance.

**Prediction on the record (so that it can fail).** A 165-node / 85-node EAP graph thresholded to density 0.1 is dense
enough that we expect `treewidth_ub ≫ 32` on both frontier models, i.e. **P0 = intractable**, and the T4 row of the grid to
open only partially. If it comes back `tractable`, that is a surprise and is reported as one.

## Steps 3 and 4 — Extract and Certify (§3.3, §3.4): the F12 audit table

Read from `hpcc/results/f12_*/F12_audit_table.csv` (written by `kitasan_black.py`; one row per layer × metric × renaming
kind with `median_cv`, `frac_inv`, `median_cv_aligned`). The classification rule is `whiteboxtopo.invariants.check`'s,
verbatim (`INVARIANT_CV = 0.01`), applied to the `combined` kind:

| verdict | rule |
|---|---|
| **invariant** | `median_cv(combined) < 0.01` |
| **covariant (aligned)** | `median_cv(combined) ≥ 0.01` and `median_cv_aligned(combined) < 0.01` — the quantity is a well-defined function of the *unit labelling*, recovered by applying `g⁻¹` to the unit axis; §2.4's "normalised by its character", here the character is the permutation itself |
| **artifact** | otherwise — discarded (§3.4) |

The scorecard reports the counts per class and the full per-metric table by reference (`F12_audit_table.csv`). No T4 pass
threshold exists in §6 for the audit table; it is a **deliverable (§10.3), reported, not claimed**.

## Step 4 (continued) — **P1**: the C4 certified feature inventory

Read from `hpcc/results/c4*/C4_summary.json` + `C4_inventory.csv` (`sirius_symboli.py`). C4's certification rule, frozen
there: a feature of the k-sparse MLP-hidden dictionary is certified iff its blind activation-profile match is `≥ 0.95`
under **every** exact renaming sampled. `C4_summary.json` carries `alive`, `certified_fraction_alive`, and a
`probe_verdict` per renaming kind.

**P1 status, frozen here:**

| status | rule |
|---|---|
| `certified` | `certified_fraction_alive > 0` **and** `probe_verdict[k] == "invariant"` for all four kinds |
| `partial` | `certified_fraction_alive > 0` and some `probe_verdict[k] == "artifact"` |
| `failed` | `certified_fraction_alive == 0` |
| `not_reached` | no C4 run for this model |

Value = the certified fraction of alive features. §6.2 sets **no numeric threshold at T2–T4** ("report certified
fraction") — so the fraction is *reported*; the only thing *claimed* is that the certified features are invariant under
the renamings actually sampled. Stated in the cell: a residual-stream Gemma Scope SAE cannot be gauge-audited at all (it
does not see the MLP-unit permutation) — C4 fits its own dictionary on the MLP-hidden coordinates for exactly this reason,
so the P1 inventory is **not** the Gemma Scope inventory.

**Llama-3-8B:** `hpcc/sbatch/c4_scale.sbatch` runs Gemma-2-9B only. Unless a second C4 run is submitted
(`sirius_symboli.py --model meta-llama/Meta-Llama-3-8B` — the script is family-generic, same `symmetry.llama` backend),
P1 for Llama is **`not_reached`**, with that sentence as the reason.

## Step 7 — **γ_cov** (§3.7, §6.5 tool-side P4): the K8 Gemma Scope reconstruction residual

Read from `hpcc/results/k8*/cache/L*.npz` (`biwa_hayahide.py`: Gemma-2-9B + `gemma-scope-9b-pt-res-canonical`, width 131k,
layers 9 / 20 / 31, `E = x − SAE(x)` on 400 × 125 tokens) and `K8_reconstruction.csv` (`fvu` per layer).

* γ_cov = `whiteboxtopo.reconstruct.coverage_gap(E_sub, q=0.99, seed=0)` — the §2.13 quantity: the fraction of
  `tr Σ` in eigenvalues above the Tracy–Widom-corrected Marchenko–Pastur edge, `σ²` fitted from the median eigenvalue.
* `E_sub` = a fixed 20 000-row subsample (`np.random.default_rng(0)`, without replacement) of the layer's residual, so
  `n = 20 000 > d = 3584` and the MP bulk exists (the function's stated failure boundary).
* **Nulls (the K-block MP null, verbatim, both returned by `coverage_gap`):** `gamma_null_gaussian` (i.i.d. Gaussian with
  the same per-coordinate marginals) and `gamma_null_shuffle` (each coordinate permuted independently). The honest
  headline alongside γ_cov is `gamma_excess = γ_cov − max(nulls)`; a γ_cov that is not clearly above both nulls is not a gap.
* **Headline layer, frozen before compute:** the K8 layer closest to the model's middle — **layer 20** of Gemma-2-9B's 42.
  All three layers are reported in the scorecard; only layer 20's γ_cov enters `1 − γ_cov`.
* γ_cov is computed by `--assemble` on the CPU from the cached residual; no GPU, no re-run of K8.

**Llama-3-8B: `not_reached`.** There is no public SAE for Llama-3-8B in this project's stack (Gemma Scope is Gemma-only,
and `sae_lens` carries no canonical Llama-3-8B release that K8 is written against). Without a decomposition there is no
residue to measure, so γ_cov is unknown and **`degree_of_opening` is reported as `unknown`, never as 0 or 1** — an unknown
gap is never reported as a zero gap (`whiteboxtopo.open.Scorecard`). If an SAE becomes available, `biwa_hayahide.py
--model meta-llama/Meta-Llama-3-8B --release <...>` is the job to run; nothing else changes.

## Step 9 — Report (§3.9, §12.9)

`bamboo_memory.py --assemble --model <M>` writes `hpcc/results/o41/scorecard_<slug>.json` and `.md`:

* JSON layout = the Block-O convention that `experiments/O_opener/symboli_kris_s.py` (O5) consumes: top-level `"cells"`
  keyed `P0 … P7` (each `{status, value, criterion, provenance, evidence}`) plus `"scorecard"` (a
  `whiteboxtopo.open.Scorecard.to_dict()` carrying `gamma_cov` and `degree_of_opening`), plus `specimen`, `kind`, `model`.
  The audit/gauge/R cells (`gauge`, `audit`, `R4`) live in `"cells"` alongside `P0 … P7`, as in `meisho_doto_*.json`.
* **Wording, §12.9:** the word *complete* is used only where `γ_cov < 0.05` and R5–R7 pass. R5–R7 are not run at T4, so the
  markdown says **"opened to degree 1 − γ_cov = …"** and nothing else; where γ_cov is unknown it says
  **"degree of opening: unknown (no certified decomposition reached)"**.
* Missing input → `not_reached` with the reason and the exact command that would produce it. **`--assemble` never raises.**

---

## The T4 grid row, cell by cell (pre-registered reachability)

| cell | reachable at T4 with existing jobs? | source | pre-registered status rule / reason if not |
|---|---|---|---|
| **P0** is it explainable | **yes, both models** (new compute, this block) | `bamboo_memory.py --p0` (EAP graph, V7 construction) | `certified` iff `treewidth_ub ≤ 8` at density 0.1; `failed` (verdict `marginal`/`intractable`) otherwise. Null: degree-preserving rewiring |
| **P1** what it learned | **Gemma only** (C4 is submitted for Gemma-2-9B) | C4 `sirius_symboli.py` | certified fraction of alive dictionary features; Llama `not_reached` unless a C4 run is submitted for it |
| **P2** why this decision | **no** | — | `not_reached`: needs attention-pattern-conditional path products (§6.3) — a frozen-attention re-implementation of the Llama/Gemma forward (RoPE + GQA + Gemma-2 logit soft-capping) plus exact-invariance verification of `A(x)` over a gauge orbit of a 9B model. No such code exists in `whiteboxtopo` (`invariants.path_product` is the ReLU-MLP instrument) and it is not cheap: ~1 new module + O(300) renamed forward passes at ≥ 40 GB. Stated as the missing code, with its cost |
| **P3** when bias entered | **no** | — | `not_reached`: needs training checkpoints (§6.4 / §3 step 3). Neither Meta-Llama-3-8B nor Gemma-2-9B publishes intermediate checkpoints. This is **structural, not a compute shortage** — it is O3.1's row (Pythia, 154 revisions), not T4's |
| **P4** what it does not know | **Gemma: `partial` (tool side only); Llama: `not_reached`** | K8 → `coverage_gap` | §6.5 has two halves. The *tool-side* P4 — γ_cov of a decomposition (K block) — is reached on Gemma. The *model-side* P4 — diffusion embedding + Nyström OOD distance with the softmax-confidence baseline, AUC > 0.9 — has no T4 job and is `not_reached` within the cell, stated |
| **P5** where it will fail | **no** | — | `not_reached`: `ρ(x) = ‖Q_h(x)‖ · Lip(x)` needs a local second fundamental form and the spectral norm of the path product per input; no path product at 9B (see P2) and no T4 job. Cost: the curvature field alone is O(n · k) k-NN PCA on 9B activations plus a perturbation sweep at 6 radii |
| **P6** memorising vs learning | **no** | — | `not_reached`: `whiteboxtopo.signals.memorization_signal` needs a family calibration, and the only LM calibration that ships (`memorization_lm_transformer.json`) is `arch: gpt2`, `layer: L12`, `n_models: 8`, whose own meta note reads *"Cross-family transfer FAILS (sign flip); never apply to another family."* M4 (`admire_vega.py`) measures that cross-family failure; it contains no 8–9B arm. Applying the gpt2 calibration to Llama-3-8B or Gemma-2-9B would be a number with no meaning, so it is not computed |
| **P7** how representations evolve | **no** | — | `not_reached`: needs checkpoints (same reason as P3). O3.1's row |
| `gauge` (step 1) | **yes, both, if F12 is run for that model** | F12 `identity.json` | the bf16-floor rule above. `f12_frontier_audit.sbatch` currently runs Gemma-2-9B only; Llama needs `kitasan_black.py --model meta-llama/Meta-Llama-3-8B` |
| `audit` (steps 3–4) | **yes, both, if F12 is run for that model** | F12 `F12_audit_table.csv` | invariant / covariant(aligned) / artifact counts at `INVARIANT_CV = 0.01` |
| `R4` / γ_cov (step 7) | **Gemma only** | K8 → `coverage_gap` | γ_cov with both MP nulls; headline layer 20 |
| steps 5, 6, 8 (count, glue, verify) | **no** | — | step 5 needs independently-trained seeds of an 8B model; step 6 needs an EAP score matrix over a prompt cover at 9B (L7 already lists gluing under `not_covered`); step 8 needs a reconstruction to verify, and none is built at T4 (§8.2 O4.1 says "partial extraction") |
| `L7` timing | **yes (Gemma)** | `L7_timing.json` | carried into the scorecard as the feasibility record + `does_not_scale` list; not a P-cell |

**Expected T4 row, stated in advance:** P0 `failed (intractable)` on both; P1 certified-fraction reported on Gemma;
P4 `partial` on Gemma; every other cell `not_reached` with the reasons above; degree of opening = `1 − γ_cov` on Gemma
(layer 20), `unknown` on Llama. If the row comes out better than that, the surplus is a result; if worse, it is reported
unchanged.

## What is reported vs. what is claimed

**Reported** (numbers, no claim attached): the identity deviations per renaming kind; `Q`, its rewiring null, the treewidth
bound and the sparsity at all three densities; the full F12 audit table; the C4 certified fraction and per-kind probe
verdicts; `fvu` and γ_cov at every K8 layer with both nulls; L7's stage timings and `does_not_scale` list; the wall time
and cost of the P0 job itself.

**Claimed** (and therefore killable): (i) the P0 *verdict* under the frozen treewidth rule; (ii) the exact/approximate
classification of each renaming kind under the frozen bf16-floor rule; (iii) the invariant / covariant / artifact
classification of each audited quantity at `INVARIANT_CV = 0.01`; (iv) that the C4-certified features are invariant under
the renamings sampled; (v) γ_cov at the headline layer **only if** `gamma_excess > 0` against both MP nulls; (vi) the
degree of opening `1 − γ_cov` for Gemma-2-9B, with the §12.9 wording and never the word "complete".

**Not claimed anywhere in this block:** that the T4 models were opened; that the un-run cells would pass; that a P0
`intractable` verdict predicts anything about legibility (V8 found no such calibration); that γ_cov measured on one
SAE decomposition is a property of the model rather than of that decomposition.

## Kill tests

* **K-O4-1 (floor).** If `identity["perm"] > 0.5`, the bf16 run cannot certify anything and *every* cell that rests on
  certification is downgraded to `partial` with that reason. A floor that large is a result about the precision, not
  about the model.
* **K-O4-2 (γ_cov null).** γ_cov is only claimed where `gamma_excess = γ_cov − max(gamma_null_gaussian,
  gamma_null_shuffle) > 0`. If the nulls match or exceed it, the cell is `failed` and the residue is recorded as
  MP-indistinguishable — the E6 kill.
* **K-O4-3 (P0 null).** If the measured `Q` does not separate from the rewiring null, `Q` is reported as
  null-indistinguishable. The verdict does not move: it never depended on `Q`.
* **K-O4-4 (provenance).** `--assemble` refuses to attribute a result to a model it was not computed on: every input file
  carries its own `model` field, and a mismatch is a `not_reached` cell naming both strings, never a silent borrow.
