# Block O2 — the opener at T2 (GPT-2 small): pre-registration

_Written 2026-09-14, **before any O2 compute**. Authority: WHITEBOX FINAL_SPEC §3 (open() steps 1–9), §5 (T2 row),
§6 (P0–P7 certified procedures), §7 (R1–R8), §8.2 (O2.1, O2.2), §12 (Operating Constitution). Conventions inherited from
`SPEC_O.md` and `AMENDMENTS.md` (Block O). Thresholds below are frozen at this commit; any change goes to
`AMENDMENTS.md` with its reason and its timestamp, never to this file._

_Runs: **O2.1 Satono Crown** = `satono_crown.py`, the GPT-2 open(); **O2.2 Narita Top Road** = `satono_crown.py --audit`,
the F11 audit (`experiments/F_gauge/narita_brian.py`, results already on disk) carried into the grid with no recompute._

Inherited discipline that this SPEC obeys without restating it: kill before believe (§12.2); baseline or no claim (§12.3);
negatives are deliverables (§12.4); every shipped cell carries its failure boundary (§12.8); "complete" is used only where
γ_cov < 0.05 and R5–R7 pass (§12.9) — **it is not expected to apply anywhere at T2** (§5: "γ_cov measured (expected
substantial)"). AMENDMENTS #5a is in force: an accuracy-based kill can be tautological, so every kill at T2 is measured on
a *function-level* quantity (logit difference, faithfulness), never on argmax accuracy.

---

## 0. Model, precision, device

GPT-2 small (124 M), Hugging Face `GPT2LMHeadModel.from_pretrained("gpt2")`, fp32, `eval()`, `cuda` (fallback `cpu`).

The HF class is fixed by the gauge backend: `whiteboxtopo.symmetry.gpt2.handles()` tests
`type(model).__name__ == "GPT2LMHeadModel"`, and every gauge sample is produced by `symmetry.gpt2.apply`. TransformerLens
is present in the run environment but is **not** used to hold the weights: a `HookedTransformer` is a different parameter
layout and no gauge backend exists for it. This is the same choice F11 (`narita_brian.py`) made.

**Mid layer** = block 6 (`hidden_states[7]`, the residual stream after block 6), the layer F11 audited — so the O2.1 glue,
R4 and P6 clouds and the O2.2 audit rows describe the same place in the same model.

**Node set** (the ablatable objects of every circuit cell below): 12 layers × 12 attention heads = 144 head nodes, plus
one node per layer for the whole MLP block = 12, **N = 156 nodes**. Individual MLP neurons (3072/layer) are *not* nodes:
36 864 nodes would make the two-pass greedy prune of `discover_circuit` cost ≈ 74 000 forward passes per circuit and 20×
that under certification. This is a stated coarsening of the node granularity, not a finding, and it is the failure
boundary of every P1/R3/R5/R6 number below: a circuit certified here is certified *at head+MLP-block granularity*.

Node *j* of the transformed model equals node *p[j]* of the base model for a head permutation *p* (the convention of
`symmetry.gpt2.apply`, recorded by F11's `topk_align`); MLP-block nodes are fixed by every element of G (the `("mlp", l)`
permutation renames hidden units *inside* a block, which this node granularity does not resolve).

---

## 1. The three behaviours B (task, input distribution D_B, metric)

Every prompt in a task has the same token length by construction, and the quantity is always read at the **last
position**. The metric is always a **logit difference** — a difference of logits at one position, which is invariant
under the softmax's additive gauge and is the standard metric for all three circuits in the literature.

For a batch of logits `L (B, V)` and a target object `(idx (B, K), w (B, K))`, the metric is
`m = mean_b Σ_k w[b,k] · L[b, idx[b,k]]`. All three tasks are instances of this one function.

### 1.1 IOI (indirect object identification) — Wang et al. 2022

Templates, both used, half the prompts each (ABBA and BABA differ only in the order of the first two names):

* ABBA: `Then, {A} and {B} went to the {PLACE}. {B} gave a {OBJ} to`  → answer ` {A}`
* BABA: `Then, {B} and {A} went to the {PLACE}. {B} gave a {OBJ} to`  → answer ` {A}`

`{A}, {B}` are drawn without replacement from a fixed list of first names that tokenise to exactly one token with a
leading space; `{PLACE}`, `{OBJ}` likewise from fixed single-token lists. IO = `{A}` (appears once), S = `{B}` (appears
twice). **Metric: logit(IO) − logit(S)**, i.e. `idx = [IO, S]`, `w = [+1, −1]`.

Corrupted counterpart (**ABC**, the pre-registered corruption): the second occurrence of `{B}` is replaced by a third
name `{C}` not in `{A, B}`, so no name is repeated and the S-inhibition mechanism has nothing to inhibit. Reported as the
behaviour's own contrast (clean vs corrupted logit difference); it is **not** the ablation baseline (see §1.4).

### 1.2 Induction — random-token repeated sequences

`R` = L = 32 token ids drawn uniformly without replacement from ids 1000–20000 (avoids the byte-level and special-token
ranges). Sequence = `[BOS] + R + R[:-1]`, token length T = 2L = 64; the model must predict `R[-1]` at the last position
from the repeat. **Metric: logit(R[-1]) − logit(D)** where `D` is a distractor token drawn from the same range and not in
`R` (`idx = [R[-1], D]`, `w = [+1, −1]`).

Corrupted counterpart: the second copy is replaced by fresh random tokens (no repeat anywhere), so induction is
impossible; the logit difference must collapse.

### 1.3 Greater-than — Hanna et al. 2023

`The {NOUN} lasted from the year 17{XX} to the year 17` with `{NOUN}` from a fixed single-token list
(war, famine, siege, plague, drought, revolt, feud, riot, strike, protest) and `XX` uniform in 02…98. The answer is a
two-digit year `YY`; the mechanism is "YY > XX".

**Metric (a logit difference between two sets):**
`m = mean_{YY > XX} logit(YY) − mean_{YY ≤ XX} logit(YY)`, over the K = 99 two-digit year tokens `"00"…"99"` that GPT-2
tokenises as a single token (verified at run time and recorded; tokens that are not single are dropped from both sets and
the count is recorded). In the `(idx, w)` form: `w = +1/|{YY>XX}|` on the greater set and `−1/|{YY≤XX}|` on the other.

Corrupted counterpart: `XX` replaced by `01` (the standard corruption — every year is then "greater", so the comparison
carries no information).

### 1.4 Sizes and splits (frozen)

| | discovery / cover split | held-out split (R3, R5) | gauge samples |
|---|---|---|---|
| full run | **256** prompts | **256** prompts (disjoint seeds) | **n_gauge = 20** |
| `--smoke` | 16 | 16 | 2 |

**Mean ablation** (`means`): per node, the mean of that node's output over the **clean discovery split**, taken per
position (heads and MLP blocks give a `(T, d_model)` mean). This is the convention of `meisho_doto.py` /
`legibility.discovery` and the wording of §7 R3 ("the complement mean-ablated"). It is *weaker* than mean-ablation over
the corrupted distribution (the clean mean still carries task structure), which is recorded here as a known
conservatism: it makes circuits *smaller*, not larger, and it is the same instrument at T1 and T2.

`m_corr` (the zero point of faithfulness) is the metric with **all** nodes mean-ablated; `m_clean` with all kept;
faithfulness `F(keep) = (m(keep) − m_corr) / (m_clean − m_corr)` (`legibility.discovery.faithfulness`, unmodified).

---

## 2. Step 1 — Gauge (FINAL_SPEC §3.1, §2.3)

`symmetry.gpt2.describe(model)` records the group: per layer, head permutations `("head", l)`, within-head GL(d_h)
rotations `("rot", l)` (Q·R, K·R^{−T}, V·R, R^{−1}·O), MLP hidden-unit permutations `("mlp", l)`, and the LayerNorm gain
rescalings `("ln1", l)`, `("ln2", l)` (γ, β scaled by a, the consuming weight rows divided by a).

Measured for each kind ∈ {perm, scale, rot, combined}, on the task's own prompts, over 5 samples per kind:
`dev = max |logits(g·θ) − logits(θ)| / max|logits(θ)|`.

* **Pass (certified):** every exact kind has `dev < 1e-4` (the fp32 realisation of "exact"; F11 measured ~1e-7 relative
  on the raw logits with the same backend).
* **Recorded, not sampled — the approximate symmetries and their residual** (§2.3: "the implementation distinguishes
  exact from approximate symmetries and reports the latter's residual"): (i) **MLP scaling through GELU** — scale
  `c_fc` rows and bias by a, divide `c_proj` rows by a; exact for ReLU, *not* for GELU. (ii) **`ln_f` rescaling** — the
  `lm_head` is tied to `wte`, so the compensating rescale would also change the input embedding. Both are applied once
  at a = 1.5 and their residual `dev` is reported as the measurement of how far they are from being gauge. A residual
  indistinguishable from the exact kinds would be a finding (it is not expected).
* **Hidden symmetries** (Jacobian corank, §2.2) are **not** computed: `dim ker dR` for 124 M parameters is out of reach
  on this hardware. Recorded as `not_reached` inside the gauge cell with that reason; every certification below is
  therefore "certified against the *known* group G", which is the honest statement.

---

## 3. Step 2 — P0, is it explainable (FINAL_SPEC §6.1, §2.14; AMENDMENTS #5d)

Attribution graph Γ over the 156 nodes plus a logit node: `E[i, j] = |Σ_batch (node_i − mean_i) · ∂ s_j / ∂ node_i|`
where `s_j = Σ (node_j − mean_j) · ∂m/∂node_j` is node *j*'s own grad × (act − mean) score, and `E[j, out] = |s_j|`.
Edges only from *i* strictly upstream of *j* (node order is (layer, kind); a head never feeds a head of the same or an
earlier layer). This is `olib.attribution_graph` lifted to GPT-2 and is one backward pass per downstream node.

At **density 0.1** (`legibility.graph.threshold_density` on the symmetrised graph): modularity `Q`
(`modularity_q`, Leiden), **treewidth upper bound** `tw` (`treewidth_bound`, min-fill), node-importance sparsity `sp`.

* **Verdict (AMENDMENTS #5d, restated here for T2): tractable iff `tw ≤ 8`; marginal iff `tw ≤ 32`; else intractable.**
  The treewidth bound is the parameter of the FPT conjecture (§2.14); Q is reported *against its null* but does not
  decide the verdict (a sparse star graph has low Q and is trivially tractable — the defect AMENDMENTS #5d fixed).
* **Null: degree-preserving rewiring**, `legibility.graph.shuffled_q_null(E, density=0.1, n=20, seed=0)`; `Q − Q_null`
  reported.
* Status: `certified` iff verdict == tractable; `failed` otherwise (a `marginal`/`intractable` verdict is a *result*,
  and by §3.2 it also means open() proceeds with **partial extraction only** and says so in the scorecard).
* FINAL_SPEC §6.1 pre-registers "Pass: T1/T2 tractable". That is the claim under test.

---

## 4. Steps 3–4 — certified circuits (FINAL_SPEC §3.3–§3.4, §6.2)

**Step 3, extract.** `whiteboxtopo.legibility.discovery.discover_circuit(view, tokens, targets, means, seed=0,
F_target=0.9, rescore_every=16, n_passes=2, metric=logit_difference)` — the **frozen** pipeline, used unmodified
(attribution-patching node scores + greedy prune, two passes). Output: `keep` (bool over 156 nodes), `size`, `F`, `cost`.

**Step 4, certify (K-O-2 at T2).** The circuit is re-derived on **n_gauge = 20** samples `g` of the combined kind
(`symmetry.gpt2.sample(..., kind="combined")`): for each *g*, build the node view on `apply(model, g)`, **recompute the
means on the transformed model** (they are not gauge-invariant objects), and

1. recompute the faithfulness of the *same* node set, mapped through *g* (base node set `S`; the transformed model's
   node *j* is base node `p[j]`, so `S` is queried at `{j : p[j] ∈ S}`) → `F_g`;
2. re-run `discover_circuit` from scratch on the transformed model → node set `S_g`, mapped back through *p* to base
   indices → `align(S_g)`.

**Criteria (frozen):**
* **node-set identity** — `align(S_g) == S` for every one of the 20 samples;
* **faithfulness invariance** — `CV(F_g) < 1e-3` (the `ART_CV` bar of SPEC_O's K-O-2, used everywhere in Block O);
* both hold → the circuit is a **certified** object; `CV ≥ 1e-3` or a differing node set → **failed (artifact)** for the
  circuit-derived cells, and the R cells then rebuild from *nothing* (they report `not_reached: no certified circuit`),
  exactly as `meisho_doto.run_grok` gates its R cells on P1.
* **Reported either way**, because it is the interesting number: the **certified fraction**
  `|∩_g align(S_g) ∩ S| / |S|` and the mean Jaccard `J(align(S_g), S)` over the orbit. A node set that is *almost*
  invariant is a `partial`, not a pass — and the fraction is the T2 answer §6.2 asks for ("report certified fraction").
* Discovery is **not** run at several seeds on the base model: `discover_circuit` is deterministic given `seed` and the
  greedy order, and the gauge orbit is the stronger test. Seed stability is therefore *not* claimed.

**P1 (what it learned), T2 form (§6.2).** The cell value is the **certified circuit inventory**: the node list
(`L{l}.h{h}` / `L{l}.mlp`), `|S|`, `|S|/156`, the faithfulness on the discovery split, and the certified fraction above.
Status `certified` iff the certified fraction is **1.0** and the K-O-2 criteria hold; `partial` if the circuit exists but
the fraction is < 1.0 (value carries the fraction); `failed` if discovery did not reach F ≥ 0.9 at all.
Null: **a random node set of the same size** — its faithfulness is reported as the baseline the circuit must beat
(§12.3, "baseline or no claim"); 20 draws, mean and 99th percentile. Pre-registered: the circuit's F must exceed the
99th percentile of the random-set F distribution. (§5a scale-free form: this is the K-O-1 analogue at T2.)

**Kill test K-O-2 is the gauge check above. K-O-1 and K-O-3 do not apply at T2** — there is no irrep inventory to
randomise and no memorising control specimen at GPT-2 scale; recorded as such, not silently dropped.

---

## 5. P2 — why this decision (FINAL_SPEC §6.3, §2.5)

For transformers §6.3 pre-registers "attention-pattern-conditional path products (pattern frozen per input), reported
with the QK caveat". Implemented as: the attention probabilities are **detached** in the backward pass, and the
per-position contribution of the input token embedding to the metric is `c[b, t] = (∂m/∂x[b,t]) · e[b,t]`, the exact
frozen-pattern linear attribution of that token (the same construction as `meisho_doto.token_contributions`).

* **Invariance:** `CV(c)` over the gauge orbit (`min(n_gauge, 20)` samples) must be `< 1e-4` — the fp32 realisation of
  §6.3's "CV < 1e-6", which fp32 GPT-2 logits (max |logit| ≈ 160) cannot reach; the exact bar is recorded as the
  pre-registered one that the precision, not the model, forbids.
* **Anchor (D1):** integrated gradients along the *constant-pattern* path `(1−δ)e → e`, δ = 0.05, 32 steps
  (AMENDMENTS #5c), `r > 0.95` against `c`. The zero-baseline global IG is reported alongside as the diagnostic it was
  at T1 (GPT-2 with LayerNorm, GELU and softmax is not piecewise linear, so a global path crosses many regions).
* **QK caveat**, recorded in the cell provenance: with the pattern frozen the attribution is exact for the OV circuit
  and *ignores* the query-side dependence of the pattern on the input. It is a statement about a linearisation of the
  model at *x*, not about the whole model.
* Status: `certified` iff CV < 1e-4 **and** anchor r > 0.95; `partial` if only the invariance holds; `failed` otherwise.
* If the orbit computation exceeds its budget (see §10) the cell is `not_reached: cost` — pre-registered escape, taken
  before the numbers are seen or not at all.

---

## 6. Step 6 — glue: Čech H0/H1 of the explanation presheaf (FINAL_SPEC §3.6, §2.12)

* **Cover of D_B:** `sheaf.cover(Z, k=10, overlap=0.5, seed=0, method="knn2", n_pca=20)` where `Z` is the **per-unit
  standardised** (recorded gauge fixing, §2.10) residual-stream cloud at the **mid layer (block 6), last position**, one
  point per prompt, over the 256 discovery prompts. k = 10 is inside the pre-registered 8–12 band and is frozen here.
* **Local explanations:** per-example atoms `E[n, j] = Σ_positions (node_j(x_n) − mean_j) · ∂m_n/∂node_j` over the 156
  nodes — the per-example form of the P0 node scores, the same atoms `meisho_doto.per_example_scores` builds at T1.
* **`sheaf.cech_H0_H1(cover, E, tau=0.5, k_e=32)`** → H0, H1, `boundaries`, `cyclomatic`, `omega`.
* **Pre-stated expectations (this is the claim, written before the numbers):** **IOI H1 > 0** (the IOI circuit is
  several mechanisms — duplicate-token/induction/S-inhibition feeding name-movers — so local explanations should fail to
  glue somewhere) and **induction H1 ≈ 0** (one mechanism). Greater-than: no prediction is pre-registered; it is
  reported.
* **Boundary localisation:** every nerve edge with J < τ is reported with its J, its widest-path agreement `J_bott`, its
  overlap size and whether it is *frustrated* — plus, for the largest-|J−J_bott| boundary, the nodes in the symmetric
  difference of the two regions' top-`k_e` sections. That is "which overlaps disagree, and about which nodes".
* **Trivial-explanation null** (AMENDMENTS #5f, mandatory): the same cover scored with a constant explanation (every
  example given the population-mean atom vector + 1e-9 noise) must return H0 = 1, H1 = 0. If it does not, the instrument
  is reported as failing its own null and the cell is `failed` regardless of the model's numbers.
* **Sanity-gate direction (AMENDMENTS #5f):** at T1 a clean single-mechanism task with H1 > 0 indicts the *instrument*.
  At T2 there is no single-mechanism ground truth, so the gate is carried by **induction**: induction H1 > 0 is reported
  as "instrument not validated at this τ/k_e", not as a property of GPT-2.
* `cyclomatic == 0` ⇒ H1 = 0 holds by construction, not by evidence; the cell is then `partial (uninformative nerve)`.
* Status: `certified` iff the trivial null passes, `cyclomatic > 0`, and the measured H1 matches the pre-stated
  expectation for that task (IOI > 0, induction = 0); `partial` if the nerve is uninformative; `failed` if the
  expectation is contradicted — which is a **result about the pre-registration**, and is reported as such.

---

## 7. Steps 7–8 — R3, R4, R5, R6 (FINAL_SPEC §7)

**R3 — rebuild from certified circuits.** `f̂` = the certified circuit with the complement **mean-ablated** — i.e. the
node view run with `keep = S` and the discovery-split means. **Pass: faithfulness ≥ 0.9 on the held-out 256 prompts**
(a split never seen by discovery). Report γ_cov (R4).
*Failure boundary:* `whiteboxtopo.reconstruct.from_circuits` is **not** used — it calls `invariants.model.as_relu_mlp`
and is a ReLU-MLP instrument (its own docstring says so). R3 here is the *same definition* implemented on the GPT-2 node
view; this substitution is recorded in the cell's provenance.
Null: the same rebuild from a **random node set of the same size** (20 draws) — its held-out faithfulness is the
baseline; the certified circuit must exceed the 99th percentile.

**R4 — residual and coverage gap.** `h` = per-unit standardised mid-layer (block 6) residual stream of the **full
model**, all positions of the held-out prompts, flattened to (B·T, 768); `ĥ` = the same activations from the
**circuit-only run** (`keep = S`, complement mean-ablated). `r = residual(h, ĥ)`;
`coverage_gap(r, q=0.99, seed=0)` → `γ_cov` with the Tracy–Widom-corrected MP edge, **the K-block MP null reused
verbatim** (`whiteboxtopo.darkmatter.rmt`, as §8.4 requires), plus the module's two nulls: Gaussian residue with matched
marginals and the coordinate-shuffled residue.
**No pass threshold at T2** — §7 R4 says "T1: γ_cov < 0.05 …; T2–T4: report". Status is therefore `partial` whenever a
number is produced, `certified` never, `failed` only if `γ_cov` is not clear of both nulls (a gap that its own null
reproduces is not a gap). `1 − γ_cov` is the degree of opening for that task and goes into the grid.

**R5 — behavioural equivalence.** On the held-out prompts, per-example **logit difference** of `f̂` vs `f`:
* Pearson `r ≥ 0.9` over examples, **and**
* `|mean m(f̂) − mean m(f)| ≤ 0.1 · |mean m(f)|` (the 10 % band, the same band §7 R2 uses for integration error).
The Gaussian/PGD perturbation curves and the OOD battery of §7 R5 are **not applicable**: `f̂` is the same network with
nodes mean-ablated, so it shares `f`'s input space and a perturbation curve would measure the model, not the rebuild.
Recorded as `not_applicable`, never counted as a pass.

**R6 — ablation symmetry.** For each node `k` in the certified circuit: ablate `k` in `f` (full model, `keep = all \ k`)
and in `f̂` (circuit, `keep = S \ k`); effect vectors = the per-example change in the metric on the held-out prompts.
`verify.ablation_symmetry(..., threshold=0.9)`; **pass iff every node has r > 0.9**; any node below is reported as
mis-mapped, with its name.

**R7, R8 are not run at T2** and are recorded `not_reached`: R7's artifact-class rebuild has no meaning for a node
circuit (a "non-certified node set" is just the complement, already measured by the random-set null of R3), and R8 is
gated on R5 failing *and* on a second extraction round that the schedule does not fund at T2. Stated here, before the
results, so it cannot become a post-hoc excuse.

---

## 8. The remaining cells

| cell | at T2 | why |
|---|---|---|
| **P1** | see §4 — certified circuit inventory, node list + certified fraction | §6.2 T2 clause |
| **P2** | see §5 — frozen-pattern path products with the QK caveat | §6.3 |
| **P6** | `signals.memorization_signal(activations=<mid-layer resid cloud>, family="lm_transformer")`, the **audited M-block signal** with its shipped calibration (`whiteboxtopo/signals/calibration/memorization_lm_transformer.json`), score + 95 % bootstrap interval | §6.7: "T2+: audited memorization_signal per family (M block) with baselines … report honestly against baselines" |
| **P3** | `not_reached` | §6.4's T1 procedure needs a planted shortcut with known onset and §6.4's T3 procedure needs checkpoints. **GPT-2 ships one final checkpoint**; there is no training trajectory to run zigzag or eigenflow over. P3 at T3 is O3.1 (Pythia, HPCC). |
| **P4** | `not_reached` | §6.5's diffusion-embedding OOD predictor **failed its pre-registration at T1** (O1.4 Fine Motion: AUC 0.41–0.51 on the clean grok specimens against the pre-registered > 0.9, and beaten by the softmax-confidence baseline at 0.999 — `results/open/fine_motion_grok_clean_s*.json`). §12.1 forbids re-running a failed procedure at a new tier without a new pre-registration, and no such pre-registration exists. Not re-run. |
| **P5** | `not_reached` | Same, for §6.6's curvature × Lipschitz failure predictor (O1.4: Spearman 0.10, p 0.09 where the pre-registration demands < −0.5 at p < 0.01, and NaN on the seeds where no perturbation radius flipped a prediction). Additionally `ρ(x) = ‖Q_h(x)‖·Lip(x)` needs the path product `A(x)`, which at GPT-2 exists only in the frozen-pattern linearisation of §5 — so even the quantity is not the pre-registered one. |
| **P7** | `not_reached` | Same reason as P3: no checkpoints. §8.2 assigns P7 at a real training run to **O3.1 (Pythia-410M/1B, 154 checkpoints, HPCC)**. |

**P6 status rule, frozen:** `partial` whenever the calibrated score is produced. It can never be `certified` here,
because §6.7's pass criterion is a *separation* (AUC) between memorising and generalising models and **T2 has exactly one
model and one family** — there is no contrast, so there is no AUC. Reporting a single calibrated score as if it were a
verdict is precisely the move §12.3 forbids. `not_reached` if the calibration file for `lm_transformer` does not load.

---

## 9. O2.2 — Narita Top Road (the audit at real scale)

`satono_crown.py --audit` reads `experiments/F_gauge/results/F11_audit_table.csv` (n = 30 renamings per quantity, kinds
perm/scale/rot/combined, layer 6) and `F11_identity.json`, **recomputes nothing**, and writes
`results/open/narita_top_road.json`: per quantity the verdict, the gauge-aligned verdict, the per-kind median CVs, and
the **classification counts** invariant / covariant / artifact — over the raw verdict and over the aligned verdict
(a quantity that is an artifact raw and invariant after alignment is *gauge-covariant bookkeeping*, and the pair of
counts is the audit's actual content). Provenance points at the F11 CSV; the file records that O2.2 is a carry, not a
run, and that F11's own thresholds (`invariants.check.INVARIANT_CV`) are the ones in force.

If the CSV is missing the cell is `not_reached: F11 results not on disk`.

---

## 10. Cost budget and the pre-registered escapes

Budgets, frozen before the smoke run, per task, on the RTX 5070 Laptop (8 GB):
gauge ≤ 10 min · P0 ≤ 20 min · discovery + 20-sample certification ≤ 4 h · P2 ≤ 30 min · glue ≤ 30 min ·
R3/R4/R5/R6 ≤ 1 h · P6 ≤ 15 min.

A cell that exceeds its budget is written `not_reached: exceeded pre-registered budget (<measured>)` and the run
continues — it is never silently shortened, and a threshold is never moved to make a cell affordable. The escape is
declared here so that taking it is a recorded event and not a judgement made after seeing a number.

The run is **resumable per task and per cell**: `results/open/satono_crown_<task>.json` is written after every cell, and
a re-run merges into it (the `meisho_doto` convention). The same script runs unchanged on the HPCC
(`hpcc/sbatch/o2_gpt2_open.sbatch`, 1 GPU, 32 GB, resumable) if the laptop does not finish.

---

## 11. What a pass looks like (O2.1, §8.2)

§8.2 pre-registers for O2.1: "IOI, induction, greater-than. Steps 1–9 with certified circuits; H0, H1 with boundary
localization (IOI expected H1 > 0, induction ≈ 0); R3/R4. Report γ_cov per task." There is **no aggregate pass/fail
threshold for T2** in FINAL_SPEC — T2 is a *measured* opening, and the deliverable is the three grid rows with their
γ_cov. The per-cell criteria above are the only thresholds, and the headline numbers are:

1. the P0 verdict per task (tractable / marginal / intractable),
2. the certified circuit and its certified fraction per task,
3. H1 per task against the pre-stated expectation,
4. the held-out faithfulness of `f̂` (R3) against its random-node-set baseline,
5. **γ_cov per task, and the degree of opening 1 − γ_cov** — the grid cell.

Every one of those is reported whether it passes or not (§12.4).
