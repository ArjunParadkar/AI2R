# O3.1 — T3 Pythia P7/P3 (Hishi Akebono): pre-registration

_Written 2026-09-14, **before any O3 compute**. FINAL_SPEC §2.8, §2.10, §6.4 (P3, T3 row), §6.8 (P7, T3 row), §8.2 (O3.1),
§12. Runner: `experiments/O_opener/hishi_akebono.py`; cluster job `hpcc/sbatch/o31_pythia_open.sbatch`.
Naming: O3.1 = **Hishi Akebono** (`experiments/HORSES.md`); the handbook code O3.1 is the pre-registration label._

Everything numeric below is frozen at this commit. Changes after any O3 result exists go to `AMENDMENTS.md` with the
reason and the date, never into this file (§12; SPEC_O.md's closing rule).

**What §8.2 asks of this row, verbatim:** "Eigenflow, normalized-persistence velocity, legibility drift, zigzag on
layer-mid activations across all checkpoints; bias probes with B2/B3 procedures. **Report.**" §6.4 and §6.8 both put T3
in the *reported* column, not the *pass/fail* column. So O3.1 issues **no pass verdict**: it issues numbers, the nulls
they were measured against, and an explicit statement of what would have counted as a positive (below). Nothing in this
row can be "certified" — see the gauge section.

---

## 1. Models, revisions, probe, cloud

| | |
|---|---|
| models | `EleutherAI/pythia-410m` (24 layers, d = 1024, 16 heads), `EleutherAI/pythia-1b` (16 layers, d = 2048, 8 heads) |
| revisions | the V7 list, **154** public checkpoints: `step0`, `step{2^i}` for i = 0…9 (1, 2, 4, …, 512), `step1000`…`step143000` in steps of 1000 (`hpcc/mejiro_ardan.REVISIONS`, imported, not re-typed) |
| probe | the **fixed 2 000-token OpenWebText-10k slice** V7 and K1 use: `experiments/K_darkmatter/eishin_flash.load_tokens(tok)[:16, :126]` = 16 sequences × 126 tokens = 2 016 tokens, identical at every revision and identical for both models (both use the gpt-neox-20b tokenizer), cached as `tokens.npy` in the out dir |
| layer | `mid = n_layers // 2` (410m → 12, 1b → 8), matching V7's graph window `[mid−2, mid+2]` |
| cloud | the **residual stream at the output of layer `mid`**, one point per token: `hidden_states[mid+1]`, positions 1…125 of all 16 sequences = 2 000 points, subsampled to a **fixed** `N_CLOUD = 300` points (`np.random.default_rng(0)`, the same 300 token positions at every revision and in both models) |

Same probe points at every checkpoint is a requirement, not a convenience: the Z-block zigzag construction tracks
homology classes through inclusions of consecutive clouds, and an independently resampled cloud per checkpoint makes
the union track different points (Z AMENDMENTS #2(d)).

## 2. Gauge fixing — and why every O3 cell is capped at `partial`

`whiteboxtopo.symmetry` has backends for ReLU MLPs, `ToyTransformer`, `GPT2LMHeadModel` (F11) and the Llama/Gemma/
Qwen/Mistral family (F12). **There is no `GPTNeoXForCausalLM` backend**, and the Llama backend does not cover it
(Pythia has fused `query_key_value`, LayerNorm rather than RMSNorm, partial-rotary RoPE (`rotary_pct` 0.25) and a
parallel attention+MLP residual). Writing one is not in this row's budget. Consequence, stated up front:

* **K-O-2 cannot be run on Pythia.** No quantity in this row is re-computed on a 300-sample gauge orbit, so no cell may
  be called `certified`. The best status available to P3 and P7 here is **`partial` — "certified under a stated gauge
  fixing"**, and that is what the runner will emit even if every threshold below is met.
* The gauge fixing that *is* applied, and is recorded in every cell's `provenance`:
  1. The residual-stream cloud is invariant **by construction** under the parts of G that act inside a block — attention-head
     permutations, within-head basis changes, MLP hidden-unit permutations — because none of them changes the residual
     stream (the same argument K6 and B2 use for GPT-2's residual `g_l`).
  2. Per-coordinate standardisation: every residual coordinate is divided by its own std over the probe **at that
     revision** (`whiteboxtopo.topology.scaling.standardized_cloud(how="std")`) — the exact E2/Z5 repair that makes every
     diagram-derived statistic invariant under per-unit scalings.
  3. Fixed 20-dim basis: the top-20 PCs of the **final revision's** (`step143000`) standardised cloud, with that
     revision's mean as the centre. Frozen once per model, reused at every revision (Z1 convention).
  4. Per-revision centring and scaling to unit median pairwise distance (Z AMENDMENTS #2(a)): shape across training,
     not size.
* What is *not* covered: LayerNorm γ-rescalings compensated into the next linear map act on block inputs, not on the
  residual, so they leave the cloud fixed; a hypothetical exact per-coordinate rescaling of the residual stream is
  removed by (2) but has not been *sampled and measured*. That is the gap, and it is why the status is `partial`.

## 3. P7 — how representations evolve (§6.8, T3 = reported)

Four instruments on the 154 gauge-fixed clouds. Each produces one scalar trace X(t) over the revision index
t = 0…153.

| # | instrument | module | trace reported |
|---|---|---|---|
| P7-a | **zigzag** intervals of H1 classes over checkpoints | `whiteboxtopo.topology.zigzag` | birth revision of every *qualifying* H1 class; β1 curve |
| P7-b | **Laplacian eigenflow** | `whiteboxtopo.spectral.eigenflow` (k = 15, n_eigs = 10, small-eigenvalue threshold 0.1) | spectral gap λ2 − λ1; `n_small` |
| P7-c | **normalised-persistence velocity** (Z5) | `whiteboxtopo.topology.normalized_persistence` (how = "std", maxdim 1) | NP(t) = total persistence H1 = Σ(d−b)² of the standardised cloud; velocity v(t) = NP(t) − NP(t−1) in **revision-index** units (a step-normalised copy v_step(t) = (NP(t) − NP(t−1)) / (step_t − step_{t−1}) is reported alongside; the index version is the primary) |
| P7-d | **legibility drift Q(t)** | `hpcc/mejiro_ardan.eap_graph` (imported, not duplicated) + `whiteboxtopo.graphs` | the V7 rows: Leiden Q, degree-preserving-rewired Q_null (n = 20), Q_excess = Q − Q_null, community count, min-fill treewidth bound, participation-ratio sparsity, at edge densities 0.01 / 0.05 / 0.1. **Primary trace: Q_excess at density 0.05.** |

Zigzag budget (frozen, inherited from Z AMENDMENTS #2/#3 and O1.3 AMENDMENTS #6, whose measurements are the reason
these numbers are not the Z-block's): Rips scale = the scale at which the **final** normalised cloud carries the most
H1 classes (quantile 0.1 of its distances if it carries none); `SIMPLEX_BUDGET = 30 000`; the ladder lowers the point
subsample `N_SUB` through 100 → 75 → 50 → 40 → 30 first and only then halves the number of clouds down to a floor of
`ZZ_MIN_CK = 20`; `ZZ_MAX_CK = 40` caps the zigzag cloud count from the start (154 clouds at 30 k simplices is not
affordable — onset resolution at 40 clouds is 1/39 = 2.6 % of the revision index). A qualifying H1 class is one
**born after the first cloud used, alive at the last, lifetime ≥ 3 positions** (Z1's "long bar"). `n_sub`,
`budget_reduced`, `n_clouds`, `clouds_reduced`, `resolution_frac` are recorded on every row.

## 4. P3 — when bias entered (§6.4, T3 = reported)

Naturally occurring gender/occupation probes, the v1 WinoBias setup as repaired by B AMENDMENTS #2:
`experiments/B_bias/blib.build_corpus_unique(0)` = 40 frames × 12 occupations = **480 distinct male-stereotyped and 480
distinct female-stereotyped sentences**, no duplicates. Mean-pooled residual at the same mid layer, per revision
(`hpcc/mejiro_ryan.pooled_resid`, imported).

**B2 arm — bias-direction invariance.** g(t) = unit(μ_M(t) − μ_F(t)) in the residual stream. Reported per revision:

* `cos_to_final` = ⟨g(t), g(153)⟩;
* `dbar` = ‖μ_M − μ_F‖ / pooled sd (the magnitude trace);
* the **B2 kill test** (B SPEC §B2), recomputed at every revision: `cos_mf_disjoint_halves` = mean cosine between
  directions built from disjoint halves of the M/F sentence sets, vs `cos_random_split_absmean` = mean |cosine| between
  directions from 20 random halvings of the pooled set. The direction is "real geometry" only if
  **`cos_mf_disjoint_halves` > 0.9 AND `cos_random_split_absmean` < 0.5**.
* Gauge: as §2. The two objects B2's audit actually stresses — g in **MLP-hidden coordinates** and in **per-head output
  coordinates**, where MLP-unit permutations and within-head rotations act — require `symmetry.apply` on a gpt-neox
  model and are therefore **not computed**. This is the concrete reason P3 is `partial`, and it is written into the
  cell's `provenance` verbatim.

**Pre-registered expectation, from the run that already exists:** on GPT-2 small the B2 kill test **failed at all four
blocks** (`experiments/B_bias/results/B2_kill.json`: disjoint-halves cosine 0.65–0.79 against a required > 0.9;
random-split |cosine| 0.31–0.61 against a required < 0.5). If Pythia behaves like GPT-2, the B2 arm of P3 reports a
direction that does not survive its own kill test, and O3.1 says so.

**B3 arm — power-adequate n.** BiasTop (W2 between the Vietoris–Rips H1 diagrams of the two PCA-20 residual clouds)
permutation test on the real M/F split: `experiments/B_bias/winning_ticket._biastop` / `._perm_test`, 100 permutations
(B AMENDMENTS #1 records that 100, not the SPEC's 200, is what the runner has always used), PCA-20 frame rebuilt from
the pooled M∪F cloud **at that revision** (the frame `mejiro_ryan` installs into `winning_ticket._B3`).

n is taken from B5 (`hpcc/mejiro_ryan.py`) for this family: if `hpcc/results/b5/B5_requirements.json` exists and gives
`n_required_power80_stretch05` for `EleutherAI/pythia-410m`, that n is used and the cell is **power-verified**;
otherwise **n = 200** (the largest B3 grid point with 2n ≤ 480) and the cell is **power-unverified**, which is recorded
in `provenance` and in the cell value.

**Pre-registered expectation, again from what exists:** GPT-2's B3 curve never reached power 0.8 at any feasible n —
`experiments/B_bias/results/B3_summary.csv` gives max power 0.60 (stretch δ = 0.25), 0.54 (δ = 0.5), 0.42 (δ = 1.0) at
n ≤ 200, and 0.25 on the real M/F split; the B5 smoke run returns `n_required_power80_stretch05: null` for GPT-2. The
BiasTop arm of P3 is therefore expected to be **underpowered on Pythia too**, and a non-rejection will be reported as
*uninformative*, never as "no bias". This is stated before the run so it cannot be claimed afterwards.

## 5. The change-point rule, frozen

Every scalar trace X(t) above (eigen gap, n_small, NP, NP velocity, Q_excess@0.05, cos_to_final, dbar, BiasTop
statistic, zigzag births-per-checkpoint) goes through one rule, the same one O1.3 Nishino Flower used:

1. **Statistic**: two-sample max-t change point — stat = max over splits i of
   |mean(X[i:]) − mean(X[:i])| / (s_p √(1/i + 1/(T−i))); `cp_index` = the argmax split. Undefined (stat 0, index −1)
   for T < 6 or zero variance.
2. **Null**: **checkpoint-order shuffle**, 2 000 draws — permute the revision order, recompute max-t. p = (1 + #{null ≥
   stat}) / 2001. **Detection iff p < α = 0.05.**
3. **Onset estimate**: the first index at which the trace crosses the midpoint between the pre-change and post-change
   means (the half-rise convention of `spectral.flow.crystallization_index`, §6.8). Reported as `onset_index`,
   `onset_frac` = index/153, **and `onset_step`** = the training step of that revision.

## 6. The concrete numeric statements O3.1 will report

Per model, per instrument, one row of: `stat`, `p` (order-shuffle null), `cp_index`, `onset_index`, `onset_frac`,
`onset_step`, `pre`, `post`, `detected`. Plus:

* **Onset agreement across instruments, within a model.** Onsets are compared on the log-step axis (the revision grid
  is log-spaced to step 512 and linear after it): Δ(a,b) = |log10(1 + step_a) − log10(1 + step_b)|. Two instruments
  **concur** iff Δ ≤ 0.5 (a factor of ≈ 3.2 in training steps). Reported: the pairwise Δ matrix over the *detecting*
  instruments and `max_pairwise_delta`.
* **Agreement across the two models.** For each instrument that detects in both models: Δ_model =
  |log10(1 + step_410m) − log10(1 + step_1b)|; **agree iff Δ_model ≤ 0.5**. Reported per instrument plus the count
  `n_agreeing / n_detecting_in_both`.
* **Trace agreement across the two models.** Spearman ρ between the two models' traces on the 154 shared revisions,
  per instrument, with a 2 000-draw checkpoint-order-shuffle null and its p.

**What would count as a positive** (stated now so it cannot be moved later; none of these makes the cell `certified` —
§2):

* **P7 positive** = **≥ 3 of the 4 instruments** detect a change point at p < 0.05 in **both** models, **and** their
  onsets concur (`max_pairwise_delta` ≤ 0.5) within each model, **and** each concurring instrument agrees across the two
  models (Δ_model ≤ 0.5). Anything less is reported as the partial pattern it is — e.g. "3/4 detect in 410m, 2/4 in 1b,
  onsets spread over 1.8 decades" is a legitimate O3.1 result and will be written that way.
* **P3 positive** = all three of: (i) the **B2 kill test passes** at ≥ 1 revision
  (`cos_mf_disjoint_halves` > 0.9 and `cos_random_split_absmean` < 0.5); (ii) at least one bias trace (`cos_to_final`
  or `dbar`) has a change point with p < 0.05, whose `onset_step` is the reported answer to "when bias entered";
  (iii) BiasTop rejects at α = 0.05 at an n whose power ≥ 0.8 is **established by B5**. If (iii) is unavailable, P3 is
  `partial` whatever (i) and (ii) do, and the value carries `power_verified: false`.
* **Negative results are deliverables** (§12). "No instrument detects" and "the instruments disagree by two decades"
  are both publishable outcomes of this row and neither is retried with different thresholds.

## 7. Nulls, one per claim

| null | applies to | draws |
|---|---|---|
| **checkpoint-order shuffle** | every change point; every cross-model Spearman | 2 000 |
| **label permutation** (M/F labels shuffled over the 960 pooled sentences) | `dbar`, at every revision (200 draws). The **BiasTop p-value is itself a label-permutation null** by construction (`winning_ticket._perm_test` pools the two clouds and re-splits), so no second null is computed for it | 200 / 100 |
| **degree-preserving rewiring** (`whiteboxtopo.graphs.rewired_null`) | Q(t): Q_null and Q_excess, at every revision and density | 20 |
| **zigzag cloud-order permutation** (shuffle the checkpoint order, recompute the longest non-initially-born H1 bar) | the zigzag instrument | `--zz-null` (default **0 = not run**; each draw is one full zigzag. When 0, the cell records "no permutation null run" and the zigzag evidence is descriptive only) |

## 8. Scorecard, and the cells that are not reached

One `Scorecard` (`whiteboxtopo.open.Scorecard` / `CellResult`) per model, written to
`hpcc/results/o31/scorecard_<model-tag>.json` (+ `.md`). Only **P3** and **P7** carry values. Every other cell is
`not_reached` with the reason naming the run that owns it:

| cell | status | reason recorded |
|---|---|---|
| P0 | `not_reached` | the legibility numbers (Q, treewidth, sparsity) *are* computed here as the P7-d instrument, but the P0 **gate verdict** needs the V-block phase diagram, and V8 found no calibrated structure→tractability map (`legibility.profile.V8_CALIBRATION_ATTEMPT`); issuing a verdict from the provisional rule would be a claim this row cannot support |
| P1 | `not_reached` | no certified extraction on Pythia in this row (no gpt-neox gauge backend; §8.2 puts T3 at P7/P3 only) |
| P2 | `not_reached` | path products require the gauge backend; O2.1 Satono Crown owns P2 at T2 |
| P4 | `not_reached` | O1.4 Fine Motion (T1); T4 P4 is K8/γ_cov on the frontier row |
| P5 | `not_reached` | O1.4 Fine Motion (T1) |
| P6 | `not_reached` | M-block `memorization_signal` per family; not this row |
| γ_cov | `None` | no reconstruction is attempted at T3 — an unknown gap is never reported as a zero gap (`Scorecard.degree_of_opening` stays `None`) |

## 9. Outputs

```
hpcc/results/o31/
  tokens.npy                       the frozen probe (shared with V7's cache format)
  basis_<tag>.npz                  mu, P20, g_final, mid layer, probe indices  (frozen from step143000)
  clouds_<tag>.npz                 the 154 gauge-fixed 20-dim clouds (300 x 20 each) — the resumable cache
  O31_<tag>_trajectory.csv         one row per revision: every scalar trace + wall time
  O31_<tag>_legibility.csv         one row per revision x density: the V7 columns
  scorecard_<tag>.json / .md       the Scorecard (P3, P7; the rest not_reached with reasons)
  O31_<tag>_cells.json             full cell evidence: criterion, null, provenance, per-instrument change points
  O31_compare.json / .csv          --aggregate: the two-model comparison of §6
```

Resume: `--resume` skips any revision already present in `O31_<tag>_trajectory.csv` (one CSV write per revision, so a
time-out is re-submitted, never restarted). The trajectory-level instruments (zigzag, eigenflow, change points) run in
the finalise stage from `clouds_<tag>.npz`, so they can be re-run without touching the GPU.

## 10. Kill tests carried over

* **K-O-2 (gauge check)** — *cannot be run*, §2. Recorded as an explicit failure of the certification path, not
  silently omitted.
* **B2 kill** (random-split direction must be unstable while the M/F-split direction is stable) — run at **every**
  revision, §4. A revision where the random-split cosine is as high as the M/F cosine has no bias direction to date.
* **Order-shuffle sanity** — the order-shuffle null is computed for every trace including the ones that do not detect,
  so the reported p-values are a complete set, not a selected one.

---

# Amendments

## #O3-1 (2026-09-14, capability — the GPT-NeoX gauge backend now exists)

`whiteboxtopo.symmetry.neox` was written and committed after this pre-registration was frozen (dispatch via
`whiteboxtopo.symmetry._backend`; `tests/test_neox.py`). §2 above states, as of the freeze, that K-O-2 is unrunnable on
Pythia and that no O3.1 cell can exceed `partial`. **That constraint is lifted.** Nothing else in the
pre-registration changes: the probe, the layer, the cloud, the four P7 instruments, the two P3 arms, the change-point
rule, the thresholds in §5–§7 and the nulls in §7 are exactly as frozen above. No threshold is touched — this
amendment *adds* a certification step that §2 said could not be run, and it is written before any orbit compute.

**What the backend gives.** Exact renamings of `GPTNeoXForCausalLM`: head permutations; per-head `GL(d_head)` on the
value/output pair; the RoPE-commuting per-plane complex scalar (phase + scale) on Q/K over the first `rotary_ndims`
dims; unrestricted `GL` on the non-rotary Q/K tail; MLP hidden-unit permutations; `input_layernorm` and
`post_attention_layernorm` gain/bias rescales compensated in their single consumer; and the final-LayerNorm rescale
(exact only because Pythia has untied embeddings — the one symmetry GPT-2 does not have). **Not** sampled: MLP unit
scalings (GELU is not positively homogeneous), residual-stream re-basing, rotary-plane permutations or any Q/K map
mixing rotary with non-rotary dims.

**The K-O-2 step, as added to the runner.**

* **Orbit**: `--n-gauge` samples of `kind="combined"`, `orthogonal=False` (the full group, GL not O) per orbit
  revision, drawn with `neox.sample` and applied with `symmetry.apply` (which returns a renamed deepcopy).
  **Default `--n-gauge 20`** on the cluster — not SPEC_O's 300 — because each sample costs a full re-measurement
  (forward + EAP backward + the bias corpus), and the cost bound is stated in the RUNBOOK row. `--n-gauge 0` skips
  the step and the cells stay `partial`.
* **Orbit revisions**, pre-stated to bound cost: **every 10th revision plus the final** (`--orbit-every 10`) — 17 of
  the 154. The remaining 137 revisions are gauge-*fixed* (§2) but not gauge-*certified*, and that is recorded.
* **Quantities recertified.** P7: `eigen_gap`, `n_small` (the per-checkpoint eigenflow pair), `np_H1` (the Z5
  normalised-persistence value whose differences are the velocity), the EAP legibility `Q`, `treewidth_ub`,
  `sparsity`, the sorted community-size vector, and the **zigzag qualifying-interval count** — the last computed as
  one zigzag per gauge sample over the orbit revisions' clouds (the orbit clouds are cached, so this costs no extra
  model loads). P3: `dbar`, `biastop_stat`, `cos_mf_disjoint_halves`, `cos_random_split_absmean`, plus **B2's
  coordinate audits, which §4 said were not computable**: `g_resid_cos` (cosine to the base direction), `g_mlp_norm`
  and `g_mlp_cos_aligned` (MLP-hidden coordinates, un-permuted with `inverse_on_units`), and `g_heads_cos_aligned`
  (per-head output coordinates, aligned by **un-permuting first and then un-rotating at the base head index** —
  `neox.apply` rotates at the original index and then permutes, B AMENDMENTS #4's ordering trap).
* **Evidence only, never a criterion**: `g_heads_per_head_norm`. It is an `O(d_head)` invariant, not a `GL` one
  (B AMENDMENTS #4), and the orbit is sampled with `orthogonal=False`, so it is expected to move and its CV is
  reported without being used to pass or fail anything.
* **Tolerance.** "Identical" means "at the fp32 arithmetic floor". The floor is *measured*, not assumed: every gauge
  sample records `max |logits_g − logits| / mean|logits|` on 32 probe tokens, and the maximum over the orbit is
  reported as `fp32_floor_rel_max` in every cell. The certification threshold is SPEC_O's own **`CV < 1e-3`**
  (`ART_CV`, the same constant `meisho_doto.py` uses at T1); a set-valued quantity certifies iff the set is identical
  across the orbit.
* **Verdict, replacing §2's blanket cap.** A cell is **`certified`** iff its instruments produced values, the orbit
  was run, and every certified scalar has `CV < 1e-3` (sets identical); **`failed`** (artifact) if any certified
  scalar has `CV ≥ 1e-3`, with the offending quantity named in `Scorecard.errors`; **`partial`** if the orbit was not
  run (gauge-fixed, not gauge-certified); `not_reached` if no instrument produced a value. §8's table of
  `not_reached` cells is unchanged — P0–P2 and P4–P6 are out of this row's scope for reasons that were never about
  the backend, and the two reasons that did cite the missing backend are reworded to say so.
* **What does not change.** §6's pre-registered positives are about *detection and agreement*, not about gauge
  invariance: a `certified` P3 cell whose B3 arm is power-unverified is still **not** a positive result, and a
  non-rejection is still uninformative. Certification says the number is a property of the function rather than of
  the parameterisation; it says nothing about whether the number is large.

**Expectation, stated before the orbit runs.** The residual-stream cloud is invariant under every element of this
group *by construction* (the backend re-bases nothing in the residual stream), so `eigen_gap`, `n_small`, `np_H1`,
`dbar` and `g_resid_cos` should sit at the fp32 floor; the EAP graph quantities and B2's MLP/head-coordinate audits
are the genuinely non-trivial tests, because head and MLP permutations and the per-head `GL` maps act on exactly
those objects. A CV above `1e-3` there would be a real finding — the instrument would be measuring the
parameterisation — and is reported as `failed (artifact)`, not repaired.

**First orbit measurement (2026-09-14, `pythia-70m`, `step0` and `step143000`, `--n-gauge 3`,
`OMP_NUM_THREADS=4 nice -n 15`; `hpcc/results/o31_ko2smoke/O31_pythia-70m_ko2.json`).** Measured fp32 floor:
`5.9e-06` (step0) and `1.6e-05` (step143000) relative. Everything the amendment expected to sit at the floor does:
`g_mlp_cos_aligned` 1e-08, `g_heads_cos_aligned` 3e-08, `g_resid_cos` 3e-08, `g_mlp_norm` 2e-07, `dbar` 9e-07,
`eigen_gap` 2e-06, `n_small` 0, `np_H1` 9e-05, `biastop_stat` 4e-05, `node_roles` identical — so **P3 certifies**,
including the two coordinate audits §4 said were not computable.

**The EAP legibility instrument (P7-d) does not.** At `step143000`: `Q` CV **0.572**, `treewidth_ub` **0.192**,
`sparsity` **0.061**, `community_sizes` not identical — all far above `ART_CV = 1e-3`, so **P7 is
`failed (artifact)`** under the frozen rule. It is left that way; the rule is not moved. Attribution (a separate
diagnostic, 3 samples per subgroup at `step143000`): every subgroup moves it — `perm` Q CV 0.103, `rot` 0.329,
`scale` 0.278 — and the symmetrised EAP score matrix carries only **91 distinct edge magnitudes among 990 node
pairs**, because `mejiro_ardan.eap_graph` writes one scalar per (source node, target *layer*) pair, so every target
in a layer receives the same score. `threshold_density` at 5 % therefore has to break very large ties, and any
relabelling — or fp32 roundoff differing by 1e-6 between two exactly-equivalent parameterisations — reshuffles which
tied edges survive. The EAP scores are invariant *analytically* (head contributions `z W_O^T` are unchanged by the
V/O basis change; q, k, v are unchanged by the LayerNorm rescales; the gradients are taken at the residual stream);
what fails is the **density-thresholded graph statistic built on top of a tied score matrix**. This is a finding
about the V7 instrument, whose numbers are already in the PAUSE_4 pipeline, and it is reported as one — not
repaired here, and not used to change any threshold in this SPEC.

**Smoke wall time**: 225.6 s in the runner (262 s total) for 3 revisions with K-O-2 at `--n-gauge 3`, against 37.9 s
for the same 3 revisions with `--n-gauge 0` — i.e. **~5 s per gauge sample** on `pythia-70m`. Cluster projection is
in `hpcc/RUNBOOK.md` §4.
