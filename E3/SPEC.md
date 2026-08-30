# E3 — Empirical explanation cohomology (pre-registration)

Status: SPEC written 2026-08-30 **before** any analysis code or run. Nothing in
`results/` exists yet. Changes after the first run go into the numbered, dated
"Amendments" section at the bottom; the body is never edited.

Rule-5 pause points were lifted by the owner for E3: ambiguous design choices are
*made here, stated, and the alternative noted*, instead of being asked.

## Frameworks we build on (not ours)

* **Sheaf-of-explanations** — arXiv 2603.17141 (March 2026): explanations as
  local sections over a cover of input space; gluing counterexamples; the
  obstruction to a global explanation is cohomological. E3 is a *measurement*
  inside that framework, on a real model. We claim only the measurement.
* **A Gauge Theory of Superposition** — sheaf atlas of feature charts with
  holonomy obstructions (Llama 3.2 3B). Our "disagreement around a cycle of the
  nerve" is the discrete analogue of their holonomy around a loop of charts.
* **IOI** — Wang et al. 2022 (the task, the ABBA/BABA templates, the ABC
  corruption, the logit-difference metric, the multi-mechanism suspicion).
* **Circuit discovery** — Conmy et al. 2023 (ACDC, edge-level circuits on the
  heads+MLPs graph); Syed, Rager & Conmy 2023 (edge attribution patching, EAP,
  for circuit discovery); Hanna, Pezzelle & Belinkov 2024 (EAP-IG; the 32 491-edge
  GPT-2-small graph we reproduce). Attribution patching itself: Nanda 2023.
* Explanation non-identifiability: Méloux et al. 2025 (why local sections are
  not unique — the "re-assignment" freedom in §5).

Our contribution, if any: the first concrete Čech-style obstruction numbers on a
real model, with a control task, nulls, CIs and kill attempts.

## Question

Given a cover of IOI prompt space by overlapping prompt clusters and a
data-driven circuit per cluster (a local explanation), is there an obstruction
to gluing the local circuits into one global explanation, and if so, *which*
pairs of clusters carry it — i.e. where in prompt space do distinct
mechanisms live? Pre-registered comparison: a single-mechanism task
(induction) must give obstruction ≈ 0 (defined numerically in §7).

## 1. Model, tasks, prompts

Model: GPT-2 small (`HookedTransformer.from_pretrained("gpt2")`, TransformerLens
3.8.0, fp32, `use_attn_result`, `use_split_qkv_input`, `use_hook_mlp_in` on).

### 1.1 IOI (the object of study)

* Templates: the 15 BABA templates of Wang et al. 2022 and their 15 ABBA
  mirrors (first two names swapped), listed verbatim in `e3lib/prompts.py`.
  A = IO (answer), B = S (subject, appears twice). BABA order: S1 first;
  ABBA order: IO first. "Name position" = this order.
* Names: Wang et al.'s name list filtered to single-token (with leading
  space) names. Places / objects: the 8 + 8 Wang et al. items (all verified
  single-token). Every prompt ends right before the IO token.
* N = 2400 prompts: 80 per (template, order) cell, names/place/object sampled
  uniformly, seed 0, IO ≠ S.
* Corruption: **ABC** — the three name slots (S1, IO, S2) are replaced by
  three fresh distinct names; the template/place/object stay. Same token
  length as the clean prompt (all names single-token).
* Metric: `logit[IO] − logit[S]` at the final position, IO and S from the
  *clean* prompt.

### 1.2 Induction (the sanity control; run FIRST)

* Sequence: `[BOS] r₁ … r_L r₁ … r_{L−1}`, r uniformly random tokens from the
  vocabulary range [1000, 40000) (no specials), L ∈ {6, 8, 10, 12} with 600
  prompts each (N = 2400), seed 0. Target = r_L.
* Corruption: first copy replaced by fresh random tokens `s₁ … s_L`
  (same length; the target is no longer inferable).
* Metric: `logit[r_L]` at the final position (the brief's choice; the
  alternative `logit[r_L] − mean logit` is not used).

## 2. Embedding of prompts (cover space)

* Primary: mean over the prompt's real tokens (BOS included) of
  `blocks.11.hook_resid_post` on the clean prompt (768-d).
* Secondary (kill iv): mean over tokens of `blocks.6.hook_resid_post`.
* Pre-processing: standardise each dimension, PCA to 20 components (fit on
  the task's 2400 prompts). Alternative not used: raw 768-d (fuzzy c-means
  memberships flatten toward uniform in high dimension).

## 3. Overlapping cover (soft clustering)

* Fuzzy c-means (own implementation, `e3lib/cluster.py`): fuzzifier m = 2,
  ≤ 300 iterations, tol 1e-6, 10 random initialisations (seeds 0–9), keep
  the lowest objective.
* k ∈ {8, 10, 12, 15}; **summary k = 10**.
* Membership rule: prompt p ∈ U_i iff u_ip ≥ θ · max_j u_jp, **θ = 0.5**
  (argmax always included). Overlap U_ij = U_i ∩ U_j.
* A cover is usable only if its nerve has a non-trivial cycle space
  (|E| − |V| + #components ≥ 1). Adaptive rule, pre-registered: if at
  θ = 0.5 the nerve's cycle space is trivial, θ is lowered to 0.33 for that
  (task, k) and this is reported. Alternative not used: GMM posteriors.
* Report per k: cluster sizes, all pairwise overlap sizes, triple overlap
  counts, and composition of each cluster by template × order (IOI) /
  by L (induction).

## 4. Sections: per-cluster circuits (EAP)

Graph (Hanna et al. 2024 / Syed et al. 2023): nodes = input (resid_pre of
layer 0), 144 attention heads, 12 MLPs, logits. Edges = (upstream node
output → downstream node input) for every causally allowed pair, where a
head has three inputs (q, k, v). Count: 32 491 edges.

Hooks: upstream outputs `blocks.0.hook_resid_pre` (input),
`blocks.L.attn.hook_result[:, :, h]` (head), `blocks.L.hook_mlp_out` (MLP);
downstream inputs `blocks.L.hook_{q,k,v}_input[:, :, h]`, `blocks.L.hook_mlp_in`,
`blocks.11.hook_resid_post` (logits).

Per-prompt EAP score of edge (u → v):

    s_p(u→v) = Σ_pos ( a_u^corr[pos] − a_u^clean[pos] ) · ∂metric_p/∂(in_v^clean[pos])

(gradient taken on the clean run; standard EAP, no integrated gradients — the
EAP-IG refinement is the noted alternative, not used, so that the per-prompt
score is exactly linear in the prompt set). Scores are cached per prompt
(`cache/`, not in git), so every cluster/overlap/bootstrap/shuffle quantity
below is a mean over prompt rows — no re-running the model.

Section over U_i: `C_i(k_e)` = the k_e edges with the largest |mean_{p∈U_i} s_p|.
k_e ∈ {50, 100, 200, 400, 800}; **summary k_e = 200**. (Top-k rather than an
absolute threshold so that Jaccard between equal-size sets is comparable
across clusters; the absolute-threshold alternative is not used.)

## 5. Restriction, agreement, disagreement

Restriction of C_i to the overlap U_ij: **re-score C_i's edges on the
overlap's prompts and keep the top half**:

    ρ_ij(C_i) = the ⌈k_e/2⌉ edges e ∈ C_i with the largest |mean_{p∈U_ij} s_p(e)|.

Rationale: "which of cluster i's circuit edges actually matter on the prompts
it shares with cluster j". Without pruning (the alternative, reported as the
secondary column `J_raw`) the restriction is the identity and agreement
reduces to Jaccard(C_i, C_j), which ignores the overlap entirely.

Agreement on U_ij: J_ij = Jaccard( ρ_ij(C_i), ρ_ji(C_j) ). Disagreement
w_ij = 1 − J_ij. Consistent at τ iff J_ij ≥ τ. τ swept on a grid of 0.02 in
[0, 1]; the obstruction curve is reported for every τ, plus the τ-free
integrated quantities in §6.

Nerve N(k): vertices = clusters; edge ij iff |U_ij| ≥ n_min = 10 prompts
(smaller overlaps cannot be re-scored meaningfully — alternative n_min = 1
noted); triangle ijk iff |U_ijk| ≥ 10. Triangles are recorded and used for
the transitivity diagnostic (§6.4); the obstruction itself lives on the
1-skeleton by the brief's construction.

## 6. Čech-style quantities (the exact construction)

Let N = (V, E) be the nerve graph with agreement J_e on each edge.

### 6.1 H⁰(τ)
G_τ⁺ = (V, {e : J_e ≥ τ}). **H⁰(τ) = number of connected components of G_τ⁺**
= the number of connected consistent explanation components.

### 6.2 H¹-type obstruction (discrete)
A *global relabelling* is any map g: V → {1..|V|} (assign each cluster a
mechanism label). It *explains* the consistency pattern at τ iff for every
edge, consistent ⇔ same label. Disagreements that no relabelling can zero
are exactly the inconsistent edges whose endpoints are joined by a
consistent path; each such edge e closes a cycle of N with exactly one
inconsistent edge (a *frustrated cycle*, cf. signed-graph balance, Harary
1953). Define

    obs(τ) = #{ e ∈ E : J_e < τ and endpoints(e) connected in G_τ⁺ }
           = dim ( Z₁^frus(τ) / (Z₁^frus(τ) ∩ Z₁(G_τ⁺)) ),

the number of independent frustrated cycles modulo consistent cycles
(each frustrated edge contributes one fundamental cycle w.r.t. a spanning
forest of G_τ⁺; these are independent; consistent cycles are quotiented
out). Equivalently, with the Z/2 1-cochain d_τ(e) = 1[J_e < τ] on the
1-skeleton, obs(τ) = 0 iff d_τ is a coboundary of some vertex labelling in
the "same-label" sense above; obs(τ) counts the independent cycles on which
it fails. Note: this is a relabelling obstruction on the nerve, not sheaf
cohomology with coefficients in the (non-abelian) sheaf of circuits — the
brief asked for exactly this cycle-space construction, and the abelianised
alternative (per-model-edge Z/2 cochains) is noted, not used.

Closed form via bottleneck agreement: let

    J_bott(e) = max over paths P in N∖{e} joining e's endpoints of min_{f∈P} J_f

(= the min agreement along the maximum-spanning-tree path; 0 if no path).
Then e is frustrated exactly for τ ∈ (J_e, J_bott(e)], so

    obs(τ) = #{ e : J_e < τ ≤ J_bott(e) }.

Algorithm: sort edges by J descending, Kruskal with union-find to get the
maximum spanning forest, then J_bott(e) for every edge by the bottleneck
query on the forest path (|E| DFS walks). Complexity O(|E| log |E| + |E|·|V|);
|V| ≤ 15 so it is instantaneous. Brute-force check (tests): enumerate all
simple cycles of small random nerves, mark those with exactly one
inconsistent edge, compute the Z/2 rank of the frustrated cycles modulo the
span of all-consistent cycles; must equal obs(τ) for every τ.

### 6.3 Continuous weighted version
Per-edge frustrated lifetime m_e = max(0, J_bott(e) − J_e) — how much worse
the direct agreement across e is than the best consistent route around the
nerve. Total obstruction mass

    Ω = Σ_e m_e = ∫₀¹ obs(τ) dτ ,   Ω_rel = Ω / Σ_e (1 − J_e) ∈ [0, 1]

(the fraction of all disagreement that no global relabelling can absorb),
and Ω/|E| (mean lifetime in Jaccard units). Alternative not used: Hodge
decomposition of w on the nerve complex (holonomy-style, as in the gauge
paper) — a non-negative disagreement cochain has no orientation, so its
gradient/harmonic split depends on arbitrary edge orientations.

### 6.4 Triangles (transitivity diagnostic)
For every nerve triangle and every τ: fraction of triangles with exactly one
inconsistent edge (agreement fails to be transitive) — reported as a curve.
Triple-overlap restrictions are *not* computed (alternative noted).

### 6.5 Blocking pairs
Rank nerve edges by m_e (ties by w_e). For each of the top pairs report:
|U_i|, |U_j|, |U_ij|, J_e, J_bott(e), m_e with bootstrap CI (§8 iii), the
template × order composition of U_i, U_j and U_ij, and the ten model edges
that most distinguish ρ_ij(C_i) from ρ_ji(C_j).

## 7. Pre-registered expectations and pass/fail

Summary configuration for all headline numbers: k = 10, k_e = 200,
embedding = layer-11 mean, θ = 0.5; everything else is a sweep.

* **H-E3-control** (must pass before IOI is believed): on induction,
  obstruction ≈ 0, defined numerically as **both** (a) Ω/|E| ≤ 0.05 (mean
  frustrated lifetime ≤ 0.05 Jaccard units) **and** (b) Ω within the central
  95 % band of the prompt-shuffle null (§8 i) for the same cover. If (a)
  fails the method is treated as broken: debug (prompt count per overlap,
  restriction rule, EAP numerics) and document every change as an amendment
  before touching IOI.
* **H-E3-IOI** (the measurement): IOI shows a non-zero obstruction:
  Ω_IOI above the 97.5th percentile of its shuffle null, and at least one
  blocking pair with a bootstrap 95 % CI on m_e excluding 0. Direction of the
  expectation: obstruction on IOI > obstruction on induction at matched k, k_e.
  A negative (IOI glues, or IOI is indistinguishable from its null) is a
  deliverable and is written up as such (rule 3).
* **Localisation**: blocking pairs are reported regardless; the claim "these
  are distinct mechanisms located in prompt space" is only made if the top
  pair survives kill iv (same pair, or a pair with the same template × order
  composition, is among the top-3 blocking pairs under the layer-6 embedding).

## 8. Kill attempts (run for any non-zero IOI obstruction)

i.  **Prompt-label shuffle**: permute the prompt indices of the membership
    matrix (200 permutations, seed 1) — identical cluster sizes, overlap
    sizes and nerve; recompute everything. Obstruction must collapse to this
    floor for the positive to die.
ii. **Hard cover**: argmax memberships, no overlaps → nerve has no edges,
    obs ≡ 0; reported as the degenerate check.
iii. **Bootstrap**: 200 resamples of the 2400 prompt rows with replacement
    (cluster memberships travel with the row; clusters are re-formed from the
    resampled rows, circuits and restrictions recomputed) → percentile CIs on
    every J_e, m_e, Ω, obs(τ). Alternative noted: within-cluster stratified
    resampling.
iv. **Second embedding**: layer-6 mean, full pipeline, same k sweep.

## 9. Deliverables

`results/`: `cover_{task}_{emb}_k{k}.csv` (memberships + composition),
`pairs_{task}_{emb}_k{k}_ke{ke}.csv` (per nerve edge: sizes, J, J_raw, J_bott,
m_e, CIs), `obstruction_curve_*.csv` (τ, obs, H⁰, triangle diagnostic),
`summary.csv` (one row per task/emb/k/k_e with Ω, Ω_rel, null band),
`blocking_pairs_*.csv`; figures at 300 dpi: `fig_obstruction_vs_tau.png`,
`fig_nerve_*.png` (disagreement-weighted nerve, frustrated edges at the
summary τ highlighted), `fig_null_vs_observed.png`; `FINDINGS.md` (control
first, then IOI, kills, judgment calls with alternatives).

`e3lib/` = the reusable core: `explanation_obstruction(model, prompts,
task_fn, ...) → dict(obstruction, h0, blocking_pairs, nerve, per_tau, ...)`,
the future `whitebox.gluing`. Tests in `tests/` (planted frustrated
triangle, consistent cover → 0, two-mechanism split → 0 with H⁰ = 2,
brute-force agreement on random tiny nerves, EAP edge count).

## 10. Judgment calls (summary list)

1. Restriction = re-score and keep the top half of C_i on U_ij (alt: identity).
2. Top-k_e sections rather than absolute score threshold.
3. Membership rule relative to the max (θ = 0.5), adaptive to 0.33 only if the
   nerve has no cycles (alt: GMM).
4. PCA-20 before fuzzy c-means (alt: raw 768-d).
5. Obstruction = frustrated-cycle count with the same-label semantics
   (alt: two-colouring/Z/2 parity; correlation-clustering frustration index,
   which is NP-hard and ≤ ours).
6. Continuous version = integral of the curve = Σ lifetimes (alt: Hodge).
7. n_min = 10 for nerve edges/triangles (alt: 1).
8. Global bootstrap over prompt rows (alt: stratified within cluster).
9. Plain EAP (alt: EAP-IG).
10. Induction metric = raw logit of the target (brief's choice).

## Amendments

(none yet)

### Amendment 1 (2026-08-30, before any obstruction analysis; after the EAP caches and cover-statistics only)

**What was found.** Fuzzy c-means as specified in §3 (m = 2, PCA-20) is degenerate on
both tasks: every centre collapses onto the global mean and U ≡ 1/k exactly, at m = 2
and m = 1.5, with Dirichlet or data-point initialisation (induction: max_p u_p = 1/k to
three decimals for every k; IOI smoke: 0.100). The relative rule then puts every prompt
in every cluster (nerve = complete graph, all sections identical, J ≡ 1, obstruction
trivially 0 — a vacuous control). Fixing the centres with k-means++ and computing fuzzy
memberships around them exposes the opposite problem on IOI: the layer-11 and layer-6
mean embeddings are template embeddings — every k-means cluster is exactly one or two
(template, both orders) cells of 160/320 prompts, and the θ-rule (0.5 or 0.33, any m)
yields **zero** overlaps (nerve has no edges); on induction the same rule puts ~80 %
of prompts in two clusters (m = 1.5) — the θ-rule has no setting that works for both.

**Change (§3 replaced).** Cover = **2-nearest-centre cover**: centres from k-means++
(sklearn, n_init = 10, seed 0) in the PCA-20 space; U_i = { p : centre i is one of the
two nearest centres of p }. Every prompt lies in exactly two clusters; U_ij = prompts
whose two nearest centres are i and j. This is the θ → 0 limit of the fuzzy relative
rule restricted to rank ≤ 2, needs no fuzzifier, and is the same rule for both tasks.
Observed nerves (n_min = 10): IOI L11 k = 8/10/12/15 → |E| = 12/15/20/21, cycle rank
5/6/10/8; induction L11 → |E| = 26/39/48/68, cycle rank 19/30/37/54. Hard cover (kill ii)
= nearest centre only. Consequence: triple overlaps are empty by construction, so the
triangle diagnostic (§6.4) is vacuous under this cover (reported as such; alternative:
a 3-nearest cover, not run). The FCM implementation stays in `e3lib/cluster.py` for the
record. Nothing in §4–§8 changes.
