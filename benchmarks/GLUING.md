# `whiteboxtopo.gluing` — the E3 port (Stage 2B, 2026-08-30)

Input: E3 of whitebox-v2 (its result tables are copied verbatim into
`results/gluing/`; its code was ported, not imported — the library has no
dependency on `transformer_lens` or an EAP/circuit-discovery implementation
for this module). E3's own docs: `E3/SPEC.md` (pre-registration, with
Amendment 1 replacing the originally pre-registered fuzzy-c-means cover) and
`E3/FINDINGS.md` (verdict) in the whitebox-v2 experiment tree.

## What ships, and — more importantly — what does not

`whiteboxtopo.gluing.explanation_obstruction(scores, cover=None, embedding=None,
...)` — an **instrument**, not a claim. It takes a plain `(n_examples,
n_edges)` array of per-example attribution/explanation weights (EAP scores in
E3, but the function has no opinion about where they came from — feature
attributions, saliency weights, anything with one row per example and a fixed
column set across examples) plus either a precomputed overlapping cover
(membership matrix) or an embedding to build one from
(`cover_from_embedding`, 2-nearest-centre cover on k-means++ centres — SPEC
Amendment 1). It reports:

* `omega`, `omega_rel`, `omega_per_edge` — the frustrated-cycle obstruction
  mass on the nerve of the cover (SPEC §6: `Ω = Σ_e max(0, J_bott(e) − J_e)`,
  the total disagreement across nerve edges that no global cluster-to-
  mechanism relabelling can explain away).
* `null_lo`, `null_hi`, `null_mean`, `p_null` — the example-label shuffle
  null (permute which examples belong to which cluster, keep cluster sizes
  and the nerve fixed, recompute; 200 permutations by default).
* `omega_ci`, `J_ci`, `m_e_ci` — bootstrap CIs (row resampling with
  replacement; memberships travel with the resampled rows).
* `blocking_pairs` — the nerve edges (cluster pairs) ranked by frustrated
  lifetime `m_e`, with sizes, overlap size, `J`, `J_bott`, and bootstrap CIs.
* `cyclomatic` — the nerve's cyclomatic number (`|E| − |V| + #components`).
  **Frustration requires a cycle** — a "clean" split into two mechanisms
  along a tree-shaped (acyclic) nerve gives `Ω = 0` by construction (it is
  fully explained by a two-label partition); only a cycle with a
  non-2-colourable disagreement pattern produces positive `Ω`. This is
  worth internalising before reading any `Ω > 0` as "the clusters disagree":
  it specifically means the disagreement pattern cannot be explained by ANY
  assignment of cluster → mechanism label, not merely that some clusters
  disagree.
* `significant_vs_null` — `p_null ≤ 0.025` (Ω above the 97.5th percentile of
  the shuffle null). **This is not, by itself, a "distinct mechanisms"
  claim** — see below.

`explanation_obstruction` does **not** take a model, a set of prompts, or a
`task_fn` (PROGRAM.md's original sketch, `explanation_obstruction(model,
prompts, task_fn)`, is not what ships): E3's own verdict is that the pipeline
should be a diagnostic on precomputed scores/cover, not a scored circuit-
discovery-plus-verdict pipeline. The model-specific half of E3
(`e3lib/eap.py`, `e3lib/prompts.py` — GPT-2-small EAP scoring, IOI/induction
prompt generation) is **not ported**: it needs `transformer_lens` and a live
model, which this library does not depend on, and — per the verdict below —
porting the scoring half without also porting the honest failure account
would invite exactly the overclaim E3's own FINDINGS.md warns against.

## Why there is no scored "distinct mechanisms" detector

E3 pre-registered a matched single-mechanism control (induction: predict the
repeated token, a task with one plausible circuit) specifically so that a
positive `Ω` on the real object of study (IOI, believed to have multiple
template-specific mechanisms) could be trusted. §7's own pass/fail gate is
explicit: **H-E3-control "must pass before IOI is believed."** It did not.

* **The control failed its own null-band test.** At the pre-registered
  summary configuration (`k=10, k_e=200`, layer-11 embedding), induction's
  `Ω = 1.517` sat *outside* its own shuffle-null band `[0.564, 0.991]`
  (`p_null = 0.000`) — not merely non-zero, but statistically distinguishable
  from its own null in the wrong direction. Across the full 40-configuration
  induction grid (2 embeddings × 4 k × 5 k_e), this held in **34/40 cells**
  (85%). The absolute-magnitude floor (`Ω/|E| ≤ 0.05`) held everywhere — the
  instrument is not "broken" in the sense of producing nonsensical numbers —
  but the null-band criterion, the one that licenses believing a positive
  reading means something, failed almost everywhere.
* **The direction inverted, in every cell.** IOI — the task with the
  genuinely non-trivial mechanism structure — showed *smaller* raw `Ω` than
  the (failed) control in **40/40** grid cells, and smaller `Ω_rel` in 35/40.
  IOI's own summary-config reading (`Ω = 0.224`, `p_null = 0.005`) does
  clear its own null in isolation, and two of its blocking pairs have
  bootstrap CIs on `m_e` excluding 0 — the literal H-E3-IOI criteria are met
  — but this cannot be read as "IOI has genuine mechanism heterogeneity and
  induction doesn't" when induction shows *more* of the same statistic. A
  plausible mechanical explanation (E3 FINDINGS.md §4): induction's cover
  (built from an unsupervised embedding that does not separate the one
  variable that plausibly matters, sequence length) is denser (cyclomatic
  rank 30 vs IOI's 6 at matched k), and `Ω`'s cycle-counting construction
  gives a denser nerve more independent cycles to accumulate frustrated
  lifetime on — a scope limitation of `Ω` as constructed, not evidence that
  induction truly has multiple mechanisms.
* **Localisation fails an embedding-robustness check.** IOI's top blocking
  pair under the layer-11 embedding (clusters keyed on templates {13, 14})
  does not appear — by pair identity or by template/order composition —
  among the top-3 blocking pairs found under a second, independent embedding
  (layer 6). The claim "these are distinct mechanisms located in prompt
  space" is not made for IOI under this pipeline.

**Bottom line (E3 FINDINGS.md): the instrument does not discriminate a
genuine single-mechanism task from IOI in the pre-registered direction, and
the one task where localisation is even nominally testable does not survive
the embedding-robustness kill.** Per the programme's rule 3 this is shipped
as a negative, not suppressed — hence a diagnostic instrument, not a scored
detector.

`GluingReport.caveat` (also exported as `whiteboxtopo.gluing.CAVEAT`) states this
in the object itself, not only in documentation someone might skip:

> A positive `significant_vs_null` here only supports "distinct mechanisms
> located in example space" if BOTH (a) a matched single-mechanism control
> run on the SAME architecture/embedding also lands inside ITS null band, and
> (b) the top blocking pair's identity or composition survives a second,
> independent embedding choice. Neither held in E3's own battery.

There is no `verdict` field, no `localisation` claim generator, and no
"`Ω > threshold` ⇒ multiple mechanisms" gate anywhere in the public API —
checked directly by `tests/test_gluing.py`
(`test_report_has_no_mechanism_verdict_field_and_carries_the_caveat`).

## The E3 battery (GPT-2 small, IOI vs an induction control)

Summary configuration for all headline numbers: `k=10, k_e=200`, layer-11
mean-pooled residual-stream embedding, 2400 prompts/task, 32 491-edge EAP
graph (heads + MLPs, Hanna et al. 2024 / Syed, Rager & Conmy 2023 / Conmy et
al. 2023), 200-permutation shuffle null, 200-resample bootstrap.

| Task | Ω | null band | p_null | Ω/\|E\| | \|E\| | cyclomatic | Result |
|---|---:|---|---:|---:|---:|---:|---|
| induction (control) | 1.517 | [0.564, 0.991] | 0.000 | 0.039 | 39 | 30 | **FAILS H-E3-control (b)** — above its own null |
| ioi | 0.224 | [0.020, 0.194] | 0.005 | 0.015 | 15 | 6 | Clears its own null, but *smaller* raw Ω than the failed control |

Two IOI blocking pairs clear a bootstrap CI excluding 0 on `m_e`: (4,5)
`m_e=0.0899` CI `[0.018, 0.172]` (clusters keyed on templates {13,14}) and
(1,2) `m_e=0.0190` CI `[0.019, 0.159]` (clusters keyed on templates {6,7}).
Neither survives the layer-6 second-embedding check (top-3 layer-6 pairs
share no template/order composition with (4,5)).

Battery record copied here: `results/gluing/summary.csv` (80 rows: 2 tasks ×
2 embeddings × 4 k × 5 k_e), `results/gluing/pairs.csv` (per-nerve-edge J,
J_raw, J_bott, m_e, bootstrap CIs for every grid cell). `tests/test_gluing.py`
reproduces the two summary-config headline rows and the top IOI blocking
pair directly from these files. Full figures and the 80-row grid, plus the
kill-attempt account (prompt-label shuffle, hard cover, bootstrap, second
embedding): `E3/FINDINGS.md` and `E3/results/` in the whitebox-v2 experiment
tree (not copied here in full — only the CSVs the test suite reads).

## Judgment calls made porting E3 → the library

* **API shape changed from PROGRAM.md's sketch.** `explanation_obstruction(model,
  prompts, task_fn)` is not what ships; the library takes precomputed
  `(scores, cover_or_embedding)`, matching the house pattern
  (`whiteboxtopo.darkmatter` takes error matrices, not a model + SAE). The
  model-scoring half (EAP on a live GPT-2) stays in `E3/e3lib/eap.py`.
* **Cover method default is `"knn2"` (2-nearest-centre), not the originally
  pre-registered fuzzy c-means.** SPEC Amendment 1 found fuzzy c-means
  degenerate on real embeddings (every centre collapses onto the global
  mean). `method="fcm"` is kept (`whiteboxtopo.gluing.cover_from_embedding`) for
  callers whose data does not hit that degeneracy, with the same adaptive
  theta-lowering fallback.
* **Triangle diagnostic is vacuous under the default cover.** The
  2-nearest-centre cover puts every example in exactly two clusters, so
  triple overlaps — and the transitivity diagnostic built on them — are
  empty by construction. `triangle_frustration` returns an all-NaN array in
  this case; reported, not hidden (E3 Amendment 1 flagged this before any
  obstruction analysis was run).
* **`k_e` is silently clipped to `n_edges`** if a caller passes a larger
  section size than the number of columns in `scores` — a defensive clip,
  not present in E3's script (which always had 32 491 columns to spare).
* **No `edge_names` / distinguishing-edges output.** E3's `blocking_pairs`
  could name the ten EAP edges that most distinguish two clusters' circuits
  (`only_in_i`/`only_in_j`, using the model's edge-name table). The library
  version has no model, hence no edge names — it reports the numeric column
  indices' set difference is available via `rep.J`/section internals but is
  not surfaced as a named field; a caller who has its own column-name list
  can index into `rep.edges` and the underlying per-cluster section sets
  directly if this is needed (not currently a public accessor — flagged in
  case a future user asks for it).
* **Planted test constructions required a deliberate asymmetric bridging
  overlap**, not merely "two clusters, two circuits, one shared overlap":
  a cover whose nerve is a tree, or whose crossing edges are symmetric
  (identical composition on both sides of a mechanism boundary), gives
  `Ω = 0` by construction — a clean two-mechanism split is explained by a
  two-label partition, not frustrated. Genuine frustration needs a cycle in
  the nerve AND an asymmetry across it (see `tests/test_gluing.py`'s
  `_two_mechanism_cover` helper and its docstring for the exact construction
  and why it works). This mirrors, in miniature, E3's own finding that real
  disagreement patterns are not simply "which mechanism," they are about
  whether the *pattern* of agreement/disagreement is 2-colourable at all.

## Scope

One battery: GPT-2 small (124M), two tasks (IOI, induction), one embedding
family (mean-pooled residual stream at two layers), one cover construction
(2-nearest-centre on k-means++ centres, PCA-20). Nothing here has been
checked on a second model, a second attribution method (EAP only), or a
cover built any other way than E3's Amendment-1 construction. The math core
(`whiteboxtopo.gluing.core`) is validated independently of any of that (brute-
force cycle enumeration against the closed-form bottleneck construction on
60 random small nerves, plus a hand-built cycle graph) — the scope limits
above are about what a real attribution-score battery has actually been
checked to mean, not about whether the arithmetic is correct.
