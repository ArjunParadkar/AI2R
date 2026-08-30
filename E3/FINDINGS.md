# E3 — Empirical explanation cohomology: FINDINGS

**Status: COMPLETE (2026-08-30).** One run (`run_e3.py`) reusing cached
per-prompt EAP scores (`cache/full/scores_{induction,ioi}.npz`, N = 2400 each,
32 491-edge graph). The original run was killed by a 13:14 reboot; it was
relaunched at 14:16 against the same cache and finished at 14:50 ("done in
34.1 min", `results/run.log`). `run.log` is append-only and contains both the
killed partial pass and the complete relaunch; every number below is read from
the final CSVs (`results/summary.csv`, `pairs.csv`, `obstruction_curves.csv`,
`blocking_pairs.json`, `covers.csv`), all written once at the end of the
completed run, plus the 8 figures. Grid: task ∈ {induction, ioi} × emb ∈
{11, 6} × k ∈ {8, 10, 12, 15} × k_e ∈ {50, 100, 200, 400, 800} = 80 rows, all
present. Summary config (pre-registered): k = 10, k_e = 200, embedding =
layer-11 mean, θ = 0.5 (Amendment 1's 2-nearest-centre cover; see below).
Tests: `tests/test_e3.py` 6 passed, 1 skipped (`test_eap_runs_on_model`,
gated behind `E3_MODEL_TEST=1`, correctly skipped — it loads GPT-2 and this
machine has a concurrent heavy E1 job).

Sanity check on the underlying task/metric (before any cover analysis, from
`run.log`): induction clean-metric 15.38 ± 3.79 vs corrupt 0.16, clean >
corrupt in 2400/2400 prompts; IOI clean 3.88 ± 1.82 vs corrupt 0.05, clean >
corrupt in 2264/2400 (94.3 %) — both tasks and corruptions behave as
expected; the EAP scores being fed into the sheaf analysis are not garbage.

## Verdict

| Hypothesis | Pre-registered criterion (§7) | Result |
|---|---|---|
| **H-E3-control** | (a) Ω/\|E\| ≤ 0.05 **and** (b) Ω inside the central 95 % shuffle-null band, at k=10,k_e=200,emb=11 | **FAILS on (b).** (a) holds everywhere (see below). (b): Ω = 1.517, null band [0.564, 0.991], p_null = 0.000 — Ω is *above* the null, not inside it. Across the full 40-config induction grid, (a) holds 40/40 but (b) holds only 6/40 (15 %); 34/40 configs have p_null ≤ 0.025 (significantly above null). |
| **H-E3-IOI** | Ω_IOI above the 97.5th pctile of its own null, **and** ≥1 blocking pair with bootstrap CI on m_e excluding 0, **and** (stated in the same hypothesis) Ω_IOI > Ω_induction at matched k,k_e | The first two literal criteria are met at summary config (p_null = 0.005; pairs (4,5) m_e=0.0899 CI[0.018,0.172] and (1,2) m_e=0.0190 CI[0.019,0.159] both exclude 0). **The direction clause fails in 40/40 grid cells** — Ω_induction > Ω_IOI at *every* (emb,k,k_e), and Ω_rel_induction > Ω_rel_IOI in 35/40. Given H-E3-control's own failure, this is reported as an **overall FAIL / uninterpretable positive**, not a clean win — see discussion. |
| **Localisation** (blocking pair reflects a real, embedding-robust mechanism split) | top pair (or same template×order composition) among top-3 blocking pairs under the kill-iv layer-6 embedding | **FAILS.** Layer-11 top pair (4,5): cluster 4 = pure templates {13,14} (both orders ~50/50), cluster 5 = mixed multi-template with template 13 as the only overlap component. Layer-6 top-3 pairs are (2,4) clusters {6,7}|mixed, (4,5) mixed|{8,9}, (4,6) mixed|mixed — template 13/14 does not appear in any of them. No overlap in composition. |

**Bottom line: the instrument does not discriminate a genuine single-mechanism
task from IOI in the direction the pre-registration expected, and the one
task where localisation is even nominally testable (IOI) does not survive the
embedding-robustness kill.** Per rule 3 this is written up as a negative, not
suppressed.

## 1. The control (induction) — read first, per SPEC's own ordering

§7's gate is explicit: H-E3-control "must pass before IOI is believed." It
does not pass. Condition (a) — the absolute-magnitude floor Ω/\|E\| ≤ 0.05 — is
satisfied everywhere on the induction grid (max observed 0.047, at k=12,
k_e=50), so the instrument is not "broken" in the sense SPEC defines breakage
(huge nonsensical Ω). But condition (b) — Ω statistically indistinguishable
from the prompt-shuffle null — fails in 34/40 configs, including the
pre-registered summary config (Ω = 1.517 vs null band [0.564, 0.991], p =
0.000). The induction cover produces a nerve with cyclomatic number 30 (39
edges, 10 vertices) at k=10, and 25 of those 39 edges are frustrated
(`n_frustrated_edges` in `summary.csv`); at τ = 1 the curve shows obs = 16,
H⁰ = 3 components (`obstruction_curves.csv`).

**Why a single-mechanism task shows a significant, non-null Čech obstruction.**
`covers.csv` shows the induction clusters (layer-11 embedding, k=10) are
*not* organized by the one variable that plausibly matters mechanistically
(sequence length L ∈ {6,8,10,12}): every cluster contains a near-uniform mix
of all four L values (e.g. cluster 0: L=12:171, L=10:145, L=8:121, L=6:77 out
of 514 — no cluster is more than ~35 % any single L). Contrast IOI, whose
layer-11/layer-6 clusters *are* close to pure (template, order) cells (per
Amendment 1 and `blocking_pairs.json` compositions below) — the embedding
finds the mechanistically relevant structure for IOI but, for induction,
finds something else (most likely token-identity/content variance in the
random repeated sequence, which has nothing to do with the induction
algorithm). The sheaf-obstruction pipeline is then measuring how sensitive
the top-k_e EAP edge set is to *that* nuisance axis of the mean-pooled
residual-stream embedding — and it is measurably, non-null sensitive to it.
This is a real limitation of the pipeline, not a coding bug: EAP-derived
per-prompt edge scores, aggregated over a cluster and top-k'd, are not
perfectly stable under an arbitrary re-partition of "the same" mechanism's
prompts, and the prompt-shuffle null (which keeps cluster *sizes* and the
nerve fixed but destroys the correlation between membership and the actual
per-prompt score pattern) is evidently a weaker null than "this exact
partition axis is nuisance" — shuffling loses the correlation structure
entirely, while the real cover retains within-cluster correlation on
whatever axis it grouped by, inflating apparent disagreement above the
shuffle floor even when the true generative mechanism is single.

Kill ii (hard cover / nearest-centre only, no overlaps) gives obs ≡ 0
trivially in all 80 configs (`hard_cover_edges` = 0 throughout, by
construction of the 2-nearest cover — Amendment 1) — this only confirms the
cover-with-overlaps machinery is doing something, it does not rescue (b).

## 2. IOI — the measurement, on its own nominal terms

At the summary config (emb=11, k=10, k_e=200): nerve has 15 edges,
cyclomatic 6 (10 vertices, sparser than induction's 39-edge/cyclomatic-30
nerve at the same k — see §4 for why this matters). mean_J = 0.954,
Ω = 0.224 (bootstrap CI [0.186, 0.460]), Ω_rel = 0.324, Ω/\|E\| = 0.0149,
p_null = 0.005 (null band [0.020, 0.194]) — Ω is above the 97.5th percentile
of the shuffle null, satisfying the first literal clause of H-E3-IOI.
4/15 edges are frustrated; max obs(τ) = 2; H⁰ collapses from 10 (τ=0) to
1 component at τ ≤ ~0.85 and back up to 6 at τ = 1 (`obstruction_curves.csv`).

Two blocking pairs have bootstrap 95 % CIs on m_e excluding 0
(`pairs.csv`, `blocking_pairs.json`):

| pair | J | m_e | CI | overlap n | composition |
|---|---|---|---|---|---|
| (4,5) | 0.852 | 0.0899 | [0.018, 0.172] | 15 | cluster 4 = templates {13,14} only (order ~50/50); cluster 5 mixed, overlap driven by template 13 |
| (1,2) | 0.942 | 0.0190 | [0.019, 0.159] | 40 | cluster 1 = templates {6,7}; cluster 2 mixed, overlap ~55 % template 7 / 45 % template 6 |

satisfying the second literal clause of H-E3-IOI. Direction across the whole
grid, however, is uniformly wrong: at every one of the 40 (emb,k,k_e) cells,
raw Ω_induction > Ω_IOI (0/40 cells favour IOI); Ω_rel favours IOI in only
5/40 cells (all at k ≥ 12, where induction's nerve is largest and densest).
Fraction of configs with p_null ≤ 0.025 (significantly above null): 34/40
for induction vs 28/40 for IOI — i.e. the "single mechanism" control is
*more* often significantly non-null than the "multi-mechanism" target task.

**H-E3-IOI is therefore not treated as a clean positive.** SPEC states the
direction inequality as part of the same registered hypothesis, not as a
separate nice-to-have; failing it in every grid cell, combined with the
control's own failure, means we cannot attribute IOI's above-null Ω to
genuine mechanism heterogeneity as opposed to the same cover-geometry /
EAP-sampling sensitivity that inflates the control. This is exactly the
scenario SPEC anticipated needing an honest write-up for ("IOI is
indistinguishable from its null is a deliverable") — except the actual
failure mode is subtler than that sentence anticipated: IOI is *not*
indistinguishable from its own null, but it is smaller in every respect than
a task that should have been the ≈0 anchor.

## 3. Localisation kill (kill iv, §7 "Localisation")

Top blocking pair under layer-11 (4,5), keyed on templates {13,14}, does not
appear (by pair identity or by template/order composition) among the top-3
layer-6 blocking pairs — see verdict table. The claim "these are distinct
mechanisms located in prompt space" is **not made** for IOI under this
pipeline; SPEC's own bar for making it is not cleared.

## 4. Why induction's nerve is denser than IOI's, and what that implies for Ω

At matched k = 10: induction |E| = 39, cyclomatic 30; IOI |E| = 15,
cyclomatic 6 (both layer-11). Ω is, by construction (§6.3), a sum over edges
of frustrated lifetime, and the number of *possible* frustrated cycles scales
with cyclomatic rank. Induction's 2-nearest-centre cover (Amendment 1)
produces near-uniform, heavily overlapping ~400-550-prompt clusters (dense
nerve), while IOI's cover produces template-locked, more disjoint clusters
(sparse nerve) because the embedding genuinely separates IOI templates but
does not separate induction's L values or find any other low-dimensional
structure to lock onto. A denser nerve gives Ω more independent cycles to
accumulate lifetime on even under equal per-edge noise, which is a plausible
mechanical contributor to induction's larger raw Ω *independent of* whether
either task has "real" mechanism heterogeneity — this is a scope limitation
of the Ω statistic as constructed (§6.3, alternative-not-used: a
cyclomatic-normalised Ω, not pre-registered, not computed here) rather than
a claim that IOI glues while induction doesn't.

## 5. Kill attempts (§8, run for the full grid, not just IOI)

i. **Prompt-label shuffle** (200 perms) — is the null band reported
   throughout; see §1/§2. Does not collapse the induction positive to zero
   in 34/40 configs — the kill fails to kill the control's spurious signal,
   which is itself the headline negative.
ii. **Hard cover** — obs ≡ 0 in all 80 rows (`hard_cover_edges` = 0
    throughout); degenerate check behaves as SPEC requires.
iii. **Bootstrap** (200 resamples) — CIs reported on every J_e, m_e, Ω
     (`pairs.csv` m_ci_lo/hi, `summary.csv` omega_ci_lo/hi). Used above to
     support the two blocking pairs' m_e > 0 claim.
iv. **Second embedding** (layer-6, full k sweep) — computed for both tasks;
    used for the localisation kill (§3), which fails.

Triangle diagnostic (§6.4): **vacuous by construction.** `n_tris` = 0 in all
80 summary rows — Amendment 1's 2-nearest-centre cover puts every prompt in
exactly 2 clusters, so triple overlaps are empty by construction. This was
already flagged in Amendment 1 before any obstruction analysis; nothing new
to add here except confirming it held for the full completed run.

## Judgment calls / SPEC status

No new amendment is added. Amendment 1 (2-nearest-centre cover, replacing
degenerate fuzzy c-means) was made *before* this analysis and is unchanged by
it. Everything reported here is a result under the pre-registered pipeline as
amended, not a further methodological change — the negative findings above
(control fails condition (b); direction reverses; localisation fails) are
written up as data (rule 3), not papered over with a post-hoc method change.
One thing worth flagging explicitly for anyone re-running this line of work:
SPEC's implicit assumption that "one algorithm ⇒ one edge-level EAP circuit
across all input variations" is, on this evidence, not safe for induction at
GPT-2 scale — sequence-length/content variation alone is enough to move the
top-200 EAP edge set around by more than the shuffle-null floor. A future
SPEC should either (a) build the induction control's cover explicitly on L
(the known mechanistic variable) rather than on an unsupervised embedding
that may lock onto nuisance structure, or (b) normalise Ω by nerve cyclomatic
rank before comparing tasks with different cover topologies.

## Library port verdict (PROGRAM.md Stage 2 — `whitebox.gluing`)

**Do not ship `explanation_obstruction(model, prompts, task_fn) →
(obstruction score, blocking pairs)` as a validated "your explanation doesn't
generalize" detector.** The measurement machinery itself is sound and
verified: the frustrated-cycle / bottleneck-agreement construction (§6.2)
passes brute-force cycle enumeration on 60 random small nerves and two hand-planted
cases (`tests/test_e3.py`, all green), the EAP graph is the correct
32 491-edge count, and the cover/null/bootstrap pipeline runs end-to-end on a
real model with a real task effect. What it measures, however, is not what
Stage 2 needs it to measure: on the one task that is supposed to be the
sanity anchor (single-mechanism induction), it reports a statistically
significant, non-null obstruction that is *larger* than the one it reports
for IOI (the task believed to have genuinely multiple template-specific
mechanisms) in every one of 40 grid cells. That is the single most important
number in this experiment, and it means Ω as constructed here cannot
currently distinguish "the cover found real mechanism heterogeneity" from
"the cover found nuisance structure and the EAP+top-k+restriction pipeline is
sensitive to it, with a magnitude that scales with the nerve's cyclomatic
rank." Reporting a bare Ω to a user without also reporting (i) a matched
single-mechanism control, (ii) the prompt-shuffle null, and (iii) the
cyclomatic rank of the nerve, would overclaim.

If ported at all for Stage 2, it should ship as a **diagnostic instrument,
not a scored signal**: `explanation_obstruction()` returns Ω, its shuffle
null band, p_null, blocking pairs with bootstrap CIs, and the nerve's
cyclomatic rank — and the docstring/CLAIMS.md entry must say explicitly:
"a positive Ω only supports 'distinct mechanisms located in prompt space' if
(a) a matched single-mechanism control on the same architecture/embedding
also gives Ω within its null band, and (b) the top blocking pair's identity
or composition survives a second embedding choice — neither held here for
GPT-2-small/IOI." Do not ship a `localisation` claim generator; do not ship a
bare "Ω > threshold ⇒ multiple mechanisms" API. The honest scope statement
for CLAIMS.md: *measured, on GPT-2-small, IOI vs induction, k∈{8,10,12,15},
k_e∈{50,...,800}, two embeddings — the control fails its own pre-registered
null-band test, and the direction of the IOI vs. induction comparison is
opposite of the pre-registered expectation in 100 % of grid cells; the
localisation claim additionally fails an embedding-robustness check.
Novelty claim stands (first concrete Čech-style obstruction numbers on a real
model, per PROGRAM.md) but the numbers argue against, not for, believing this
particular pipeline's positive on IOI.*

Files: `results/summary.csv` (80 rows), `results/obstruction_curves.csv`,
`results/pairs.csv`, `results/covers.csv`, `results/blocking_pairs.json`,
`results/fig_obstruction_vs_tau.png`, `results/fig_null_vs_observed.png`,
`results/fig_sweep_omega_rel.png`, `results/fig_nerve_{induction,ioi}_emb{11,6}_k10_ke200.png`,
`results/run.log`, `tests/test_e3.py` (6/6 non-model tests green).
