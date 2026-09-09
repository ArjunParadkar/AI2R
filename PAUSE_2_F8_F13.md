# PAUSE 2 — Block F, the audit (F8–F13) — FINAL 2026-09-09 (F8–F10 on the full enumeration; F11 final; F12 HPCC; F13 shipped)

## 1. Plain English
The audit ran on all 3 143 enumerated specimens (F8: every renaming exact to 1.1e-13; F9: the weight-product anchor
invariant to 2.7e-14 on every specimen) and on GPT-2 small (F11). The table says the same thing in both places: quantities
computed from the *function* (weight products, round-trip invariant polynomials, loop products, persistence of the
per-unit-standardised cloud) are exactly invariant; quantities indexed by unit labels (per-unit activations,
gradient×input attributions, top-k unit sets, top principal directions, weight norms) are artifacts of the labelling and —
except weight norms — become exactly invariant after a gauge alignment that undoes the permutation *and* the scaling
(Hungarian + standardisation); undoing only the permutation leaves a clean covariant of degree 1, and Procrustes on the unit
axis repairs nothing. Two entries the SPEC had filed under "expected invariant" are not: the Frobenius invariant polynomial
tr(WᵀW) and the raw-cloud persistence summaries are permutation-invariant but move under per-unit scalings (persistence
as a degree-2 covariant, as E2 found). Those deviations are reported with both classes, as the F10 rule requires, not
re-filed.

## 2. Pre-registered thresholds and outcomes
| Item | Threshold | Outcome |
|---|---|---|
| F8 identity | every renaming exact | 3 143/3 143 specimens, 300 renamings each, max deviation 1.1e-13 — **met** |
| F9 anchor | weight-product CV < 1e-6 on every specimen | 3 143/3 143, max CV 2.7e-14 — **met** |
| F10 | expected classes as listed; deviations reported with both classes | 13 pre-registered candidates + 5 alignment columns: 9 as expected, **4 deviations** (below) — **met as specified** (deviations reported) |
| F11 | flips vs toy flagged | 3 flips, all flagged in §3 |

**F10 table (`results/F10_summary.csv`; n = specimens with the candidate defined; agreement = fraction of specimens
with the majority verdict; medians of per-specimen CV under perm / scale / combined):**

| candidate | n | majority | agreement | cv perm | cv scale | exponent (R²) | expected | note |
|---|---|---|---|---|---|---|---|---|
| weight_product | 657 | invariant | 1.000 | 1e-15 | 1e-15 | — | invariant | anchor |
| inv_poly_roundtrip | 657 | invariant | 1.000 | 2e-16 | 3e-16 | — | invariant | |
| loop_ratios | 657 | invariant | 1.000 | 2e-16 | 3e-16 | — | invariant | |
| ph_std_probe | 657 | invariant | 1.000 | 4e-16 | 4e-16 | — | invariant | E2's scaling repair |
| inv_poly_gram_det | 657 | covariant | 1.000 | 2e-15 | 4.5 | 3.000 (1.00) | invariant | **deviation**: det Gram is homogeneous of degree 2·(units) in the scalings; SPEC expectation wrong, class = covariant |
| inv_poly_frobenius | 657 | artifact | 1.000 | 1e-15 | 1.09 | 0 (n/a) | invariant | **deviation**: tr(WᵀW) is perm-invariant but per-unit scalings move it with no single power law; class = perm-invariant / scale-artifact |
| ph_raw_probe | 657 | covariant | 0.976 | 3e-16 | 0.15 | 2.000 (1.00) | invariant(perm) / measured(scale) | measured: degree-2 covariant (Z5 law); 2.4 % invariant (degenerate diagrams) |
| ph_raw_truthtable | 657 | covariant | 0.892 | 0 | 0.82 | 2.000 (1.00) | invariant(perm) / measured(scale) | 10.8 % invariant (truth-table clouds of 8–16 points are often degenerate) |
| singular_values | 657 | covariant | 1.000 | 1e-15 | 0.88 | 1.000 (1.00) | covariant | |
| per_unit_activation | 2 300 | artifact | 1.000 | 0.84 | 1.17 | — | artifact | aligned (Hungarian + std): **invariant 1.000** |
| per_unit_activation_hungarian | 1 643 | covariant | 0.991 | 1e-15 | 1.17 | 1.000 (1.00) | — | permutation undone, scaling left → covariant k = 1 |
| per_unit_activation_hungarian_std | 1 643 | invariant | 1.000 | 1e-15 | 1e-15 | — | — | full gauge alignment |
| per_unit_gradxinput | 657 | artifact | 1.000 | 2.24 | 1e-15 | — | artifact | scale-invariant (attribution = activation × gradient cancels the unit scale), perm-artifact |
| topk_units | 657 | artifact | 1.000 | 0.28 | 0 | — | artifact | |
| weight_norms | 657 | artifact | 1.000 | 0.35 | 1.30 | 0.59 (0.86) | artifact | no alignment repairs it (F11 agrees) |
| top_direction | 1 643 | artifact | 1.000 | 1.98 | 2.16 | — | artifact | |
| top_direction_procrustes | 1 643 | artifact | 0.984 | 2e-15 | 1.59 | — | (align column) | Procrustes undoes the permutation only; scaling remains → still artifact |
| top_direction_hungarian_std | 1 643 | invariant | 0.980 | 1e-15 | 1e-15 | — | (align column) | 2 % artifact: ties in the leading direction |

## 3. Kill-test outcomes
* **Kill F11 (flips between the MLP toy and GPT-2).** Three, all explained by the *group*, none by the auditor:
  (i) the MLP-hidden persistence summaries are scale-**covariant** (k = 2) on ReLU MLPs but **invariant** on GPT-2 —
  GPT-2's sampled scalings are LayerNorm-compatible rescalings that are compensated before the GELU (the hidden cloud
  itself does not move), whereas a ReLU unit's positive scaling is a symmetry that rescales its coordinate; (ii) singular
  values of the layer map are a degree-1 **covariant** on MLPs (uniform-α power law) but an **artifact** on GPT-2 (0.14)
  — the ln_2 rescaling is a per-row diagonal, not a uniform α, so no single power law exists; the LN-folded singular values
  are invariant, which is the fold the audit identifies; (iii) head-z quantities (per-head norms, head-z persistence)
  exist only on GPT-2 and are O(d_head)- but not GL(d_head)-invariant (2–5 %). Everything else agrees: per-unit
  quantities artifact → invariant after alignment in both; weight norms artifact in both, unrepaired by alignment in both.
* The 4 F10 deviations (§2) are SPEC-expectation errors, not auditor errors: each is reproduced by hand (a per-unit scaling
  by c multiplies tr(WᵀW)'s column by c²; det Gram by ∏c²; TC by α² under uniform α).

## 4. What contradicts the plan's framing
* The handbook's "expected invariant: invariant polynomials" was wrong for two of the four polynomials; the audit is
  what caught it. The class of "invariant polynomials" that survives is the one built from *round trips* (products that
  return to the same unit index) — exactly the class the gauge group cannot touch.
* "Persistence summaries expected invariant under permutation" was right, but the more useful statement is that they
  are *covariant of degree 2 under scaling* on ReLU nets and must be standardised (`ph_std_probe`) before any
  cross-network claim — which is the E2 result, now on 657 specimens.
* Procrustes alignment (the handbook's suggested repair for direction features) does not work on ReLU nets; Hungarian +
  standardisation does. `check_invariance(align="procrustes")` stays available but the docs say so.

## 5. Novelty
`experiments/F_gauge/NOVELTY.md` searches 1–3 (2026-09-07) + NOVELTY_BOARD #2 (2026-09-09): the audit is OPEN as the
empirical instantiation of 2601.12913 (Barbiero et al., symmetries as the definition of actionable interpretability)
against the exact group, with 2605.27836 and 2506.18053 as the empirical neighbours.

## 6. Claimable / not claimable
* Claimable: the audit table (F10 + F11) as a *mechanistic Implementation-Invariance* result — with the explicit
  statement that verdicts are conditional on the sampled group and that "covariant" needs a one-parameter family.
* Not claimable: anything about models with GELU/gated units beyond GPT-2's LN-compatible subgroup; F12 (frontier) until
  the HPCC run; a "which alignment to use" rule beyond ReLU MLPs (Hungarian + std) and GPT-2 (unit inverse + head inverse).
* F13 (ship): `whiteboxtopo.symmetry` + `invariants.audit_quantities` are public with VALIDATION.md and CLAIMS.md rows;
  NOVELTY.md carries the two handbook searches with dates and hits.

## F11 — GPT-2 small (final, unchanged from the 2026-09-08 draft)



## F11 — GPT-2 small, layer 6, 2 000 OpenWebText tokens, n = 30 renamings per kind (results/F11_audit_table.csv)
**Renamings actually sampled** (`symmetry.gpt2`, documented in the backend docstring): head permutations; within-head
GL(d_head) changes of basis (Q,K by R, R^{-T}; V by R, O by R^{-1}); MLP-unit permutations; LayerNorm-compatible
rescalings ln_1 → c_attn rows and ln_2 → c_fc rows. **Not sampled** (not exact): scalings through GELU (not positively
homogeneous) and ln_f (lm_head tied to wte).
**Identity check.** max |Δlogit| in fp32 over 2 000 tokens: perm 2.8e-3, scale 3.6e-3, rot 4.7e-3, combined 2.9e-3.
The SPEC asked for < 1e-3; the measured floor is 3–5× above it. Stated as a miss of the pre-registered number: the
renamings are exact in exact arithmetic (S1 verified 1e-14 on MLPs and 4e-15 on the toy transformer), and on GPT-2 the
deviation is the fp32 rounding of LN rescalings by factors up to 10 and of GL(64) head changes of basis on logits of
order 10²; every metric classified `invariant` below has CV ≤ 1e-5, four orders below the identity floor, so no verdict
hinges on it. A 1e-3 requirement would need fp64 or scale range narrower than [0.1, 10]; not rerun.

| Candidate | perm | scale | rot | combined | verdict | aligned |
|---|---|---|---|---|---|---|
| logits (sample) | 2e-7 | 2e-7 | 3e-7 | 3e-7 | invariant | — |
| PH summary, residual PCA-20 | 2e-8 | 2e-8 | 4e-8 | 6e-8 | invariant | — |
| PH summary, MLP-hidden PCA-20 | 3e-8 | 2e-8 | 9e-8 | 9e-8 | invariant | — |
| PH summary, MLP-hidden standardised PCA-20 | 3e-8 | 2e-8 | 6e-8 | 7e-8 | invariant | — |
| PH summary, concatenated head outputs z PCA-20 | 6e-8 | 5e-8 | **0.021** | 0.024 | artifact (GL only) | — |
| PH summary, head-0 z | **0.26** | 6e-8 | 0.040 | 0.31 | artifact | — |
| singular values W_fc | 1e-5 | **0.14** | 0 | 0.14 | artifact (covariant under ln_2 scaling) | — |
| singular values W_proj | 1e-5 | 0 | 0 | 1e-5 | invariant | — |
| singular values ln_2-folded W_fc | 1e-5 | 1e-5 | 0 | 1e-5 | invariant | — |
| per-unit MLP activation | **0.96** | 2e-7 | 5e-7 | 0.96 | artifact | invariant (5e-7) |
| per-unit gradient×input | **7.6** | 4e-6 | 6e-6 | 7.7 | artifact | invariant (6e-6) |
| top-10 unit set | **0.25** | 0 | 0 | 0.29 | artifact | invariant (0) |
| per-unit weight norms | 0.15 | **0.14** | 0 | 0.21 | artifact | artifact (0.14: scaling survives Hungarian matching) |
| top principal direction, MLP-hidden | **3.3** | 5e-7 | 2e-6 | 3.3 | artifact | invariant (2e-6) |
| per-head output norm | **0.31** | 1e-7 | 0.054 | 0.31 | artifact | artifact (0.056 = GL basis change; O(d_head)-invariant only) |
| per-head output norm, sorted | 1e-7 | 1e-7 | 0.048 | 0.048 | artifact (GL only) | — |
| head QK top-5 singular values, sorted | 0 | 8e-6 | 8e-6 | 8e-6 | invariant | — |
| head OV top-5 singular values, sorted | 0 | 8e-6 | 8e-6 | 8e-6 | invariant | — |
| attention entropy per head | **0.34** | 7e-8 | 5e-7 | 0.32 | artifact | invariant (7e-7) |

**Readings.** (i) Every cloud-level persistence summary on the residual stream and the MLP-hidden layer is invariant to
1e-7 — permutations and LN rescalings do not move them (the residual stream is not acted on at all; the MLP-hidden cloud
is only permuted). (ii) Head-level quantities are invariant only under head permutation + *orthogonal* basis changes;
under the GL(d_head) changes of basis the handbook allows, head-output norms, entropies of nothing (entropy is exactly
invariant, as it must be: attention patterns are GL-invariant) — but z-norms and z-cloud persistence move by 2–5 %. This
is the one place the audit's group is *larger* than what practitioners assume: anything computed from head outputs z
(rather than from attention patterns or the QK/OV bilinear forms) is only an O(d_head) invariant. (iii) The per-unit
quantities (activation, grad×input, top-k, top direction) are artifacts of unit labels and become exactly invariant
after gauge alignment — the F10 pattern reproduces on the real model. (iv) Weight norms are the one candidate that
alignment cannot repair: ln_2 scalings change c_fc column norms with no compensating relabelling. (v) `singular_values_Wfc`
is covariant (ln_2 scaling), and folding LN into W_fc makes it invariant — the audit finds the correct fold.
**Kill F11 (flips vs MLP toy).** Filled in §3 above: three flips (MLP-hidden persistence scale-covariant → invariant;
singular values covariant → artifact; head-z quantities GPT-2-only), each attributable to the difference between the
ReLU gauge group and GPT-2's LN-compatible subgroup, and flagged rather than smoothed.
