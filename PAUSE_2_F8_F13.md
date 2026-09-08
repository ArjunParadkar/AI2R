# PAUSE 2 — Block F, the audit (F8–F13) — DRAFT 2026-09-08 (F8–F10 rerunning in `wb-cpu-queue`; F11 final)

_F11 section final; F8–F10 filled when `run_F8_F10.py` + `summarize_F10.py` complete (cpu queue, after F3/F4)._

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
**Kill F11 (flips vs MLP toy).** _to be filled against F10's table when it lands: expected flips are (a) head-z
quantities, which have no toy-MLP counterpart, and (b) nothing else if F10 reproduces the F11 pattern._
