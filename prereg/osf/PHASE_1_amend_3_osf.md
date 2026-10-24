# Pre-registration, Phase 1, amendment 3: a family-wise criterion (i) for the quotient test (M10)

WhiteBox E&E. Written 2026-10-23 after amendment 2 failed P1f. DRAFT until the annotated tag `prereg-phase-1-amend-3`; the tag's hash and SHA-256 are recorded in `docs/record.md`. Nothing here runs on the validation seeds before the tag. The tag is made only if `validation/pretag_check.py` exits 0 on the final development run (see the last section).

## Why

Amendment 2 failed P1f. On seed 114, the untrained-copy V4 null accepted 4 of 160 candidates (FP 0.025 > 0.02); on seeds 110–113 it accepted g0, g1 and g2 (FP 0.019). That failure stays in `docs/record.md`.

**Development diagnosis** (dev seeds 0–4; files `runs/phase1/dev_a4diag`, `dev_a4c`, `dev_a4qs`):
- Every node the untrained-V4 null accepts has a pooled swap effect at or below the random-realiser null (e.g. A = 0.027 against η = 0.037).
- These nodes pass criterion (i) of the quotient test, "A_b > η_b for at least one behaviour". V4 has 20 behaviours, and the rule has no multiplicity control, so a realiser that matters no more than random directions clears it through some one behaviour by chance.
- The within-fibre test (δ_Q) does not catch them: on an untrained copy δ_Q is near 0 for any partition of a single input's embedding.
- The first hypothesis was that removing the coarsening rule caused the failure. The data reject it as the fix. Re-enabling the rule removes most of these nodes but not all (seed 1 keeps two), and it fails P4 (scaffold recall 0.67) on every development seed.

## Change — M10 (theory §5): criterion (i) becomes a family-wise max-statistic test

For a candidate realiser at site s with k dimensions, take R norm-matched random realisers at the same site and dimension (R = `n_random_realisers` = 20, unchanged). Let a_{r,b} be their unconditioned swap effects per behaviour b.
- Standardise every behaviour by the null: z_b = (A_b − mean_r a_{r,b}) / max(sd_r a_{r,b}, 10⁻³).
- The statistic is T = max_b z_b.
- Its null distribution is the leave-one-out maxima T_r = max_b (a_{r,b} − mean_{−r}) / max(sd_{−r}, 10⁻³), one value per random realiser.
- **Criterion (i) passes if some z_b exceeds the 95th percentile of {T_r}.** The behaviours that exceed it are the "used" behaviours for δ_Q (as before).
- Every other part of the estimator is that of amendment 2: M1–M9, dual-mode realisers, no coarsening rule, input containers at the primary site.

This is the standard family-wise (Westfall–Young max-T) form of the Phase 1 per-behaviour rule. In the development data:
- the amendment-2 rule accepted 10 null nodes on the untrained V4 copy; max-T rejects 9 of them;
- max-T loses none of the 129 nodes accepted in the trained planted nets (V1, V3, V4, V5);
- with R = 10 random realisers instead of 20, more null nodes survive, so Phase 2 must use the validated R = 20 (`prereg/PHASE_2_amend_1.md`).

Implementation: `whitebox_ee.filter.maxT_use`, with `Thresholds.criterion_i = "maxT"`. The unit test is `tests/test_estimators.py::test_maxT_use_multiplicity`.

## Criteria (frozen at the tag)

The criteria and their bars are exactly those of `prereg/PHASE_1_amend_2.md` for P1a–P1f, P2 on V2D, P3, P4 and P5. V2 is reported as a control and does not gate.

**P6 (formation, FORMD) is not evaluated in this amendment,** so formation stays unvalidated and K14 stays blocked, as it has been since the original P6b failure. This is declared before any amendment-3 FORMD result exists. The reason is compute: no FORMD cell has ever finished. Development and amendment-2 runs at 4 threads went more than 2 h without completing the first of the 7 checkpoints. P6 gates only K14, not Phase 2. Formation gets its own amendment, with a cheaper planted sequence, when K14 is scheduled.
- **Seeds:** 115, 116, 117, 118, 119.
- **Consequences:** as in `prereg/PHASE_1.md` §4. Phase 2 requires P1a–P1f, P2 (V2D) and P3 to pass.

## Pre-tag check (new; the response to the amendment-2 process failure)

Before the tag, `validation/pretag_check.py runs/phase1/dev_a6 0,1,2 --cells V1 V2D V3 V4 V5 NULL --commit <code commits>` must exit 0 on the final development run. Its checks:
- every expected file is present and error-free, and was produced by the tagged code;
- every single null cell has FP ≤ 0.02;
- every criterion this amendment evaluates passes on every development seed on its own, and pooled.

Development null cells for seeds 3 and 4 are checked as well. The check's full output is committed with the tag (`runs/phase1/dev_a5/PRETAG.txt`).

## Development check (filled in before the tag)

TO BE FILLED from `runs/phase1/dev_a5/PRETAG.txt`.
