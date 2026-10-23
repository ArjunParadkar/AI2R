# Record of pre-registered cells

One row per pre-registered cell (master prompt rule 0.1.3). Rows are added only for cells listed in a tagged `prereg/PHASE_X.md`, and each row is filled exactly as pre-registered. Post-hoc analyses are labelled post hoc and never change a verdict.

| Phase | Cell ID | Hypothesis | Statistic | Threshold | Result (95% CI) | Pass/Fail | Deciding null or baseline | Access level | Unit of analysis | Prereg tag | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|

| 1 | P1a V1-onehot | planted-answer estimator test (not a claim) | node precision & recall >= 0.9 on 5/5 | as pre-registered | P 1.00-1.00, R 1.00-1.00 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P1b V1-onehot | planted-answer estimator test (not a claim) | edge F1 >= 0.9 on 5/5 | as pre-registered | 1.00-1.00 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P1c V1-onehot | planted-answer estimator test (not a claim) | decoy represented-not-used 5/5 | as pre-registered | 5/5 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P1e V1-onehot | planted-answer estimator test (not a claim) | generality MAE vs oracle <= 0.05 on 5/5 | as pre-registered | 0.005 (max) | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P1a V1-noise05 | planted-answer estimator test (not a claim) | node precision & recall >= 0.9 on 5/5 | as pre-registered | P 1.00-1.00, R 1.00-1.00 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P1b V1-noise05 | planted-answer estimator test (not a claim) | edge F1 >= 0.9 on 5/5 | as pre-registered | 1.00-1.00 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P1c V1-noise05 | planted-answer estimator test (not a claim) | decoy represented-not-used 5/5 | as pre-registered | 5/5 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P1e V1-noise05 | planted-answer estimator test (not a claim) | generality MAE vs oracle <= 0.05 on 5/5 | as pre-registered | 0.007 (max) | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P1d V1 | planted-answer estimator test (not a claim) | wrong hypotheses accepted <= 10% (pooled) | as pre-registered | 0/50 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P1f NULL | planted-answer estimator test (not a claim) | FP <= 0.02 every seed; mean accepted <= 1 | as pre-registered | max FP 0.000, mean accepted 0.00 (20 cells) | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P2 V2 | planted-answer estimator test (not a claim) | AUROC(-mu) >= 0.9 on 5/5 and lower CI > 0.8 on 4/5 | as pre-registered | pt 1.000-1.000, lo 1.000-1.000 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P3a V3-comp | planted-answer estimator test (not a claim) | move accuracy >= 0.9 (withheld and seen) 5/5 | as pre-registered | 1.00 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P3b V3-comp | planted-answer estimator test (not a claim) | predicted-correctness accuracy >= 0.9 5/5 | as pre-registered | 1.00 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P3a V3-lookup | planted-answer estimator test (not a claim) | move accuracy >= 0.9 (withheld and seen) 5/5 | as pre-registered | 1.00 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P3b V3-lookup | planted-answer estimator test (not a claim) | predicted-correctness accuracy >= 0.9 5/5 | as pre-registered | 1.00 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P4 V4 | planted-answer estimator test (not a claim) | AUROC >= 0.95 and scaffold recall >= 0.9, 5/5 | as pre-registered | AUROC 1.000, recall 1.00 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P5a V5-valid | planted-answer estimator test (not a claim) | exactly one valid node N <- {A,B}, 4/5 | as pre-registered | 5/5 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P5b V5-decoy | planted-answer estimator test (not a claim) | zero valid nodes, 4/5 | as pre-registered | 5/5 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P5b V5-leaf | planted-answer estimator test (not a claim) | zero valid nodes and N 'new leaf (used, not derived)', 4/5 | as pre-registered | 5/5 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P5b V5-relabel | planted-answer estimator test (not a claim) | zero valid nodes, 4/5 | as pre-registered | 5/5 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P5b V5-generic | planted-answer estimator test (not a claim) | zero valid nodes and N 'exposure-generic', 4/5 | as pre-registered | 5/5 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P6a FORM | planted-answer estimator test (not a claim) | grounding event 4+-1 and sum principle 3+-1, 4/5 | as pre-registered | 5/5 | PASS | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |
| 1 | P6b FORM | planted-answer estimator test (not a claim) | \|Lambda_t - (1 - memo_t)\| <= 0.2 every checkpoint, 4/5 | as pre-registered | 0/5 (max deviations 0.58, 0.74, 0.88, 0.50, 0.83) | FAIL | planted truth / oracle realisers / nulls | C (planted) | planted network instance (5 seeds) | prereg-phase-1 | bc6036a |

Phase 1 rows are estimator tests on planted systems (master prompt §5). They are never evidence for a claim. Source: `runs/phase1/val/` (computed by `validation/verdicts_phase1.py`).

## Approvals and tags

| Date | Item | Text / hash |
|---|---|---|
| 2026-10-22 | Phase 0 gate and Phase 1 budget (200 A100-h, 2,000 CPU core-h) approved; programme to run at full speed with self-gates | Arjun, verbatim: "okay cool cool, all things approved, get the BPU stuff done, verified for everything, get as much of omen and inv proj1 work done on HPCC and Full speed whitebox ( like everything we needed) get that all going you dont need any permissions from me." |
| 2026-10-23 | Tag `prereg-phase-1` (annotated, unsigned: no GPG key) | commit bc6036a303a6cf2065d935f5a0c79c54be1385d6; SHA-256 of prereg/PHASE_1.md = 6fd3c0bb80b2125291c892a9fbb781c1d46fc73a69cc9d70165947d38a1e8792; OSF text: prereg/osf/PHASE_1_osf.md (Arjun uploads) |
| 2026-10-23 | Tag `prereg-phase-1-amend-1` (annotated, unsigned) | commit 9a57d93bcb1834c12decf61880bd1c285ccc5061; SHA-256 of prereg/PHASE_1_amend_1.md = 9d7044e7247b986119adf954be4da2a6507df6cfae37c3350dcc77c553d64d3b; OSF text: prereg/osf/PHASE_1_amend_1_osf.md |
