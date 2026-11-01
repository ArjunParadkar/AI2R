# Phase 2 report: K1 and K2 — RUNNING under `prereg-phase-2-amend-1`

**Update 2026-10-24 ~00:45 CDT.**
- Phase 1 closed under amendment 3: the validation passed P1a–P1f, P2 and P3 (`reports/phase1.md` §9).
- `prereg/PHASE_2_amend_1.md` is tagged (commit da2cb07). It covers the amendment-3 estimator, R = 20, model × task units, the K1 null condition and anchor, and 41 cells.
- The cells run on lanes 1367020–1367023 and write to `/scrfs/storage/paradkar/wb/results/k12_a1`.
- Analysis: `python -m bench.analyse_k12 runs/k12_a1 --out reports/phase2_tables.json`, run once every cell is finished. The K2 decision rule is then applied by the letter.

The history below is kept as written.

Status on 2026-10-23, about 11:05 CDT.

`prereg/PHASE_2.md` (tag `prereg-phase-2`, commit 25b772e) runs only if the amendment-2 re-validation shows P1a–P1f, P2 (on V2D) and P3 passed. **P1f failed**: on seed 114, the untrained-copy V4 null accepted 4 of 160 candidates (FP 0.025, threshold 0.02). Details are in `reports/phase1.md` §8 and `docs/record.md`.

Consequences, applied by the letter:
- No K1 or K2 statistic is computed, and K2 has no verdict: it neither passes nor fails. The kill rule of master prompt §6.2 is not triggered, because the test was never run.
- The Phase 2 outputs produced before the verdict are discarded unanalysed:
  - one cell, Qwen2.5-7B add3 seed 0, cancelled after stage 1;
  - the pythia-160m smoke cell, a development model.
- `bench/analyse_k12.py` has not been run on any Phase 2 output.
- Phase 2 needs an estimator that passes a re-validation, which means amendment 3 (a new development cycle, new seeds, tagged before its run). Whether `prereg/PHASE_2.md` can then run unchanged depends on whether that amendment changes the estimator the pre-registration names. If it does, Phase 2 needs its own amendment, tagged before any confirmatory run.

Notes for when Phase 2 runs (found while preparing, written before any data was analysed):
- `bench/analyse_k12.py`'s `k2_decision` pools every non-untrained K2 cell. That includes the extra seeds (Qwen2.5-7B and OLMo-2-7B add3 seeds 1 and 2) and the 8-bit quantization cell. The pre-registration's unit is the model × task cell. The decision must therefore be applied to the seed-0, native-precision model × task cells, and the analysis code needs to match that before it is used. The development pythia-160m cell must be excluded as well.
- The analysis code does not compute the K1 untrained-null condition (≤ 2 % of candidates accepted) or the family-level anchor rule ("at least one family has one passing add3 cell"). Both have to be computed as written in prereg §3.
