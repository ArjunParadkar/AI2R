# Pre-registration, Phase 2, amendment 1: estimator, analysis units, runner

WhiteBox E&E. DRAFT until the annotated tag `prereg-phase-2-amend-1`. Its hash and SHA-256 will be recorded in `docs/record.md`. No Phase 2 output has been analysed: the only outputs that exist are the pythia-160m smoke cell (a development model) and stage 1 of Qwen2.5-7B add3 seed 0 (training accuracy only, from the cancelled job 1364995). Both are discarded (`docs/record.md`), and every Phase 2 cell is recomputed from scratch after this tag.

Everything in `prereg/PHASE_2.md` stands except the four changes below.

## Change 1 — the estimator

Phase 2 uses the estimator validated under `prereg-phase-1-amend-3` (AMEND-3 SUMMARY TO BE FILLED AT ITS TAG), not the amendment-2 estimator. Amendment 2 failed P1f (`reports/phase1.md` §8). Phase 2 runs only if the amendment-3 re-validation (seeds 115–119) shows P1a–P1f, P2 (on V2D) and P3 passed. Discovery settings: TO BE FILLED (any setting the amendment-3 criterion depends on, such as the number of random realisers, must equal the validated value).

## Change 2 — analysis units (bringing `bench/analyse_k12.py` in line with §0 and §4 of the frozen text)

The frozen text defines the cross-model unit as the model × task cell (§0) and runs extra seeds for "resampling stability of the graph" (§2) and one 8-bit cell for the quantization check (§4.4). The frozen code instead pooled every trained cell into the K2 decision. The text governs:
- **K2 decision units:** seed-0 cells at the pre-registered precision (bf16, except Qwen2.5-32B, which is 8-bit and counts only if the §4.4 quantization check passes), trained, and not a development model.
- **Not units, reported only:** extra seeds 1 and 2, the Qwen2.5-7B 8-bit check cell, untrained nulls, and development models.
- **K1 per cell:** the specimen gate must pass, K1a–K1d must meet their thresholds, and the cell's untrained-copy null must accept ≤ 2 % of its candidates (accepted nodes / candidates).
  - The untrained nulls are run for models ≤ 7B (§3).
  - A model > 7B uses its family's largest ≤ 7B null: Qwen2.5-14B and -32B use Qwen2.5-7B; OLMo-2-13B uses OLMo-2-7B; pythia-12b uses pythia-6.9b. This is stated now, before any data. If a cell has no applicable null, it cannot pass K1.
- **K1 anchor:** PASS if at least one family has at least one seed-0 add3 cell passing K1 (§3, "in at least one open LLM family").
- **Quantization check (§4.4)** is computed by the analysis code: node-label Jaccard ≥ 0.8 and Spearman(λ) ≥ 0.8.

Implemented in `bench/analyse_k12.py` (commit at the tag) and tested on synthetic cells in `tests/test_analyse_k12.py`.

## Change 3 — cell list

The frozen text requires an untrained copy for every model ≤ 7B (§3). The frozen cell list had none for pythia-1.4b and pythia-2.8b, so `EleutherAI/pythia-1.4b add3 --untrained` and `EleutherAI/pythia-2.8b add3 --untrained` are added (41 cells).

## Change 4 — runner (no statistic changes)

- **Gate early exit.** `bench/k12.py --gate_exit` stops a cell after stage 1 when the training-accuracy part of the specimen gate fails. §1 says such a cell contributes no K1/K2 statistic, so nothing is lost.
- **Lanes.** Cells run on singleton lanes (`runs/slurm/k12_lane.sbatch`). The lanes resume by stage and resubmit themselves on the 6 h partitions, plus one 72 h lane for models whose discovery does not fit in 6 h.
