# Compute ledger

Running record of compute used against approved envelopes. Units:
- GPU: A100-hours. V100-hours count 0.5; laptop RTX 5070 hours are listed separately and not charged.
- CPU: core-hours.

Each row is one batch of work: a job array or a local run.

## Envelopes

| Phase | Approved GPU (A100-h) | Approved CPU (core-h) | Approval |
|---|---|---|---|
| 1 | 200 | 2,000 | Arjun, 2026-10-22 (`reports/phase0.md` §10) |
| 2 | estimate in `prereg/PHASE_2.md` | | Approved by the 2026-10-22 full-speed authorisation |

## Ledger

| Date | Phase | What | Where | Job IDs | GPU-h | CPU core-h | Laptop GPU-h | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-22 | 1 | Env build `wb-ee` (attempt 1, failed: no bitsandbytes 0.45 wheel for glibc 2.17) | Pinnacle cloud72 | 1364346 | 0 | 0.6 | 0 | 4 cores × 9 min |
| 2026-10-23 | 1 | Env build `wb-ee` (attempt 2, bitsandbytes 0.42) | Pinnacle cloud72 | 1364364 | 0 | ≤ 1 | 0 | Status unknown: the ControlMaster dropped at 01:20. Check when Pinnacle is back. |
| 2026-10-23 | 1 | Planted development and validation (V1–V5, nulls, formation) | laptop CPU | — | 0 | ~6 (to date) | 0 | 8 processes × 2 threads |
| 2026-10-23 | 1 | LLM development run, Qwen2.5-1.5B (excluded from preregistered lists) | laptop | — | 0 | — | ~0.5 | Development only |
| 2026-10-23 | 1 | Env build `wb-ee` (attempt 2 failed: sentencepiece/tiktoken wheels) | Pinnacle cloud72 | 1364364 | 0 | 0.1 | 0 | |
| 2026-10-23 | 1 | Env build `wb-ee` (attempt 3, success) | Pinnacle cloud72 | 1364678 | 0 | 0.5 | 0 | torch 2.5.1+cu124, transformers 4.49, bnb 0.42 |
| 2026-10-23 | 1 | Planted dev cells (amendment 2 code, partial; old code) | Pinnacle cloud72 | 1364717 | 0 | ~4 | 0 | informational |
| 2026-10-23 | 1→2 | LLM realiser diagnostics + LLM dev (Qwen2.5-1.5B, development) | Pinnacle agpu06 | 1364727 | ~1.0 | — | 0 | 24 cores/GPU |
| 2026-10-23 | 1 | Planted dev cells, amendment 2 (seeds 0–2) | Pinnacle cloud72 (throttled to 4×4 cores after 10:00) | 1364799 | 0 | ~10 (running) | 0 | |
| 2026-10-23 | 1→2 | LLM dev, dual-mode realisers (Qwen2.5-1.5B) | Pinnacle agpu06 | 1364800 | ~1 (running) | — | 0 | |
| 2026-10-23 | 1 | Amend-2 validation, short cells (seeds 110–114) | Pinnacle comp01, 32 cores, lanes wb-p1-cA/cB | 1364950, 1364951, 1364952, 1364953 | 0 | ~4 per job | 0 | ~6–10 min each |
| 2026-10-23 | 1 | Amend-2 validation, long cells (V2D, FORMD, FORMATION, SENS) | Pinnacle comp06, 32 cores, lanes wb-p1-cA/cB | 1364993, 1364994 | 0 | ≤ 192 each (6 h cap) | 0 | replaces 1364949 (gpu72, would not start before 10-25; cancelled unstarted) |
| 2026-10-23 | 2 | K1/K2 cells, lanes wb-k12-S1..S3 (agpu06, 6 h, self-resubmitting) + wb-k12-L1 (agpu72, 14B/13B/12B/32B) | Pinnacle A100 40 GB, 24 cores each | 1364995–1364998 (+ resubmits) | running | — | 0 | replaces array 1364954 (cancelled unstarted; it would have been 39 submitted GPU jobs) |
| 2026-10-23 | 2 | Phase 2 lanes cancelled at the P1f verdict (outputs discarded unanalysed) | Pinnacle agpu06 | 1364995 (16 min), 1364996–1364998 (never started) | 0.27 | — | 0 | |
