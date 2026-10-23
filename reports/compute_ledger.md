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
