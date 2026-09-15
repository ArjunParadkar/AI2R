# PILE — big tasks not finishable on the laptop by 2026-09-15 morning (each one is HPCC-ready; nothing here is lost work)

| task | state on the laptop | what is in the repo for the cluster | how to reconcile |
|---|---|---|---|
| **F2 d3w10 enumeration tail** (449 cells; THR3 cells take 5–10 h each on 6 CPU workers) | 214/449 done at 22:00; `wb-gpu-queue` keeps running (resumable, checkpointed per specimen in `experiments/F_gauge/results/enumeration_status.csv`) | `hpcc/sbatch/f2_d3w10_enumeration.sbatch` (20 shards, `--valid-budget 500M`, 24 h each) | `hpcc/sync_down.sh` merges shard CSVs and skips specimens the laptop finished; then `experiments/F_gauge/aggregate_F.py` |
| **O2 GPT-2 open()** (Satono Crown) | runner being written 2026-09-14 night; full run = 3 tasks × (discovery + ≥ 20 gauge samples + glue + R3–R6): estimate from the smoke run | `hpcc/sbatch/o2_gpt2_open.sbatch` | if launched locally as `wb-satono-crown` and unfinished, the same command resumes on the cluster (per-task JSON checkpoints) |
| **O3.1 Pythia** (154 revisions × 2 models) | not runnable locally in time (≥ 1 min/revision/instrument on an A100; laptop GPU shared) | `hpcc/sbatch/o31_pythia_open.sbatch` | results only from the cluster |
| **O4.1 frontier** (Llama-3-8B, Gemma-2-9B) | impossible locally (8 GB GPU, no HF token) | `hpcc/sbatch/{f12_frontier_audit,c4_scale,k8_frontier_residue,l7_end_to_end,o41_frontier_p0,o41_assemble}.sbatch` | results only from the cluster |
| **V1 large arms, V5 rest, K7, M4, B5 Gemma part** | not started locally (Handbook blocks, all "pending HPCC" in PAUSE_1–5) | their sbatch files (#4–#11 in MANIFEST) | PAUSE_1–5 placeholders |

Nothing else is outstanding on the laptop: T1 (Block O) is complete and reported.
