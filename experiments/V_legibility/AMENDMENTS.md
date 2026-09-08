# Block V — amendments

## #1 (2026-09-08, operational — V1 sharded)
V1 ran at ~4 min/model (20 000 steps, Python-bound on tiny models): 130 models ≈ 9 h serial on the shared GPU queue.
`run_V1.py --shard i n` trains `plan()[i::n]`; shard 0 runs inside `gpu_queue.sh`, shards 1–2 as units `wb-v1-s1/2`;
the queue waits for all three and for `wb_ready.py v1` before V2/V3. Rows are appended to `V1_models.csv` by all
three processes (one line per write); a crash between `history.json` and the CSV row is repaired from the history.
The 5 models trained before the restart are kept (the model in flight at the restart is retrained). No change to
the design, seeds, budget or checkpoint schedule.

## #2 (2026-09-08, design — V6 and V8 had no runner)
The 2026-09-07 runners cover V1–V5 only. `run_V6.py` (loss-invisible reorganisation: SV between consecutive
checkpoints, shuffle null per peak, gauge audit of SV on the cloud) and `analyze_V8_Z3.py` (Z3 β1 ΔBIC in the V4
logistic; V8 held-out and within-architecture AUC gates for `legibility_score`) are written 2026-09-08 and run after
V2/V3/V5/Z3 complete. Thresholds unchanged from SPEC.
