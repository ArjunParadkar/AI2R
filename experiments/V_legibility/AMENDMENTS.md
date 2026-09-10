# Block V — amendments

## #1 (2026-09-08, operational — V1 sharded)
V1 ran at ~4 min/model (20 000 steps, Python-bound on tiny models): 130 models ≈ 9 h serial on the shared GPU queue.
`mejiro_dober.py --shard i n` trains `plan()[i::n]`; shard 0 runs inside `gpu_queue.sh`, shards 1–2 as units `wb-v1-s1/2`;
the queue waits for all three and for `wb_ready.py v1` before V2/V3. Rows are appended to `V1_models.csv` by all
three processes (one line per write); a crash between `history.json` and the CSV row is repaired from the history.
The 5 models trained before the restart are kept (the model in flight at the restart is retrained). No change to
the design, seeds, budget or checkpoint schedule.

## #2 (2026-09-08, design — V6 and V8 had no runner)
The 2026-09-07 runners cover V1–V5 only. `nice_nature.py` (loss-invisible reorganisation: SV between consecutive
checkpoints, shuffle null per peak, gauge audit of SV on the cloud) and `analyze_V8_Z3.py` (Z3 β1 ΔBIC in the V4
logistic; V8 held-out and within-architecture AUC gates for `legibility_score`) are written 2026-09-08 and run after
V2/V3/V5/Z3 complete. Thresholds unchanged from SPEC.

**#2 (2026-09-08 18:15, operational — V1 shard deadlock, no scientific change).** After the third power-loss
relaunch the three V1 shards sat at 0 % GPU utilisation for 30 min at 34/130: each shard's headroom gate
(`gpu_free_mb() < 2500`) waited on memory the *other* shards held in their own caching allocators after finishing a
model (1.2–3.8 GB each), and nothing ever released it. Fix: `torch.cuda.empty_cache()` before and inside the wait
and after each successful `train_one`. Shards restarted 18:13 (locks reclaimed by PID check); training resumed at
98 % utilisation. Model results are unaffected (same training code, same seeds; interrupted models retrain from 0).
Addendum 18:55: the fixed 2 500 MB headroom gate also serialised the shards — a shard with a small MLP waited 27 min
while two transformers trained. Headroom is now per architecture (`NEED_MB`: mlp 700, tf2 1500, tf3 2200, tf4 3000,
from observed peaks); OOM retry unchanged. Shards restarted 18:55.
Addendum 22:35: two shards OOM'd 13× each in lock-step on `sort_tf2_*` (both retrying every 120 s with 4.6 GB free
between collisions). Transformers now train one at a time behind a PID-checked GPU lock (`ckpt/.tf_gpu.lock`), with
jittered OOM backoff; a shard that cannot get the lock releases the model and takes the next MLP (multi-pass loop,
`done` re-read from the CSV before every model so nothing is retrained or double-logged). Headroom for tf raised
to 3.5–4 GB. Remaining at 22:35: 73 models (33 transformers ≈ 25–45 min each serialised, 40 MLPs). Same training
code and seeds; interrupted models retrain from 0.

**#3 (2026-09-09 11:36) V2/V3 crashed on its first model: `discover_circuit` passed `metric=` to `node_scores`, which the
E4/V1 node-instrumented models (`legibility.e4models._Base`) do not accept (they score with their own log-prob metric; the
S4 protocol models in `legibility.models` do accept it). Fix in `whiteboxtopo/legibility/discovery.py`: pass `metric` only
where the signature accepts it (the same rule `graphs.attribution_graph` already used for `edge_scores`). Discovery tests
(9) green; V3 smoke on `modadd_mlp64_l10.0_wd0.0_dr0.0_s0`: F = 0.90, size 85, 3-seed Jaccard 0.55. Relaunched as the
standalone unit `wb-v2v3` (the gpu queue had already moved on to M2; V5 later in the queue uses the fixed code).**

**#4 (2026-09-09 12:20) V2/V3 OOM while sharing the GPU** (C1_C3's renamed GPT-2 copies + V6 on the same 8 GB card):
`edge_scores` on a 512-wide MLP needed 586 MiB with 425 MiB free. The run is resumable per model (8 of 130 done);
relaunched as `wb-v2v3` with a headroom gate (wait for ≥ 2.5 GB free, up to 60 attempts, 3-min back-off after a crash).

**#5 (2026-09-09 12:22) V6 crashed on its first transformer model:** `cloud()` indexed `read_pos` (several output
positions for sort/copy) and produced a 3-D array (512, positions·nodes, d); `compute_diagrams` requires (n, features).
Fixed by flattening to (n, positions·nodes·d) before PCA-20 (the MLP branch was already 2-D). Smoke-tested on the first
transformer row; relaunched as `wb-v6` (resumable per model; 3 MLP models were already recorded) with BLAS threads capped at 4.

**#6 (2026-09-09 13:55) Twin Turbo fell at the first fence after the relaunch:** `graphs.rewired_null` let
`NetworkXAlgorithmError` through (a graph with 2–3 edges at density 0.01 cannot be double-edge-swapped); the except clause
caught only `NetworkXError`. Now catches `NetworkXException`; the null for such a graph is the graph itself (Q_excess ≈ 0),
which is the honest answer for a graph with no rewirable structure. Relaunched as `wb-twin-turbo` (headroom-gated).

**#7 (2026-09-09 14:00) GPU serialisation.** After the relaunch three GPU horses shared the 8 GB card: Twin Turbo (standalone,
5.9 GB with the caching allocator), Gold Ship (paddock) and then Agnes Tachyon — Gold Ship and Agnes Tachyon each fell to
CUDA OOM within seconds and the paddock cascaded on to Seiun Sky. Paddock stopped; `gpu_queue.sh` now (a) waits for
`wb-twin-turbo` to finish before Gold Ship, (b) waits for ≥ 4 GB free before every stage, (c) retries each stage once after
5 min. Results so far are intact (all three horses are resumable).

**#8 (2026-09-09 14:15) Twin Turbo OOMs on her own on the largest transformer models** (`edge_scores` allocates ~1 GiB blocks
and the caching allocator already holds ~6 GB after 40 models): per-model fallback — on `torch.OutOfMemoryError` the model
is reloaded and scored on the CPU (a `device` column records which). The paddock no longer has a Twin Turbo stage (she is
the standalone `wb-twin-turbo`; the paddock waits for her before Gold Ship). The V2/V3 CSVs written before this amendment
have no `device` column; DictWriter appends the new column only to new rows — readers must use `csv.DictReader` (they do).

**#9 (2026-09-09 14:25) Nice Nature fell on `numpy.linalg.svd` "SVD did not converge"** inside the PCA-20 step of the SV
film. `dgm1` now returns an empty diagram for a cloud with non-finite values (a diverged checkpoint) and falls back to
LAPACK gesvd when gesdd fails to converge. Relaunched (resumable, 16 models recorded).

**#10 (2026-09-09 15:00) Chunked EAP edge scores for the transformers.** `TinyTF.edge_scores` materialised two
(N, B·T·d) fp32 matrices (tf4 on sort: 1 040 nodes × 512 × 16 × 128 ≈ 4.3 GB each) — the reason sort_tf4 could not fit even
an empty card and fell back to a ~1 h CPU run per model. The score is a sum over batch elements, so it is now computed in
token chunks under a 1.5 GB budget (per-example means sliced with the chunk, each chunk weighted by B_chunk/B) and summed;
`tests/test_legibility.py::test_tinytf_edge_scores_chunked_equals_full` checks equality with the one-shot result. Protocol
unchanged (same batch, same metric); Twin Turbo restarted so the transformers run on the GPU alongside Gold Ship.

**#11 (2026-09-10 08:40) Twin Turbo's CSVs were ragged.** After #8 the rows carried a 26th field (`device`) under the original
25-field header (`DictWriter(fieldnames=list(r))` never rewrote the header), so `pandas.read_csv` refused both files. The files were
rewritten with the full header (backups `results/.V2_structure.ragged.bak`, `.V3_discovery.ragged.bak`); the 40 pre-#8 rows ran on the
GPU and are marked `cuda` — and so are all 91 later rows: after #10 the CPU fallback never fired. `twin_turbo.append_row` now grows
the header (rewrites the file with the union of columns) whenever a row carries a new key; Silence Suzuka uses it too.

**#12 (2026-09-10 08:45) Mejiro Palmer (V4) kill added.** The Q_excess changepoint was the only pre-registered pass (ΔBIC 18.4 > 10)
and the 17-candidate changepoint search inflates ΔBIC, so `analyze_V4_V5.py` now also reports: a 200-permutation label null of the
ΔBIC (observed 18.4; null max 0.96, 95th percentile −3.6, p < 0.005), the same test with the mlp/tf family one-hot in both models
(ΔBIC 19.9 — not a family offset), and per family (mlp n = 73: ΔBIC 8.4, below the bar; tf n = 57: −2.9). Recorded in `V4_fits.json`
under `kill_Qexcess_changepoint`. `analyze_V8_Z3.py`: `g.index.values` is read-only under pandas 2 — copied before the shuffle.
