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

**#13 (2026-09-10 14:10) Silence Suzuka (V5) scoped.** After 53 min she had 33 checkpoint rows of 2 480 (two width-512 MLPs):
profiled on one of them, a checkpoint costs 285 s of which 250 s is the 20-rewiring modularity null at three densities (structure
without the null 27 s, discovery 3 s). The SPEC §V5 asks for density 0.1 only, so the V5 branch now computes the null at 0.1 alone
(≈ 2× faster; the 33 existing rows keep their 0.01/0.05 columns, later rows leave them blank). And she runs first on a **pre-declared
balanced subset of 48 models**: the 6 random-label models (the null trajectory) then 7 per (task × architecture family) cell in
sorted-name order, round-robin so any prefix stays balanced (`--limit 48`; `--limit 0` resumes over the remaining 76 models later —
the CSV is resumable per (model, checkpoint)). The paddock was restarted (one in-flight checkpoint lost).

**#14 (2026-09-10 17:45) Silence Suzuka (V5): 5 rewirings in the null instead of 20.** After 3 h she had 149 of ~960 checkpoint rows
(≈ 225 s per checkpoint on the transformers). Timed on `copy_tf3_randlab_s0` at step 10 000 (780 nodes, 20 587 edges at density
0.1): graph 0.7 s, Leiden 1.9 s, treewidth bound 9.8 s, **null 93.7 s with 20 rewirings vs 23.6 s with 5** — `nx.double_edge_swap`
in pure Python is the cost. The null mean moves by 0.0006 (0.2132 → 0.2138; its sd is 0.0015), against a Q_excess spread of 0.06
across the rows so far (median Qnull sd 0.002, max 0.010): the standard error with 5 rewirings (≤ 0.005) is an order of magnitude
below the drift signal. The 149 rows already written keep their 20-rewiring null (`n_null` column: blank = 20, 5 from here on);
V2's final-checkpoint numbers (20 rewirings) are unchanged. Queue relaunched as `wb-gpu-queue` (MemoryMax 20 G) replacing the
other session's `wb-gpu-queue-redo` wrapper; one in-flight checkpoint lost.

**#15 (2026-09-14 23:30) V7 (Mejiro Ardan): `eap_graph` reads one gradient per LAYER, so every consumer node in a layer
gets the same score — instrument defect, found by O3.1's K-O-2 gauge orbit; repair added behind `--per-node`, DEFAULT OFF.**
`hpcc/mejiro_ardan.eap_graph` computes `g = grads[lv]` once per target layer and then writes the same scalar `s` to every
node of that layer (`for nv in [f"h{h}" ...] + ["mlp"]: E[idx[(lu,nu)], idx[(lv,nv)]] = s`). The score matrix is therefore
massively tied: on `pythia-70m` at `step143000` with the 4-sequence probe it carries **91 distinct edge magnitudes among 990
node pairs**. `threshold_density` has to break those ties, and the resulting Leiden partition is discontinuous in the scores,
so a perturbation at the fp32 floor (≈ 1e-6 relative) changes which edges survive. O3.1 measured this as a **K-O-2 failure**:
under exact renamings from the new `whiteboxtopo.symmetry.neox` backend, on the GPU orbit, **Q CV 0.572, treewidth bound
0.192, sparsity 0.061, community sizes not identical** (threshold 1e-3) — and, on a CPU orbit with the same model and
revision, Q CV 4.9e-04. That backend-dependence is the point: at the threshold the statistic is reading arithmetic noise.
The EAP *scores* are invariant analytically (head contributions `z W_O^T` are unchanged by the V/O basis change; q, k, v are
unchanged by the LayerNorm rescales; gradients are taken at the residual stream); the *thresholded graph statistic* is not
stable on a tied matrix.

**Repair** (`eap_graph_per_node`, new, additive): every consumer node reads its own cloned copy of the layer input, so
∂L/∂x_in(v) is head/MLP-specific. A pre-hook keeps the block input `x` in the autograd graph; `query_key_value` is
temporarily wrapped so head h computes its own 3·d_head rows from `input_layernorm(x + eps_h)` with `eps_h` a zero leaf
(the rows are head-major `[q_h|k_h|v_h]`, so concatenating the per-head blocks reproduces the fused projection exactly, at
the same matmul flops); a pre-hook on `post_attention_layernorm` (single consumer: the MLP) replaces its input with
`x + eps_mlp`. `score(u→v) = Σ_t (mean_u − a_u(t)) · ∂lp/∂eps_v(t)` — same formula, same mean corruption, same metric, same
node set, same `lv > lu` edge rule, same densities, Leiden, treewidth bound, sparsity and 20-draw rewiring null. Only the
gradient changes. Effect on the same probe: **91 → 811 distinct edge magnitudes**, Q 0.243 → 0.277, treewidth 2 → 5,
sparsity 0.672 → 0.823, r = 0.15 between the two score matrices, and it is *faster* (8.4 s vs 11.1 s — the per-head split
replaces the full backward).

**Scope of this amendment.** `--per-node` is **OFF by default** and `hpcc/sbatch/v7_pythia_drift.sbatch` is unchanged, so the
V7 rows stay the pre-registered instrument; a per-node run writes to `V7_<tag>_pernode.csv` and every row now carries a
`per_node` column. O3.1 (`experiments/O_opener/hishi_akebono.py`, SPEC_O3 #O3-2) runs **both** at every revision with a K-O-2
orbit each, and keeps its P7 status driven by the pre-registered one. **The already-published V7 numbers are not withdrawn,
but they should be read with this defect stated**: V7's Q/treewidth/sparsity drift over the 154 Pythia checkpoints is
computed on a 91-distinct-value matrix whose threshold tie-breaking is not numerically stable, and the repair does not by
itself make the thresholded Leiden Q gauge-stable at 1e-3 either (per-node Q CV 1.8e-02 on the CPU orbit). A graph statistic
that certifies under G needs a threshold rule continuous in the scores, or a tie-free construction plus a stability margin —
a separate pre-registration, not a change to V7's.

**#15 addendum (2026-09-14 23:55).** The repair was then measured under the orbit and **does not by itself make the statistic gauge-stable**: on CPU, `pythia-70m` step0 + step143000, 3 samples, fp32 floor 2.4e-05 — pre-registered `Q` CV 1.4e-04 (community sizes identical), per-node `Q_pn` CV **2.1e-02** (community sizes NOT identical); on CUDA the pre-registered `Q` CV was **0.572** on the same model and revisions. The K-O-2 verdict therefore **flips with the arithmetic backend**, which is the real finding: the density threshold + Leiden pipeline is discontinuous in the scores, so at the 5 % cut line it reads reduction order, not the network. With 91 distinct magnitudes the cut falls between large tied blocks (reproducible under deterministic tie-breaking); with 811 it falls among near-ties (not reproducible). `treewidth_ub` was CV 0 in every orbit on both backends and both matrices. Consequence for V7's published rows: the Q and community-count drift curves over the 154 Pythia checkpoints should be read as backend-conditional and are not certified under G; the treewidth and sparsity curves are unaffected at this resolution. Fixing this needs a threshold rule continuous in the scores (or a tie-free construction plus a stability margin) — a separate pre-registration, not a change to V7's.
