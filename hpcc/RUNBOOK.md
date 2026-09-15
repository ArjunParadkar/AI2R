# HPCC runbook — Arkansas cluster, first session (prepared 2026-09-10 evening)

Everything below is scripted; the only inputs that are not in the repo are in step 0.

## 0. What Arjun must supply (nothing else is blocking)
1. Cluster login: `HPCC_HOST=<user>@<login host>` and a scratch path `HPCC_ROOT=/scratch/<user>/whitebox-v2` (or wherever large files go).
2. Scheduler facts: GPU partition name, account/allocation (if required), GRES string (`gpu:1`, `gpu:a100:1` …). `sinfo -o "%P %G %l"` shows them.
3. Module names for Python ≥ 3.10 and CUDA 12.x (`module avail python cuda`) → the commented `module load` line in `hpcc/env.sh`.
4. A Hugging Face token (`HF_TOKEN`) whose account has accepted the **Gemma** licence (google/gemma-2-2b, gemma-2-9b, gemma-scope) and,
   if F12 is to use Llama, **Meta-Llama-3-8B**. Without it only B5 (GPT-2/Pythia part), M4, V1-large, V7 and the F2 tail can run.

## 1. Sync (laptop → cluster), ~1 GB
```
export HPCC_HOST=... HPCC_ROOT=...
bash hpcc/sync_up.sh            # add --with-ckpt (2.2 GB) only if v5_drift_rest is wanted on the cluster
```
Included on purpose: `benchmarks/results/battery_*` (M4 inputs, 184 MB, git-ignored) and `experiments/F_gauge/results/networks`
(3 648 specimens, 69 MB). Excluded: venvs, the 1.5 GB K1 cache (K8 rebuilds it in the same format), `hpcc/results`.

## 2. Environment (once, on a login or compute node)
```
ssh $HPCC_HOST; cd $HPCC_ROOT
# edit hpcc/env.sh: uncomment/replace the `module load` line
export HF_TOKEN=...; source hpcc/env.sh          # builds .venv-hpcc (torch cu128 + whiteboxtopo[dev] + transformer_lens + sae_lens + spacy)
python -c "import whiteboxtopo, transformer_lens, sae_lens, torch; print(torch.cuda.is_available())"
python -m pytest tests -q -x                     # 228 tests, ~45 min on a loaded CPU (2026-09-14: 228 passed) — the library must be green on the cluster's Python
bash hpcc/configure.sh <partition> [account] [gres]   # fills the placeholders in hpcc/sbatch/*.sbatch
```
If the cluster's CUDA is not 12.8, change the `--index-url` in env.sh to the matching wheel index before the first `source`.

## 3. Smoke (interactive GPU node, ~1–2 h total, mostly model downloads)
```
srun --partition=<gpu partition> --gres=<gres> --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
source hpcc/env.sh; bash hpcc/smoke.sh          # PASS/FAIL per job in hpcc/logs/smoke.log
```
Every FAIL is fixed before its sbatch is submitted (the scripts were syntax-checked and, where no gated model is needed, smoke-run on
the laptop on 2026-09-10 — see §7 — but never on a ≥ 40 GB GPU).

## 4. Submit (from the checkout root, always; the jobs `cd` to `$SLURM_SUBMIT_DIR`)
Order and rough cost (A100-class; all resumable with `--resume` / per-unit CSVs, so a time-out is re-submitted, not restarted):
| sbatch | Handbook | Script | GPUs | Est. | Notes |
|---|---|---|---|---|---|
| `f2_d3w10_enumeration` | F2 tail (depth-3 width-10 cells) | `experiments/F_gauge/rice_shower.py --shard i 20` | array 0-19 | ~2–3 h wall typical, up to ~24 h for a shard holding THR3 cells (Méloux 60 s cap × thousands of minimal circuits; laptop saw one THR3 cell take >5 h) — limit raised to 24 h 2026-09-12 | replaces the 40 h laptop run; `sync_down.sh` merges the shards into `experiments/F_gauge/results/` and skips specimens the laptop already finished |
| `f12_frontier_audit` | F12 | `kitasan_black.py` (Gemma-2-9B) | 1 × ≥ 40 GB | 6–12 h | identity check first (`identity.json` in the out dir) |
| `k8_frontier_residue` | K8 | `biwa_hayahide.py` (Gemma-2-9B + Gemma Scope 9b) | 1 × ≥ 40 GB | 4–8 h | writes the K1 cache format; then rerun `experiments/K_darkmatter/smart_falcon.py` on it |
| `k7_width_scaling` | K7 | `mayano_top_gun.py` (Gemma-2-2B, 16k→1M widths) | 1 | 3–6 h | labelled a replication of Engels et al. |
| `c4_scale` | C4 | `sirius_symboli.py` (Gemma-2-9B) | 1 × ≥ 40 GB | 4–8 h | |
| `b5_family_calibration` | B5 | `mejiro_ryan.py` (GPT-2, Pythia-410M, Gemma-2-2B) | 1 | 3–6 h | |
| `m4_cross_family` | M4 | `admire_vega.py --train-vit` | 1 | 2–4 h | trains the ViT-S/8 battery (18 models) |
| `v1_ensemble_large` | V1 large arms | `mejiro_dober_large.py` | array 0-7 | ~12 h | no smoke flag — check shard 0's log after 10 min |
| `v7_pythia_drift` | V7 | `mejiro_ardan.py` | array 0-1 | ~3 h each | 154 revisions × 2 models. **Known instrument defect** (V AMENDMENTS #15, found by O3.1's gauge orbit): `eap_graph` reads one gradient per *layer*, so the score matrix is massively tied (91 distinct edge magnitudes among 990 node pairs) and the density threshold's tie-breaking is not numerically stable — Q moved by CV 0.57 under exact renamings. The sbatch is **unchanged** (the V7 rows are pre-registered with that instrument); `--per-node` runs the repair into `V7_<tag>_pernode.csv` if a repaired trace is wanted |
| `l7_end_to_end` | L7 | `kawakami_princess.py` (Gemma-2-9B) | 1 × ≥ 40 GB | 2–4 h | writes `L7_timing.json` + `does_not_scale` list |
| `o31_pythia_open` | O3.1 Hishi Akebono (T3 Pythia P7/P3, `experiments/O_opener/SPEC_O3.md`) | `experiments/O_opener/hishi_akebono.py` | array 0-1 | 410m ~5 h, 1b ~7.5 h at a 100 MB/s HF link (~6.5 h / ~11 h at 30 MB/s); `--time` 12 h | 154 revisions × 2 models; **download-bound**, not GPU-bound: 246 GB (410m) / ~616 GB (1b) of checkpoints, so the job passes `--purge` and keeps ≤ 4 GB on scratch at a time — do **not** run it concurrently with `v7_pythia_drift` on the same `HF_HOME`. Reuses `mejiro_ardan.eap_graph` (V7) and `mejiro_ryan.pooled_resid` + `winning_ticket` (B2/B3). Resumable per revision. Afterwards run `hishi_akebono.py --aggregate --out hpcc/results/o31` once (no GPU). Includes **K-O-2** (SPEC_O3 AMENDMENT #O3-1, `whiteboxtopo.symmetry.neox`): `--n-gauge 20` gauge samples at every 10th revision plus the final, ~1.0 h (410m) / ~1.4 h (1b) of the estimate above, plus ≤ 0.5 h for the orbit zigzag in finalise. `--n-gauge 0` skips it and the cells stay `partial`. Since SPEC_O3 #O3-2 it builds **two** EAP graphs per revision — `mejiro_ardan.eap_graph` (pre-registered, one gradient per layer) and `eap_graph_per_node` (the repair, one gradient per target node), each with its own K-O-2 orbit; that is the ~+1 h in the estimate above. The P7 status stays on the pre-registered instrument |
| `o41_frontier_p0` | O4.1 Bamboo Memory step 2 — T4 frontier P0 (`experiments/O_opener/SPEC_O4.md`) | `experiments/O_opener/bamboo_memory.py --p0` | array 0-1, 1 × ≥ 40 GB | ~10–60 min each, almost all of it the first-run 16–19 GB Hugging Face download (laptop pre-flight §7: GPT-2 small, 65 nodes, 2 016 tokens = 2.1 s EAP + 3.1 s graph; the frontier arithmetic is ~65× the FLOPs at the same probe, i.e. still minutes on an A100); `--time` 2 h | EAP attribution graph over the attention heads + MLP of layers mid-2..mid+2 (Llama-3-8B 14–18, Gemma-2-9B 19–23) on the fixed 2 000-token OpenWebText-10k probe; Leiden Q against the degree-preserving rewiring null, min-fill treewidth bound and sparsity at densities 0.01 / 0.05 / 0.1. Verdict by the treewidth bound at density 0.1 (≤ 8 tractable, ≤ 32 marginal, else intractable — AMENDMENTS #5d of O1.2). Resumable per batch (`p0_<slug>_E.npz`). Since AMENDMENTS #O4.1 (d) it also builds the **per-target-node** EAP graph (`p0_<slug>_E_pernode.npz`) and, in float32 only, a 3-sample K-O-2 orbit on both (`--gauge-seq 4`); `--no-per-node` / `--n-gauge 0` skip them. **The P0 verdict stays on the pre-registered graph**, but the two instruments can disagree — on GPT-2 the pre-registered graph gives treewidth 8 (**tractable**, exactly on the boundary) and the per-node graph 11 (**marginal**), and the per-layer gradient can only lower the thresholded treewidth, so read `per_node.verdict` alongside `verdict`. The orbit needs **`--dtype float32`** (`symmetry.*.apply` inverts with `torch.linalg.inv`, which refuses bf16; at bf16 the graphs and the verdict are still produced with `gauge_orbit.ran = false` and the reason). GPT-2 float32 pre-flight: both graphs certified (Q CV 1.7e-06 / 7.2e-07, treewidth CV 0, fp32 floor 1.5e-06). Needs **both** the Llama and the Gemma licences on `HF_TOKEN`; without the Llama licence submit `--array=1` only |
| `o41_assemble` | O4.1 Bamboo Memory step 9 — the two T4 scorecards | `experiments/O_opener/bamboo_memory.py --assemble` | CPU, no GRES | minutes (three 20 000 × d eigendecompositions per K8 layer, for γ_cov and its two MP nulls) | **Submit with a dependency — the job ids are only known at submission, so the line is written on the command line, not in the sbatch:** `f12=$(sbatch --parsable hpcc/sbatch/f12_frontier_audit.sbatch); c4=$(sbatch --parsable hpcc/sbatch/c4_scale.sbatch); k8=$(sbatch --parsable hpcc/sbatch/k8_frontier_residue.sbatch); p0=$(sbatch --parsable hpcc/sbatch/o41_frontier_p0.sbatch); sbatch --dependency=afterany:$f12:$c4:$k8:$p0 hpcc/sbatch/o41_assemble.sbatch` — add `:$l7` (from `l7_end_to_end`) to wait for L7 as well. `afterany`, never `afterok`: a scorecard whose inputs failed or timed out is still a deliverable. Reads `hpcc/results/{f12_*,c4*,k8*,l7*,o41}`, writes `hpcc/results/o41/scorecard_<model>.{json,md}`; every missing or model-mismatched input becomes a `not_reached` cell carrying the reason and the command that would produce it, never an exception. Re-runnable any time more inputs land |
| `o2_gpt2_open` | O2.1 Satono Crown + O2.2 Narita Top Road | `experiments/O_opener/satono_crown.py` (GPT-2 small: IOI, induction, greater-than) | 1 (any, ≥ 8 GB; peak 5.0 GB) | **~1.1 h measured on an RTX 5070 Laptop**, ~35–50 min expected on an A100 (the job is kernel-launch- and CPU-bound at GPT-2-small batch sizes, so a bigger card buys little); wall set to 4 h | pre-registration `experiments/O_opener/SPEC_O2.md`; **resumable per task AND per cell** — re-submit the same file after a time-out; no gated model; the `--audit` carry of the F11 table runs at the end in seconds. Copy `experiments/O_opener/results/open/satono_crown_*.json` + `O2_cells.csv` back |
| `v5_drift_rest` | V5, remaining 76 models | `twin_turbo.py --all-ckpts --limit 0` | 1 | ~6 h | optional; needs `--with-ckpt` sync; writes into `experiments/V_legibility/results/V5_drift.csv` on the cluster — copy that file back by hand |
```
for j in f2_d3w10_enumeration f12_frontier_audit k8_frontier_residue k7_width_scaling c4_scale b5_family_calibration m4_cross_family v1_ensemble_large v7_pythia_drift l7_end_to_end o31_pythia_open o41_frontier_p0; do sbatch hpcc/sbatch/$j.sbatch; done
# o41_assemble is NOT in that loop: it is submitted afterwards with --dependency=afterany:<f12>:<c4>:<k8>:<o41p0> (see its row above).
squeue -u $USER
```
Submit everything at once if the allocation allows; the jobs are independent (K8 → then `smart_falcon.py` by hand).

## 5. Bring results back
```
bash hpcc/sync_down.sh      # hpcc/results/ + logs, and merges the F2 d3w10 shards into the local F results
```
Then locally: `experiments/F_gauge/aggregate_F.py` (F3/F4 aggregates now include the d3w10 cells), PAUSE_1 §F2 numbers, PAUSE_2 F12
section, PAUSE_3 K7/K8, PAUSE_4 M4/C4/B5/V7, PAUSE_5 L7 — all the report sections already have "pending HPCC" placeholders.

## 6. Failure modes to expect
* `HF_TOKEN` unset → env.sh refuses to source (by design). Gated-model 403 → the licence was not accepted on that account.
* CUDA wheel mismatch → `torch.cuda.is_available()` False in step 2; fix the index URL, delete `.venv-hpcc`, re-source.
* OOM on the 9B jobs → they are written for ≥ 40 GB in bf16 with a renamed copy; on a 40 GB card pass `--cpu-copy` (F12) or lower `--batch`.
* Time-out → re-submit the same sbatch; every job resumes from its CSV/JSON checkpoints.

## 7. Laptop pre-flight (2026-09-10)
Laptop smoke runs of the token-free scripts (2026-09-10 evening, `hpcc/logs/smoke_local*.log`): **M4 PASS, B5 PASS** (after fixing a missing PCA frame and a p-value indexing bug), **V7 PASS** on the CPU (after fixing a forward pre-hook that returned a tuple; `--device` added — the laptop GPU was shared with the F2 tail), **F2 shard PASS** (one depth-3 width-10 specimen through `--shard/--out`). V1-large and the five Gemma jobs (F12, K7, K8, C4, L7) can only be smoked on the cluster.

**O4.1 Bamboo Memory laptop pre-flight (2026-09-14, RTX 5070 Laptop 8 GB, fp32, `OMP_NUM_THREADS=4 nice -n 15`).** `--p0` end-to-end on the public GPT-2: the single-layer smoke `--model gpt2 --n-tokens 250 --layers 6` ran in **18.4 s wall** (2.1 s of it compute: 0.4 s EAP + 1.4 s graph; 13 nodes, treewidth 1 — a single layer has only within-layer head→MLP edges, so the graph is a star and the `tractable` verdict there is degenerate), and the representative mid-window run `--model gpt2 --n-tokens 2000` (layers 4–8, 65 nodes, 2 016 tokens) in **13.2 s wall / 2.1 s EAP + 3.1 s graph**, giving Q 0.181 against a rewiring null of 0.221 ± 0.010, treewidth bound 8, sparsity 0.646 → `tractable` (at the threshold). The Llama and Gemma-2 code paths were exercised on randomly-initialised tiny configs (no weights, no token): per-node contributions sum to the block's residual delta to 3.9e-7 (Llama) and 2.8e-7 (Gemma-2, with the post-sublayer RMSNorm rescale). `--assemble` against an empty results tree produced a scorecard with all eight cells `not_reached` and a per-cell reason in **5.9 s**. Frontier extrapolation: the compute is ~65× the FLOPs of the GPT-2 run at the same probe, i.e. **minutes on an A100**; the 2 h wall request is the model download and load. Artifacts: `hpcc/results/smoke_local/o41/`.

O3.1 Hishi Akebono (2026-09-14) was smoke-run on the laptop GPU: **O3.1 PASS** end-to-end on `EleutherAI/pythia-70m` over `step0 / step1000 / step143000` with a tiny probe — 76 s wall for the three revisions including downloads, writing `scorecard_pythia-70m.json`; a second full-settings timing pass gave 11.1 s/revision (70m) and 18.6–27.4 s/revision (pythia-410m) of compute with the weights already on disk. The `--aggregate`, `--finalise` and `--purge` paths were exercised separately. At three revisions every change point is undefined by the frozen rule (T < 6) and the scorecard says so — that is the expected smoke output, not a failure. **K-O-2 added the same evening** (SPEC_O3 AMENDMENT #O3-1, after `whiteboxtopo.symmetry.neox` landed) and re-smoked at `--n-gauge 3`: 225.6 s in the runner (262 s total) for the same 3 revisions against 37.9 s without it, i.e. ~5 s per gauge sample on pythia-70m; measured fp32 floor 5.9e-06 / 1.6e-05 relative. Verdict on the smoke: **P3 `certified`** (every P3 scalar at the floor, including B2's MLP-hidden and per-head coordinate audits) and **P7 `failed (artifact)`** — the EAP legibility instrument moves under the gauge at step143000 (Q CV 0.572, treewidth 0.192, sparsity 0.061, community sizes not identical). That is a real finding about V7's own graph statistic, not a runner bug: the score matrix has 91 distinct edge magnitudes among 990 pairs, so the 5 % density threshold breaks massive ties and any relabelling reshuffles the surviving edges (see SPEC_O3 AMENDMENT #O3-1). It is reported, not repaired — the threshold is frozen. (On **GPT-2 in float32**, by contrast, both graphs certify to six digits — see the `o41_frontier_p0` row — so the instability is specific to the heavier tie structure and the numerically larger gauge group on Pythia, not to the measure itself.) **Re-smoked 2026-09-14 on CPU with the #O3-2 repair in place** (both EAP graphs, each with its own orbit; 879 s, but the box was at load 25 so that is not a clean timing): pre-registered `Q` CV **1.4e-04** and per-node `Q_pn` CV **2.1e-02**, i.e. **P7 `certified` on CPU and `failed` on CUDA for the same model and revisions**. The K-O-2 verdict for the legibility instrument flips with the arithmetic backend; every other P7 instrument and every P3 quantity sit at the fp32 floor on both. `k_o_2.device`, `k_o_2.torch_version` and `k_o_2.backend_dependence` are recorded in every scorecard — read the P7 legibility verdict as backend-conditional, and note that `treewidth_ub` (what O4.1's P0 verdict uses) was CV 0 in every orbit.
