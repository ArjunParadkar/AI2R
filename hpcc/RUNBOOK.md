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
python -m pytest tests -q -x                     # 143 tests, ~2 min on CPU — the library must be green on the cluster's Python
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
| `f2_d3w10_enumeration` | F2 tail (depth-3 width-10 cells) | `experiments/F_gauge/rice_shower.py --shard i 20` | array 0-19 | ~2–3 h wall | replaces the 40 h laptop run; `sync_down.sh` merges the shards into `experiments/F_gauge/results/` and skips specimens the laptop already finished |
| `f12_frontier_audit` | F12 | `kitasan_black.py` (Gemma-2-9B) | 1 × ≥ 40 GB | 6–12 h | identity check first (`identity.json` in the out dir) |
| `k8_frontier_residue` | K8 | `biwa_hayahide.py` (Gemma-2-9B + Gemma Scope 9b) | 1 × ≥ 40 GB | 4–8 h | writes the K1 cache format; then rerun `experiments/K_darkmatter/smart_falcon.py` on it |
| `k7_width_scaling` | K7 | `mayano_top_gun.py` (Gemma-2-2B, 16k→1M widths) | 1 | 3–6 h | labelled a replication of Engels et al. |
| `c4_scale` | C4 | `sirius_symboli.py` (Gemma-2-9B) | 1 × ≥ 40 GB | 4–8 h | |
| `b5_family_calibration` | B5 | `mejiro_ryan.py` (GPT-2, Pythia-410M, Gemma-2-2B) | 1 | 3–6 h | |
| `m4_cross_family` | M4 | `admire_vega.py --train-vit` | 1 | 2–4 h | trains the ViT-S/8 battery (18 models) |
| `v1_ensemble_large` | V1 large arms | `mejiro_dober_large.py` | array 0-7 | ~12 h | no smoke flag — check shard 0's log after 10 min |
| `v7_pythia_drift` | V7 | `mejiro_ardan.py` | array 0-1 | ~3 h each | 154 revisions × 2 models |
| `l7_end_to_end` | L7 | `kawakami_princess.py` (Gemma-2-9B) | 1 × ≥ 40 GB | 2–4 h | writes `L7_timing.json` + `does_not_scale` list |
| `v5_drift_rest` | V5, remaining 76 models | `twin_turbo.py --all-ckpts --limit 0` | 1 | ~6 h | optional; needs `--with-ckpt` sync; writes into `experiments/V_legibility/results/V5_drift.csv` on the cluster — copy that file back by hand |
```
for j in f2_d3w10_enumeration f12_frontier_audit k8_frontier_residue k7_width_scaling c4_scale b5_family_calibration m4_cross_family v1_ensemble_large v7_pythia_drift l7_end_to_end; do sbatch hpcc/sbatch/$j.sbatch; done
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
Laptop smoke runs of the token-free scripts are in `hpcc/logs/smoke_local_*.log`; results in `hpcc/results/smoke_local/`.
