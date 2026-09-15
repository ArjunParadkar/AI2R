# HPCC MANIFEST — everything that runs on the cluster (WHITEBOX Final Specification, 2026-09-14) — prepared 2026-09-14 night

_One file. Every job below is scripted, resumable, and either smoke-tested on the laptop (token-free jobs) or syntax-checked and
written for a ≥ 40 GB GPU (gated Gemma/Llama jobs). Step 0 of `hpcc/RUNBOOK.md` lists the four things only Arjun can supply
(login/scratch, partition/account/GRES, module names, HF token with Gemma + Llama licences). Submission order and the exact
commands are in `hpcc/launch_all.sh`; results come back with `hpcc/sync_down.sh`._

## A. Jobs (sbatch in `hpcc/sbatch/`, scripts in `hpcc/` or `experiments/`)

| # | sbatch | Spec | What it produces | Script | GPU / mem / time | Needs token | Status |
|---|---|---|---|---|---|---|---|
| 1 | `f2_d3w10_enumeration` (array 0-19) | F2 tail (Handbook F2; FINAL_SPEC §8.3) | the remaining depth-3 width-10 gauge-enumeration cells (laptop has 214 of 449 done; `sync_down.sh` merges and skips finished specimens) | `experiments/F_gauge/rice_shower.py --shard i 20 --gpu --valid-budget 500M` | 1 GPU / 64 GB / ≤ 24 h per shard | no | smoked on laptop (1 shard cell) |
| 2 | `f12_frontier_audit` | F12 → O4.1 step 1+4 (gauge identity, audit table) | `hpcc/results/f12_*/identity.json`, audit table | `hpcc/kitasan_black.py` (Gemma-2-9B; Llama-3-8B with `--model`) | 1 × ≥ 40 GB / 96 GB / 12 h | **yes** | written, not smoked (gated) |
| 2b | `f12_llama3_8b` | F12 on Llama-3-8B (T4 gauge + audit, second model) | identity + audit table | `hpcc/kitasan_black.py --model meta-llama/Meta-Llama-3-8B` | 1 × ≥ 40 GB / 96 GB / 12 h | **yes (Llama licence)** | added 2026-09-14 night |
| 3 | `k8_frontier_residue` | K8 → O4.1 step 7 (γ_cov with Gemma Scope) | K1-format cache + K2–K5 rows | `hpcc/biwa_hayahide.py` then `experiments/K_darkmatter/smart_falcon.py` | 1 × ≥ 40 GB / 96 GB / 8 h | **yes** | written, not smoked; a module-shadowing bug (would have lost the K2–K5 CSVs after the cache write) fixed 2026-09-14 night |
| 4 | `k7_width_scaling` | K7 (replication anchor) | Gemma-2-2B width scaling rows | `hpcc/mayano_top_gun.py` | 1 GPU / 64 GB / 6 h | **yes** | written |
| 5 | `c4_scale` | C4 → O4.1 step 3 (certified SAE/dictionary inventory) | certified-feature counts per renaming | `hpcc/sirius_symboli.py` | 1 × ≥ 40 GB / 96 GB / 8 h | **yes** | written |
| 5b | `c4_llama3_8b` | C4 on Llama-3-8B (T4 P1, second model) | certified-feature counts | `hpcc/sirius_symboli.py --model meta-llama/Meta-Llama-3-8B --layer 16` | 1 × ≥ 40 GB / 96 GB / 8 h | **yes (Llama licence)** | added 2026-09-14 night |
| 6 | `b5_family_calibration` | B5 (→ O3.1 P3 n-requirement) | B3 power curves per family | `hpcc/mejiro_ryan.py` | 1 GPU / 32 GB / 6 h | Gemma part only | smoked on laptop (GPT-2/Pythia) |
| 7 | `m4_cross_family` | M4 | ViT battery + memorization_signal per family | `hpcc/admire_vega.py --train-vit` | 1 GPU / 32 GB / 4 h | no | smoked on laptop |
| 8 | `v1_ensemble_large` (array 0-7) | V1 large arms (phase diagram) | V1 rows for the large models | `hpcc/mejiro_dober_large.py` | 1 GPU / 32 GB / 12 h | no | written (check shard 0 after 10 min) |
| 9 | `v7_pythia_drift` (array 0-1) | V7 → O3.1 P7 legibility drift | Q(t) over 154 revisions × 2 models | `hpcc/mejiro_ardan.py` | 1 GPU / 32 GB / 8 h | no | smoked on laptop (CPU) |
| 10 | `l7_end_to_end` | L7 (release gate) → O4.1 timing | `L7_timing.json`, does-not-scale list | `hpcc/kawakami_princess.py` | 1 × ≥ 40 GB / 96 GB / 4 h | **yes** | written |
| 11 | `v5_drift_rest` | V5 remaining 76 models (optional) | V5 drift rows | `experiments/V_legibility/twin_turbo.py --all-ckpts --limit 0` | 1 GPU / 32 GB / 6 h | no | optional; needs `sync_up.sh --with-ckpt` |
| 12 | `o31_pythia_open` (array 0-1) | **O3.1 Hishi Akebono** (T3: P7 zigzag/eigenflow/normalised-persistence velocity/drift; P3 bias probes B2/B3) | `hpcc/results/o31/scorecard_<model>.json` + trajectories | `experiments/O_opener/hishi_akebono.py` | 1 GPU / 64 GB / 12 h (download-bound: 246 GB + 616 GB of revisions; `--purge` keeps ≤ 4 GB — do not run concurrently with #9 on the same HF_HOME) | no | **READY**: SPEC_O3.md; smoke on pythia-70m PASS (38 s); cells capped at `partial` until the GPT-NeoX gauge backend lands (being written) |
| 13 | `o41_frontier_p0` (array 0-1) + `o41_assemble` | **O4.1 Bamboo Memory** (T4: P0 on Llama-3-8B / Gemma-2-9B; assembles F12 + C4 + K8 + L7 into the scorecard with 1 − γ_cov) | `hpcc/results/o41/scorecard_<model>.json` | `experiments/O_opener/bamboo_memory.py --p0` / `--assemble` | 1 × ≥ 40 GB / 64 GB / 2 h (P0, minutes of compute + download); CPU 48 GB / 1 h (assemble, `--dependency=afterany` on #2 #3 #5 #10 #13 and the Llama variants #2b #5b) | **yes** | **READY**: SPEC_O4.md; smoke gpt2 P0 PASS (20 s), empty assemble PASS (all cells not_reached with reasons). Reachable T4 cells: P0 both; P1/P4(partial)/γ_cov Gemma (+ Llama with #2b/#5b for gauge+P1; no public Llama SAE → Llama degree unknown) |
| 14 | `o2_gpt2_open` | **O2.1 Satono Crown / O2.2 Narita Top Road** (T2 GPT-2 small: IOI/induction/greater-than certified circuits, H0/H1 boundaries, R3/R4/R5/R6; F11 audit into the grid) | `experiments/O_opener/results/open/satono_crown_*.json` | `experiments/O_opener/satono_crown.py` | 1 GPU / 32 GB / (see RUNBOOK row) | no | **being written tonight**; runs on the 5070 if it fits before morning, else here |

## A2. Data shipped by `sync_up.sh` but not committed (git-ignored)

* `hpcc/data/openwebtext-10k.parquet` (30 MB; the K1/F11 probe corpus — the two loaders fall back to it via `WB_OWT_PARQUET`, set in `hpcc/env.sh`).
* `benchmarks/results/battery_*` (M4 inputs, 184 MB) and `experiments/F_gauge/results/networks` (16 MB) — already in the sync set.

## B. Not on the cluster (laptop, done or running)

* Block O T1 (O1.1–O1.4, O5 T1 rows): **done 2026-09-14** — `PAUSE_A_O1.md`, `PAUSE_B_O1.md`, `experiments/O_opener/results/grid.md`.
* F2 d3w10 tail: running locally in `wb-gpu-queue` (235 specimens left at 22:00); the cluster array (#1) is the same job sharded; whichever
  finishes first wins — `sync_down.sh` merges without double-counting.

## C. After the cluster runs (laptop, scripted)

`bash hpcc/sync_down.sh` → `experiments/O_opener/symboli_kris_s.py` picks up `hpcc/results/o31/scorecard_*.json`, `o41/scorecard_*.json` and
`satono_crown_gpt2.json` (T2/T3/T4 rows of the grid replace the "pending" rows) → PAUSE_D release review → CLAIMS.md rows for T2–T4.

## D. Pile — big tasks that will not finish on the laptop by the morning of 2026-09-15

See `hpcc/PILE/README.md`.
