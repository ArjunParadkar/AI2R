#!/usr/bin/env bash
# Run one WHITEBOX job on Colab with results mirrored to Google Drive every 10 minutes (so a session timeout loses nothing).
#   bash colab/run_job.sh <JOB> [DRIVE_DIR]      DRIVE_DIR default /content/drive/MyDrive/whitebox-results
# JOB names (same scripts / --out dirs as hpcc/sbatch, so hpcc/results/ stays one namespace):
#   o31-410m o31-1b        O3.1 Hishi Akebono (Pythia P7/P3; no token needed)
#   o41-p0-gemma o41-p0-llama o41-assemble-gemma o41-assemble-llama   O4.1 Bamboo Memory (token + licences)
#   o2                     O2 GPT-2 open() (no token; already done on the laptop — re-run only for larger n)
#   f2-s<i>                F2 d3w10 enumeration shard i of F2_SHARDS (default 20; CPU-bound: open several sessions)
#   v7-410m v7-1b          V7 legibility drift (no token)
#   v1-s<i>                V1 large arms shard i of 8 (no token)
#   m4 b5 k7               M4 (needs the battery tarball on Drive), B5, K7 (Gemma-2-2B: licence)
#   f12-gemma f12-llama c4-gemma c4-llama k8-gemma l7-gemma            frontier jobs (A100 40 GB; token + licences)
set -e
JOB="${1:?job name}"; DRIVE="${2:-/content/drive/MyDrive/whitebox-results}"; F2_SHARDS="${F2_SHARDS:-20}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; mkdir -p "$DRIVE" hpcc/results hpcc/logs
[ -f /content/.owt_parquet ] && export WB_OWT_PARQUET="$(cat /content/.owt_parquet)"
# restore anything a previous session already computed for this job (resumable scripts pick it up)
[ -d "$DRIVE/hpcc/results" ] && rsync -a "$DRIVE/hpcc/results/" hpcc/results/ 2>/dev/null || true
[ -d "$DRIVE/O_results" ] && rsync -a "$DRIVE/O_results/" experiments/O_opener/results/ 2>/dev/null || true
sync_out() { rsync -a --exclude 'cache' hpcc/results/ "$DRIVE/hpcc/results/" 2>/dev/null; rsync -a --exclude 'ckpt' --exclude '*_clouds.npz' experiments/O_opener/results/ "$DRIVE/O_results/" 2>/dev/null; cp -f hpcc/logs/*.log "$DRIVE/logs/" 2>/dev/null || true; }
mkdir -p "$DRIVE/logs"; ( while true; do sleep 600; sync_out; done ) & SYNC=$!
trap 'kill $SYNC 2>/dev/null; sync_out; echo "[colab] final sync done"' EXIT
case "$JOB" in
  o31-410m) python experiments/O_opener/hishi_akebono.py --model EleutherAI/pythia-410m --out hpcc/results/o31 --resume --purge ;;
  o31-1b)   python experiments/O_opener/hishi_akebono.py --model EleutherAI/pythia-1b   --out hpcc/results/o31 --resume --purge ;;
  o41-p0-gemma) python experiments/O_opener/bamboo_memory.py --p0 --model google/gemma-2-9b --out hpcc/results/o41 --dtype float32 --n-gauge 20 ;;
  o41-p0-llama) python experiments/O_opener/bamboo_memory.py --p0 --model meta-llama/Meta-Llama-3-8B --out hpcc/results/o41 --dtype float32 --n-gauge 20 ;;
  o41-assemble-gemma) python experiments/O_opener/bamboo_memory.py --assemble --model google/gemma-2-9b --out hpcc/results/o41 ;;
  o41-assemble-llama) python experiments/O_opener/bamboo_memory.py --assemble --model meta-llama/Meta-Llama-3-8B --out hpcc/results/o41 ;;
  o2)       python experiments/O_opener/satono_crown.py ;;
  f2-s*)    i="${JOB#f2-s}"; cd experiments/F_gauge && python rice_shower.py --gpu --min-hidden 30 --max-hidden 30 --procs "$(nproc)" --valid-budget 500000000 --shard "$i" "$F2_SHARDS" --out "../../hpcc/results/f2_d3w10/shard_$i" ;;
  v7-410m)  python hpcc/mejiro_ardan.py --model EleutherAI/pythia-410m --out hpcc/results/v7 --resume ;;
  v7-1b)    python hpcc/mejiro_ardan.py --model EleutherAI/pythia-1b   --out hpcc/results/v7 --resume ;;
  v1-s*)    i="${JOB#v1-s}"; python hpcc/mejiro_dober_large.py --out hpcc/results/v1_large --shard "$i" 8 ;;
  m4)       python hpcc/admire_vega.py --train-vit --out hpcc/results/m4 --resume ;;
  b5)       python hpcc/mejiro_ryan.py --out hpcc/results/b5 --resume ;;
  k7)       python hpcc/mayano_top_gun.py --model google/gemma-2-2b --out hpcc/results/k7_gemma2_2b --resume ;;
  f12-gemma) python hpcc/kitasan_black.py --model google/gemma-2-9b --layers 8 20 32 --n 20 --out hpcc/results/f12_gemma2_9b --resume ;;
  f12-llama) python hpcc/kitasan_black.py --model meta-llama/Meta-Llama-3-8B --layers 8 16 24 --n 20 --out hpcc/results/f12_llama3_8b --resume ;;
  c4-gemma)  python hpcc/sirius_symboli.py --model google/gemma-2-9b --out hpcc/results/c4_gemma2_9b --resume ;;
  c4-llama)  python hpcc/sirius_symboli.py --model meta-llama/Meta-Llama-3-8B --layer 16 --out hpcc/results/c4_llama3_8b --resume ;;
  k8-gemma)  python hpcc/biwa_hayahide.py --model google/gemma-2-9b --layers 9 20 31 --out hpcc/results/k8_gemma2_9b --resume ;;
  l7-gemma)  python hpcc/kawakami_princess.py --model google/gemma-2-9b --out hpcc/results/l7 ;;
  *) echo "unknown job $JOB"; exit 2 ;;
esac 2>&1 | tee -a "hpcc/logs/colab_$JOB.log"
echo "[colab] JOB $JOB finished rc=${PIPESTATUS[0]}"
