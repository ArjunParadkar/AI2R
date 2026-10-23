#!/bin/bash
# Phase 2 lane runner. One lane = one job at a time (same job name + --dependency=singleton).
# The lane walks the cell indices in $ORDER (line numbers of $CELLS), claims the first cell that is not
# finished and not claimed by a live job, runs it (bench.k12 resumes by stage), and repeats until the
# job's time is nearly used up; then it resubmits itself if eligible cells remain.
#
#   sbatch --export=NONE,CELLS=runs/cells/phase2.txt,ORDER=runs/cells/phase2_order_short.txt,SKIP='14B|13B|12b|32B',MAXTRY=2,TLIM=6:00:00 \
#          -J wb-k12-S1 -p agpu06 -q gpu -c 24 --gres=gpu:1 -t 6:00:00 --dependency=singleton runs/slurm/k12_lane.sbatch
#
# SKIP: cells whose line matches this regex are not started here unless their graph.pkl already exists
# (discovery done; the remaining stage may fit). MAXTRY: starts per cell on this lane class before the
# cell is left to the long lanes.
#SBATCH -o /scrfs/storage/paradkar/wb/logs/%x-%j.out
set -uo pipefail
cd $HOME/wb-ee
source /scrfs/storage/paradkar/lab/miniforge3/etc/profile.d/conda.sh
conda activate /scrfs/storage/paradkar/envs/wb-ee
export HF_HOME=/scrfs/storage/paradkar/wb/hf
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WB_COMMIT=$(cat COMMIT 2>/dev/null || echo unknown)
R=/scrfs/storage/paradkar/wb/results/k12
ST=/scrfs/storage/paradkar/wb/state/k12
KIND=$(echo "$SLURM_JOB_NAME" | sed -E 's/.*-([A-Z])[0-9]+$/\1/')   # S (short) or L (long)
mkdir -p $ST/lock $ST/tries_$KIND $ST/failed
SKIP=${SKIP:-__none__}; MAXTRY=${MAXTRY:-99}; TLIM=${TLIM:-6:00:00}
lim_s() { IFS=: read -r h m s <<< "$1"; echo $((10#$h * 3600 + 10#$m * 60 + 10#$s)); }
T0=$(date +%s); END=$((T0 + $(lim_s $TLIM) - 600)); MIN_START=${MIN_START:-1800}
echo "[wb] lane $SLURM_JOB_NAME start $(date -Is) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) commit=$WB_COMMIT"

celldir() {  # mirror of bench.k12's cell naming
  local line="$1" model task seed
  model=$(echo "$line" | sed -E 's/.*--model ([^ ]+).*/\1/'); task=$(echo "$line" | sed -E 's/.*--task ([^ ]+).*/\1/')
  seed=$(echo "$line" | sed -E 's/.*--seed ([0-9]+).*/\1/')
  local d="$R/${model//\//__}_${task}_s${seed}"
  [[ "$line" == *--untrained* ]] && d="${d}_untrained"
  [[ "$line" == *--load_in_8bit* ]] && d="${d}_8bit"
  echo "$d"
}
is_done() {
  local line="$1" d; d=$(celldir "$line")
  if [[ "$line" == *--untrained* ]]; then [[ -f $d/k1.json ]]; else [[ -f $d/k2.pkl || -f $d/GATE_FAILED ]]; fi
}
eligible() {  # prints 1 if this lane may start cell i
  local i=$1 line; line=$(sed -n "${i}p" "$CELLS")
  is_done "$line" && return 1
  [[ -f $ST/failed/$i ]] && return 1          # crashed (not a timeout): needs a look, never retried blindly
  if [[ "$line" =~ $SKIP ]] && [[ ! -f $(celldir "$line")/graph.pkl ]]; then return 1; fi
  local n=$(cat $ST/tries_$KIND/$i 2>/dev/null || echo 0)
  (( n >= MAXTRY )) && return 1
  return 0
}
claim() {
  local i=$1
  if mkdir $ST/lock/$i 2>/dev/null; then echo $SLURM_JOB_ID > $ST/lock/$i/job; return 0; fi
  local j=$(cat $ST/lock/$i/job 2>/dev/null)
  if [[ -n "$j" ]] && ! squeue -h -j "$j" 2>/dev/null | grep -q .; then   # stale lock (job gone)
    rm -rf $ST/lock/$i && mkdir $ST/lock/$i 2>/dev/null && { echo $SLURM_JOB_ID > $ST/lock/$i/job; return 0; }
  fi
  return 1
}

while :; do
  now=$(date +%s); (( END - now < MIN_START )) && { echo "[wb] lane time nearly used"; break; }
  picked=""
  for i in $(cat "$ORDER"); do
    eligible $i || continue
    claim $i && { picked=$i; break; }
  done
  [[ -z "$picked" ]] && { echo "[wb] no eligible cell"; break; }
  line=$(sed -n "${picked}p" "$CELLS")
  n=$(cat $ST/tries_$KIND/$picked 2>/dev/null || echo 0); echo $((n + 1)) > $ST/tries_$KIND/$picked
  echo "[wb] cell $picked start $(date -Is) try $((n + 1)): $line"
  timeout --signal=TERM --kill-after=60 $((END - $(date +%s))) python -m bench.k12 $line --gate_exit --out $R
  rc=$?
  echo "[wb] cell $picked end $(date -Is) rc=$rc"
  if (( rc != 0 && rc != 124 && rc != 137 )) && ! is_done "$line"; then
    echo "rc=$rc job=$SLURM_JOB_ID $(date -Is)" > $ST/failed/$picked; echo "[wb] cell $picked marked failed"
  fi
  rm -rf $ST/lock/$picked
done

# resubmit this lane if any cell is still eligible for it
left=0; for i in $(cat "$ORDER"); do eligible $i && { left=1; break; }; done
if (( left )); then
  sbatch --export=NONE,CELLS=$CELLS,ORDER=$ORDER,SKIP="$SKIP",MAXTRY=$MAXTRY,TLIM=$TLIM -J $SLURM_JOB_NAME \
    -p $SLURM_JOB_PARTITION -q gpu -c $SLURM_CPUS_PER_TASK --gres=gpu:1 -t $TLIM --dependency=singleton \
    runs/slurm/k12_lane.sbatch && echo "[wb] lane resubmitted"
fi
echo "[wb] lane $SLURM_JOB_NAME end $(date -Is)"
