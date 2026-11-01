#!/bin/bash
# Phase 2 K1/K2 cells. One array task = one line of the cell list ($CELLS).
#   sbatch --export=NONE,CELLS=runs/cells/phase2_a100.txt -p agpu72 -q gpu -c 24 --gres=gpu:1 \
#          --array=1-N%4 runs/slurm/k12.sbatch
#SBATCH -J wb-k12
#SBATCH -t 72:00:00
#SBATCH -o /scrfs/storage/paradkar/wb/logs/%x-%A_%a.out
set -uo pipefail
cd $HOME/wb-ee
source /scrfs/storage/paradkar/lab/miniforge3/etc/profile.d/conda.sh
conda activate /scrfs/storage/paradkar/envs/wb-ee
export HF_HOME=/scrfs/storage/paradkar/wb/hf
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WB_COMMIT=$(cat COMMIT 2>/dev/null || echo unknown)
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CELLS")
echo "[wb] start $(date -Is) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) commit=$WB_COMMIT cell: $LINE"
python -m bench.k12 $LINE --out /scrfs/storage/paradkar/wb/results/k12
rc=$?
echo "[wb] end $(date -Is) rc=$rc"
exit $rc
