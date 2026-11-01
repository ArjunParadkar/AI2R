#!/bin/bash
# Generic array runner: task i runs line i of $CMDS (a file of shell commands, run from ~/wb-ee).
#   sbatch --export=NONE,CMDS=runs/cells/x.txt -J wb-x -p cloud72 -q cloud -c 4 --array=1-N%10 runs/slurm/cmd_array.sbatch
#SBATCH -t 72:00:00
#SBATCH -o /scrfs/storage/paradkar/wb/logs/%x-%A_%a.out
set -uo pipefail
cd $HOME/wb-ee
source /scrfs/storage/paradkar/lab/miniforge3/etc/profile.d/conda.sh
conda activate /scrfs/storage/paradkar/envs/wb-ee
export HF_HOME=/scrfs/storage/paradkar/wb/hf
NT=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WB_COMMIT=$(cat COMMIT 2>/dev/null || echo unknown)
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMDS")
echo "[wb] start $(date -Is) node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) commit=$WB_COMMIT cmd: $LINE"
eval "$LINE"
rc=$?
echo "[wb] end $(date -Is) rc=$rc"
exit $rc
