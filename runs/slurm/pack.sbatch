#!/bin/bash
# Packed CPU runner: one whole-node job runs every line of $CMDS, $PAR at a time, each with $NT threads.
#   sbatch --export=NONE,CMDS=runs/cells/x.txt,PAR=8,NT=4 -J wb-x -p comp72 -q comp -c 32 runs/slurm/pack.sbatch
#SBATCH -t 72:00:00
#SBATCH -o /scrfs/storage/paradkar/wb/logs/%x-%j.out
set -uo pipefail
cd $HOME/wb-ee
source /scrfs/storage/paradkar/lab/miniforge3/etc/profile.d/conda.sh
conda activate /scrfs/storage/paradkar/envs/wb-ee
export HF_HOME=/scrfs/storage/paradkar/wb/hf
NT=${NT:-4}; PAR=${PAR:-8}
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WB_COMMIT=$(cat COMMIT 2>/dev/null || echo unknown)
echo "[wb] start $(date -Is) node=$(hostname) commit=$WB_COMMIT cmds=$CMDS par=$PAR nt=$NT"
LOGDIR=/scrfs/storage/paradkar/wb/logs/pack-$SLURM_JOB_ID; mkdir -p $LOGDIR
nl -ba "$CMDS" | xargs -P $PAR -I{} bash -c 'line="{}"; i=$(echo "$line" | awk "{print \$1}"); cmd=$(echo "$line" | cut -f2-); echo "[wb] cell $i start $(date -Is)"; eval "$cmd" > '$LOGDIR'/cell_$i.log 2>&1; echo "[wb] cell $i end $(date -Is) rc=$?"'
echo "[wb] end $(date -Is)"
