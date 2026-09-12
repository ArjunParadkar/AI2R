#!/bin/bash
#SBATCH --job-name=wb-f2_d3w10_enumeration
#SBATCH --partition=__PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --gres=__GRES__
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-19
#SBATCH --output=hpcc/logs/%x-%A_%a.out
cd "${SLURM_SUBMIT_DIR:?submit from the whitebox-v2 checkout root}"; source hpcc/env.sh; set -x
cd experiments/F_gauge; python rice_shower.py --gpu --min-hidden 30 --max-hidden 30 --procs 4 --valid-budget 500000000 --shard $SLURM_ARRAY_TASK_ID 20 --out ../../hpcc/results/f2_d3w10/shard_$SLURM_ARRAY_TASK_ID
