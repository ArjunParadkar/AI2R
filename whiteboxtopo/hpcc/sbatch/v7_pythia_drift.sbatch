#!/bin/bash
#SBATCH --job-name=wb-v7_pythia_drift
#SBATCH --partition=__PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --gres=__GRES__
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=hpcc/logs/%x-%A_%a.out
cd "${SLURM_SUBMIT_DIR:?submit from the whitebox-v2 checkout root}"; source hpcc/env.sh; set -x
M=(EleutherAI/pythia-410m EleutherAI/pythia-1b); python hpcc/mejiro_ardan.py --model ${M[$SLURM_ARRAY_TASK_ID]} --out hpcc/results/v7 --resume
