#!/bin/bash
#SBATCH --job-name=wb-c4_llama3_8b
#SBATCH --partition=__PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --gres=__GRES__
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=hpcc/logs/%x-%A_%a.out
cd "${SLURM_SUBMIT_DIR:?submit from the whitebox-v2 checkout root}"; source hpcc/env.sh; set -x
python hpcc/sirius_symboli.py --model meta-llama/Meta-Llama-3-8B --layer 16 --out hpcc/results/c4_llama3_8b --resume
