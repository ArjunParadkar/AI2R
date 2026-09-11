#!/bin/bash
#SBATCH --job-name=wb-k7_width_scaling
#SBATCH --partition=__PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --gres=__GRES__
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=hpcc/logs/%x-%A_%a.out
cd "${SLURM_SUBMIT_DIR:?submit from the whitebox-v2 checkout root}"; source hpcc/env.sh; set -x
python hpcc/mayano_top_gun.py --model google/gemma-2-2b --out hpcc/results/k7_gemma2_2b --resume
