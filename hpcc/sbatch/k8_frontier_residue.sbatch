#!/bin/bash
#SBATCH --job-name=wb-k8_frontier_residue
#SBATCH --partition=__PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --gres=__GRES__
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=hpcc/logs/%x-%A_%a.out
cd "${SLURM_SUBMIT_DIR:?submit from the whitebox-v2 checkout root}"; source hpcc/env.sh; set -x
python hpcc/biwa_hayahide.py --model google/gemma-2-9b --layers 9 20 31 --out hpcc/results/k8_gemma2_9b --resume
