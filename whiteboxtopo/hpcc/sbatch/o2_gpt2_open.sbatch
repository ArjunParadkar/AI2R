#!/bin/bash
#SBATCH --job-name=wb-o2_gpt2_open
#SBATCH --partition=__PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --gres=__GRES__
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=hpcc/logs/%x-%j.out
# O2.1 Satono Crown — open() on GPT-2 small for IOI, induction and greater-than (SPEC: experiments/O_opener/SPEC_O2.md)
# plus O2.2 Narita Top Road (the F11 audit carry, no recompute; runs at the end of the same job, seconds).
# Resumable per task AND per cell: results/open/satono_crown_<task>.json is written after every cell and a re-submission
# of this same file skips the tasks whose cells are already present. A time-out is re-submitted, never restarted.
# Needs no gated model (GPT-2 is public), so HF_TOKEN is only the rate-limit token.
cd "${SLURM_SUBMIT_DIR:?submit from the whitebox-v2 checkout root}"; source hpcc/env.sh; set -x
python experiments/O_opener/satono_crown.py --tasks ioi induction greater_than --n-gauge 20 --device cuda
