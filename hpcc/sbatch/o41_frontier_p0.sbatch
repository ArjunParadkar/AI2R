#!/bin/bash
# O4.1 Bamboo Memory, step 2 (P0): the frontier attribution graph.  Array over the two T4 models (SPEC_O4.md step 2:
# layer window = mid-2 .. mid+2, probe = the fixed 2 000-token OpenWebText-10k slice, as V7).
# GPU: >= 40 GB requested to match the other frontier jobs; the measured need is the model in bf16 (Llama-3-8B ~16 GB,
# Gemma-2-9B ~18.5 GB) plus < 2 GB of activations at this probe size, so a 24 GB card would also carry it.
# Time: 02:00:00 — the compute itself is minutes (laptop pre-flight: GPT-2 small, 65 nodes, 2 016 tokens = 2.1 s EAP +
# 3.1 s graph); the wall time is the first-run Hugging Face download of a 16-19 GB checkpoint.  Resumable per batch
# (hpcc/results/o41/p0_<slug>_E.npz), so a time-out is re-submitted, not restarted.
#SBATCH --job-name=wb-o41_frontier_p0
#SBATCH --partition=__PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --gres=__GRES__
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=hpcc/logs/%x-%A_%a.out
cd "${SLURM_SUBMIT_DIR:?submit from the whitebox-v2 checkout root}"; source hpcc/env.sh; set -x
M=(meta-llama/Meta-Llama-3-8B google/gemma-2-9b)
L=("14 15 16 17 18" "19 20 21 22 23")          # mid-2 .. mid+2 of 32 and 42 layers (SPEC_O4.md, frozen before compute)
python experiments/O_opener/bamboo_memory.py --p0 --model "${M[$SLURM_ARRAY_TASK_ID]}" --layers ${L[$SLURM_ARRAY_TASK_ID]} \
    --n-tokens 2000 --batch 2 --dtype bfloat16 --out hpcc/results/o41 --resume
