#!/bin/bash
#SBATCH -o /scrfs/storage/paradkar/wb/logs/%x-%j.out
#SBATCH --exclude=c1905
set -uo pipefail
: "${HOME:=/home/paradkar}"; export HOME
cd $HOME/wb-ee-dev/bnbtest
source /scrfs/storage/paradkar/lab/miniforge3/etc/profile.d/conda.sh
conda activate /scrfs/storage/paradkar/envs/wb-ee
export HF_HOME=/scrfs/storage/paradkar/wb/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
NV=/scrfs/storage/paradkar/envs/wb-ee/lib/python3.11/site-packages/nvidia
export BNB_CUDA_VERSION=123
export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cusparse/lib:$NV/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
nvidia-smi --query-gpu=name --format=csv,noheader


echo "== shim, pythia-1.4b"
USE_SHIM=1 python hfsys_test.py EleutherAI/pythia-1.4b add3 2>&1 | grep -v "non-existent\|Loading checkpoint\|^\s*warn\|UserWarning"
echo "== shim, Qwen2.5-7B"
USE_SHIM=1 python hfsys_test.py Qwen/Qwen2.5-7B add3 2>&1 | grep -v "non-existent\|Loading checkpoint\|^\s*warn\|UserWarning"
echo "== launcher path: shim -> bench.k12 stage 1 (pythia-1.4b add3 8-bit, --gate_exit; output to the dev tree)"
cd $HOME/wb-ee && WB_COMMIT=$(cat COMMIT) timeout 1200 python $HOME/wb-ee-dev/bnbtest/bnb8_k12.py --model EleutherAI/pythia-1.4b --task add3 --seed 0 --load_in_8bit --gate_exit --out $HOME/wb-ee-dev/bnbtest/out 2>&1 | grep -v "non-existent\|Loading checkpoint\|^\s*warn\|UserWarning\|BNB_CUDA_VERSION=XXX\|If this was\|If you use\|For example\|^=*$\|^\s*$" | cut -c1-240
echo "rc=${PIPESTATUS[0]}"; ls -la $HOME/wb-ee-dev/bnbtest/out/* 2>&1
echo "== done $(date -Is)"
