#!/bin/bash
#SBATCH -J wb-env
#SBATCH -p cloud72
#SBATCH -q cloud
#SBATCH -c 4
#SBATCH -t 06:00:00
#SBATCH -o /scrfs/storage/paradkar/wb/logs/%x-%j.out
# Builds /scrfs/storage/paradkar/envs/wb-ee (idempotent: skips if marker exists).
set -euo pipefail
ENV=/scrfs/storage/paradkar/envs/wb-ee
MF=/scrfs/storage/paradkar/lab/miniforge3
echo "[wb-env] start $(date -Is) node=$(hostname)"
if [ -f $ENV/.wb_ok ]; then echo "already built"; exit 0; fi
export CONDA_OVERRIDE_GLIBC=2.17
export PIP_CACHE_DIR=/scrfs/storage/paradkar/wb/pipcache
export CONDA_PKGS_DIRS=/scrfs/storage/paradkar/wb/condapkgs
source $MF/etc/profile.d/conda.sh
if [ ! -x $ENV/bin/python ]; then
  conda create -y -p $ENV -c conda-forge --override-channels python=3.11 pyarrow=16 numpy=1.26 scipy pandas scikit-learn networkx pip
fi
conda activate $ENV
pip install --no-input torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-input "transformers>=4.46,<4.50" "datasets>=2.20,<4" "accelerate>=1.0,<1.3" "bitsandbytes==0.45.0" safetensors sentencepiece tiktoken "huggingface_hub<0.30" pytest hypothesis einops
python -c "import torch,transformers,datasets,pyarrow,sklearn;print(torch.__version__,torch.version.cuda,transformers.__version__,datasets.__version__,pyarrow.__version__)"
touch $ENV/.wb_ok
echo "[wb-env] end $(date -Is)"
