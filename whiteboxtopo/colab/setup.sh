#!/usr/bin/env bash
# Colab environment for WHITEBOX jobs. Run from the repo root inside a Colab session (torch + CUDA already present).
#   source colab/setup.sh
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
pip install -q -e ".[dev]" transformers accelerate datasets transformer_lens sae_lens spacy pyarrow ripser persim huggingface_hub 2>&1 | tail -1
python -m spacy download en_core_web_sm -q 2>&1 | tail -1 || true
export HF_HOME="${HF_HOME:-/content/hf}"; mkdir -p "$HF_HOME"
if [ -n "${HF_TOKEN:-}" ]; then python -c "from huggingface_hub import login; login('$HF_TOKEN', add_to_git_credential=False)" 2>&1 | tail -1; fi
# the K1/F11 probe corpus (public): download once, point the loaders at it
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download("stas/openwebtext-10k", "plain_text/train/0000.parquet", repo_type="dataset")
open("/content/.owt_parquet", "w").write(p); print("openwebtext-10k parquet:", p)
PY
export WB_OWT_PARQUET="$(cat /content/.owt_parquet)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p hpcc/results hpcc/logs
python -c "import whiteboxtopo, transformer_lens, torch; print('whiteboxtopo ok; cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
