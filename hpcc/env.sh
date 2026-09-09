#!/usr/bin/env bash
# HPCC environment recipe (Arkansas cluster) — TEMPLATE: module names to be filled on first SSH session (2026-09-09).
# Usage: source hpcc/env.sh   (from the whitebox-v2 checkout on the cluster)
set -e
# module load python/3.12 cuda/12.8        # TODO: replace with `module avail python cuda` results
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN (Gemma / Llama licences) before sourcing}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -d "$ROOT/.venv-hpcc" ]; then
  python -m venv "$ROOT/.venv-hpcc"; "$ROOT/.venv-hpcc/bin/pip" install -U pip
  "$ROOT/.venv-hpcc/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu128     # match the cluster's CUDA
  "$ROOT/.venv-hpcc/bin/pip" install -e "$ROOT[dev]" transformers accelerate transformer_lens sae_lens spacy pyarrow
  "$ROOT/.venv-hpcc/bin/python" -m spacy download en_core_web_sm
fi
export PATH="$ROOT/.venv-hpcc/bin:$PATH"; mkdir -p "$ROOT/hpcc/logs" "$ROOT/hpcc/results"
# smoke runs first (one unit of work each), then the full order from hpcc/README.md:
#   python hpcc/f12_frontier_audit.py --model google/gemma-2-9b --layers 20 --n 2 --n-tokens 250 --out hpcc/results/f12_smoke
#   python hpcc/k8_frontier_residue.py --layers 20 --n-seq 8 --out hpcc/results/k8_smoke
