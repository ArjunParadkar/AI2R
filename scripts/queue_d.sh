#!/bin/bash
# gated on the structure-penalty runs (frees ~1.5 GB of GPU); strictly one job at a time
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 20; done; done
python analysis/resnet_depth.py > logs/resnet_depth.log 2>&1; echo "[queue_d] resnet_depth $(date +%H:%M)"
python analysis/resnet_analysis.py > logs/resnet_analysis_v2.log 2>&1; echo "[queue_d] resnet_analysis v2 $(date +%H:%M)"
python analysis/z113_unlabeled_G.py > logs/z113_unlabeled_G.log 2>&1; echo "[queue_d] unlabeled G $(date +%H:%M)"
