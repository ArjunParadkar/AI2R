#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 20; done; done
python analysis/resnet_analysis.py > logs/resnet_analysis_v2.log 2>&1
echo "[queue_c] resnet_analysis v2 done $(date +%H:%M)"
