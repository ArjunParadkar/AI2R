#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 20; done; done
python analysis/z113_p3_decoy.py > logs/p3_decoy.log 2>&1; echo "[queue_e] decoy scan $(date +%H:%M)"
