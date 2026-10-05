#!/bin/bash
# second sequential queue, gated on queue_a's PID
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 20; done; done
python analysis/z113_unlabeled_G.py > logs/z113_unlabeled_G.log 2>&1
echo "[queue_b] unlabeled G done $(date +%H:%M)"
