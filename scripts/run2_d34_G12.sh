#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 15; done; done
for s in 1 2; do python analysis/run2_blockD34.py --seed $s --sched G --steps 6000 > logs/run2_align_G_s$s.log 2>&1 & done
wait; echo "[G12] done $(date +%H:%M)"
