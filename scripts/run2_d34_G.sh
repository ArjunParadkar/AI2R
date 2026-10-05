#!/bin/bash
# schedule G, 6000 steps (AMENDMENTS A-5)
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1 2 3 4; do python analysis/run2_blockD34.py --seed $s --sched G --steps 6000 > logs/run2_align_G_s$s.log 2>&1 & done
wait; echo "[d34-G] done $(date +%H:%M)"
