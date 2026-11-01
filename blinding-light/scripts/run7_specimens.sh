#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 18 19 20; do python systems/z113.py --mode clean --seed $s --steps 12000 --frac 0.3 --outdir runs/r7_z113_clean_s$s > logs/r7_z113_clean_s$s.log 2>&1 & done
wait; echo "[r7] specimens done $(date +%H:%M)"
