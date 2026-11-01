#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 WB_DEV=cpu
python analysis/run4_z113.py finals > logs/r4_z113_finals.log 2>&1 &
for s in 5 6; do OMP_NUM_THREADS=4 python analysis/run4_z113.py sweep r4_z113_decoy_s$s $s decoy 0.5 > logs/r4_sweep_decoy_s$s.log 2>&1 & done
python analysis/run4_resnet.py > logs/r4_resnet_analysis.log 2>&1
wait; echo "[r4-a2] done $(date +%H:%M)"
