#!/bin/bash
# starts when wave 1 (clean 5-8, memorisers 5-6) has finished training; CPU only
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
until [ $(ls runs/r4_z113_clean_s{5,6,7,8}/history.json runs/r4_z113_scrambled_decoy_s{5,6}/history.json 2>/dev/null | wc -l) -eq 6 ]; do sleep 10; done
echo "[r4-a1] start $(date +%H:%M)"; export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 WB_DEV=cpu
python analysis/run4_z113.py finals > logs/r4_z113_finals_wave1.log 2>&1 &
for s in 5 6 7 8; do python analysis/run4_z113.py sweep r4_z113_clean_s$s $s clean 0.3 > logs/r4_sweep_clean_s$s.log 2>&1 & done
for s in 5 6; do python analysis/run4_z113.py sweep r4_z113_scrambled_decoy_s$s $s scrambled_decoy 0.3 > logs/r4_sweep_mem_s$s.log 2>&1 & done
wait; echo "[r4-a1] done $(date +%H:%M)"
