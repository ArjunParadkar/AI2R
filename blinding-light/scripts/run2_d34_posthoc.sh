#!/bin/bash
# POST-HOC arm (AMENDMENTS A-4): run-1's penalty (z vs z+1, z vs NULL) with the run-2 direct trigger, schedule R, 6000 steps
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1 2 3 4; do python analysis/run2_blockD34.py --seed $s --sched R --penalty run1 --steps 6000 > logs/run2_align_Rrun1pen_s$s.log 2>&1 & done
wait; echo "[d34-posthoc] done $(date +%H:%M)"
