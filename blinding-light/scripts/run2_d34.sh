#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for sched in R G; do
  for s in 0 1 2 3 4; do python analysis/run2_blockD34.py --seed $s --sched $sched > logs/run2_align_${sched}_s$s.log 2>&1 & done
  wait
done
echo "[d34] done $(date +%H:%M)"
