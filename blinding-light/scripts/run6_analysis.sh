#!/bin/bash
# Run 6 analysis queue: waits on process exit.  Z113 (CPU + a little cuda) after the Z113 trainers exit; ResNet (cuda) after the ResNet trainers exit.
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export WB_DEV=cpu
while [ $(ps -eo args | grep -c "[s]ystems/z113.py --mode") -gt 0 ]; do sleep 15; done; echo "[r6-a] Z113 trainers gone $(date +%H:%M)"
T=3; export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
for s in 12 13 14; do python analysis/run6_z113.py sweep r6_z113_clean_s$s $s clean 0.3 > logs/r6_sweep_clean_s$s.log 2>&1 & done
( while [ $(ps -eo args | grep -c "[s]ystems/resnet_subset.py") -gt 0 ]; do sleep 15; done; echo "[r6-a] ResNet trainers gone $(date +%H:%M)"
  OMP_NUM_THREADS=4 python analysis/run6_z113.py finals > logs/r6_z113_finals.log 2>&1 ) &
( while [ $(ps -eo args | grep -c "[s]ystems/resnet_subset.py") -gt 0 ]; do sleep 15; done
  WB_DEV=cuda OMP_NUM_THREADS=4 python analysis/run6_resnet.py > logs/r6_resnet.log 2>&1 ) &
wait; echo "[r6-a] all analysis processes exited $(date +%H:%M)"; grep -l Traceback logs/r6_*.log
