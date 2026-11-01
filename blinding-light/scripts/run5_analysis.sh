#!/bin/bash
# Run 5 analysis queue.  Waits on PROCESS EXIT (never on a log line).  Z113 analysis is CPU-only and starts when the seven Z113 trainers are gone;
# the ResNet analysis starts when the ResNet trainers are gone.  Nothing here scores anything: scoring is analysis/run5_score.py, run by hand after PAUSE 1.
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export WB_DEV=cpu
while [ $(ps -eo args | grep -c "[s]ystems/z113.py --mode") -gt 0 ]; do sleep 15; done
echo "[r5-a] Z113 trainers gone $(date +%H:%M)"
T=2; export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
for s in 9 10 11; do python analysis/run5_z113.py sweep r5_z113_clean_s$s $s clean 0.3 > logs/r5_sweep_clean_s$s.log 2>&1 & done
for s in 9 10; do python analysis/run5_z113.py sweep r5_z113_scrambled_decoy_s$s $s scrambled_decoy 0.3 > logs/r5_sweep_mem_s$s.log 2>&1 &
                  python analysis/run5_z113.py sweep r5_z113_decoy_s$s $s decoy 0.5 > logs/r5_sweep_decoy_s$s.log 2>&1 & done
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python analysis/run5_z113.py finals > logs/r5_z113_finals.log 2>&1 &
( while [ $(ps -eo args | grep -c "[s]ystems/resnet_subset.py") -gt 0 ]; do sleep 15; done; echo "[r5-a] ResNet trainers gone $(date +%H:%M)"
  WB_DEV=cuda python analysis/run5_resnet.py gate > logs/r5_resnet_gate.log 2>&1; WB_DEV=cuda python analysis/run5_resnet.py > logs/r5_resnet.log 2>&1 ) &
wait; echo "[r5-a] all analysis processes exited $(date +%H:%M)"; grep -l Traceback logs/r5_*.log
