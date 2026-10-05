#!/bin/bash
# Sequential queue, gated on PIDs (systemctl can fail transiently under load, and a failed
# is-active must not read as "finished").  One heavy CPU job at a time; BLAS capped.
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 20; done; done
echo "[queue] gate open $(date +%H:%M)"
python analysis/z113_sweep.py --run runs/z113_cleanctrl_s2 --mode clean --seed 2 > logs/sweep3_cleanctrl_s2.log 2>&1
python analysis/score_p6_p7.py > logs/score_p6_p7.log 2>&1
python analysis/fig_grokking.py >> logs/score_p6_p7.log 2>&1
echo "[queue] sweep3 done $(date +%H:%M)"
python analysis/resnet_depth.py > logs/resnet_depth.log 2>&1
echo "[queue] resnet_depth done $(date +%H:%M)"
