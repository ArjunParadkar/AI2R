#!/bin/bash
# re-sweep the clean control (for G_interior) once sweep2 has released the cores
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
while systemctl --user is-active --quiet wb-sweep2; do sleep 30; done
python analysis/z113_sweep.py --run runs/z113_cleanctrl_s2 --mode clean --seed 2 > logs/sweep3_cleanctrl_s2.log 2>&1
python analysis/score_p6_p7.py > logs/score_p6_p7.log 2>&1
python analysis/fig_grokking.py >> logs/score_p6_p7.log 2>&1
