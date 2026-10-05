#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python analysis/z113_static.py --run runs/z113_clean_s1 --mode clean --seed 1 > logs/static_clean_s1.log 2>&1
python analysis/z113_static.py --run runs/z113_clean_s0 --mode clean --seed 0 > logs/static_clean_s0.log 2>&1
python analysis/z113_static.py --run runs/z113_cleanctrl_s2 --mode clean --seed 2 > logs/static_cleanctrl_s2.log 2>&1
