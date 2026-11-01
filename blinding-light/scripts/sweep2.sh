#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
python analysis/z113_sweep.py --run runs/z113_clean_s1 --mode clean --seed 1 > logs/sweep2_clean_s1.log 2>&1
python analysis/z113_sweep.py --run runs/z113_scrambled_s0 --mode scrambled --seed 0 > logs/sweep2_scrambled_s0.log 2>&1
python analysis/z113_sweep.py --run runs/z113_clean_s0 --mode clean --seed 0 > logs/sweep2_clean_s0.log 2>&1
