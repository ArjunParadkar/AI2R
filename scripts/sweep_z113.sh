#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"
source .venv/bin/activate
python analysis/z113_sweep.py --run runs/z113_clean_s1     --mode clean     --seed 1 > logs/sweep_clean_s1.log 2>&1
python analysis/z113_sweep.py --run runs/z113_clean_s0     --mode clean     --seed 0 > logs/sweep_clean_s0.log 2>&1
python analysis/z113_sweep.py --run runs/z113_scrambled_s0 --mode scrambled --seed 0 > logs/sweep_scrambled_s0.log 2>&1
python analysis/z113_sweep.py --run runs/z113_shortcut_s0  --mode shortcut  --seed 0 > logs/sweep_shortcut_s0.log 2>&1
python analysis/z113_sweep.py --run runs/z113_shortcut_s1  --mode shortcut  --seed 1 > logs/sweep_shortcut_s1.log 2>&1
python analysis/z113_sweep.py --run runs/z113_cleanctrl_s2 --mode clean     --seed 2 > logs/sweep_cleanctrl_s2.log 2>&1
echo "SWEEPS DONE"
