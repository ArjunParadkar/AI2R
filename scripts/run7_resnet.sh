#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate; export OMP_NUM_THREADS=4
python analysis/run7_resnet.py X > logs/r7_X.log 2>&1 && python analysis/run7_resnet.py K > logs/r7_K.log 2>&1 && python analysis/run7_resnet.py P5 > logs/r7_P5_resnet.log 2>&1; echo "[r7] resnet chain exited $(date +%H:%M)"
