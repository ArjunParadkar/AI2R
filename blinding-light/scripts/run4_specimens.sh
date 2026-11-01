#!/bin/bash
# Run 4 specimens (fixed by the brief; launched before SPEC4.md is finished; nothing analyses them until SPEC4 exists)
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
z () { python systems/z113.py --mode $1 --seed $2 --steps 25000 --frac $3 --outdir runs/r4_z113_$1_s$2 > logs/r4_z113_$1_s$2.log 2>&1 & }
( for mode in scrambled clean; do for s in 1 2; do python systems/resnet_subset.py --mode $mode --seed $s --outdir runs/r4_resnet_${mode}_s$s > logs/r4_resnet_${mode}_s$s.log 2>&1; done; done ) &
RES=$!
for s in 5 6 7 8; do z clean $s 0.3; done; for s in 5 6; do z scrambled_decoy $s 0.3; done
while [ $(ps -eo args | grep -c "[s]ystems/z113.py --mode") -gt 0 ]; do sleep 10; done
echo "[r4] wave 1 done $(date +%H:%M)"
for s in 5 6; do z decoy $s 0.5; z shortcut $s 0.5; done; z aonly 5 0.3
wait; echo "[r4] all specimens done $(date +%H:%M)"
