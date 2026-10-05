#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
z () { python systems/z113.py --mode $1 --seed $2 --steps 25000 --frac $3 --outdir runs/r6_z113_$1_s$2 > logs/r6_z113_$1_s$2.log 2>&1 & }
for s in 12 13 14; do z clean $s 0.3; done; for s in 15 16 17; do z scrambled_decoy $s 0.3; done; for s in 12 13; do z decoy $s 0.5; done
( for s in 5 6 7; do python systems/resnet_subset.py --mode scrambled --seed $s --outdir runs/r6_resnet_scrambled_s$s > logs/r6_resnet_scrambled_s$s.log 2>&1; done
  for s in 5 6; do python systems/resnet_subset.py --mode clean --aug --max-epochs 300 --seed $s --outdir runs/r6_resnet_cleanaug_s$s > logs/r6_resnet_cleanaug_s$s.log 2>&1; done ) &
wait; echo "[r6] all specimens done $(date +%H:%M)"
