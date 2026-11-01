#!/bin/bash
# wave 2 + ResNet (A-13).  Starts when wave 1's six history files exist; stops the original unit so it cannot start its own wave 2.
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
until [ $(ls runs/r4_z113_clean_s{5,6,7,8}/history.json runs/r4_z113_scrambled_decoy_s{5,6}/history.json 2>/dev/null | wc -l) -eq 6 ]; do sleep 5; done
systemctl --user stop wb-r4-spec; sleep 3; rm -rf runs/r4_resnet_scrambled_s1 runs/r4_z113_decoy_s5 runs/r4_z113_decoy_s6 runs/r4_z113_shortcut_s5 runs/r4_z113_shortcut_s6 runs/r4_z113_aonly_s5
echo "[r4-2] wave 1 done, starting wave 2 $(date +%H:%M)"
z () { python systems/z113.py --mode $1 --seed $2 --steps $4 --frac $3 --outdir runs/r4_z113_$1_s$2 > logs/r4_z113_$1_s$2.log 2>&1 & }
for s in 5 6; do z decoy $s 0.5 20000; z shortcut $s 0.5 8000; done; z aonly 5 0.3 8000
for mode in scrambled clean; do for s in 1 2; do python systems/resnet_subset.py --mode $mode --seed $s --outdir runs/r4_resnet_${mode}_s$s > logs/r4_resnet_${mode}_s$s.log 2>&1; done; done
wait; echo "[r4-2] all specimens done $(date +%H:%M)"
