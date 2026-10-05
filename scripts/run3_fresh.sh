#!/bin/bash
# Run 3, P6.4: the only retraining in the run -- fresh seeds, fixed by the brief (launched 13:40, before SPEC3.md was finished; no analysis touches them until SPEC3 is written)
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for s in 3 4; do
  python systems/z113.py --mode clean --seed $s --steps 25000 --frac 0.3 --outdir runs/r3_z113_clean_s$s > logs/r3_z113_clean_s$s.log 2>&1 &
  python systems/z113.py --mode scrambled --seed $s --steps 25000 --frac 0.3 --outdir runs/r3_z113_scrambled_s$s > logs/r3_z113_scrambled_s$s.log 2>&1 &
done
for s in 1 2; do for mode in clean scrambled; do
  python systems/resnet_cifar.py --mode $mode --seed $s --epochs 15 --outdir runs/r3_resnet_${mode}_s$s > logs/r3_resnet_${mode}_s$s.log 2>&1
done; done
wait; echo "[fresh] done $(date +%H:%M)"
