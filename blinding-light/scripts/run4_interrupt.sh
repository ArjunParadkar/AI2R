#!/bin/bash
# Run 4, P3.2: five seeds concurrently on a GPU that runs nothing else.  Waits for the specimen unit to finish first.
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
while [ $(ps -eo args | grep -c "[s]ystems/z113.py --mode\|[s]ystems/resnet_subset.py\|[a]nalysis/run4_resnet.py") -gt 0 ]; do sleep 15; done
echo "[r4-int] GPU idle, starting $(date +%H:%M)"
for s in 0 1 2 3 4; do python analysis/run2_blockD34.py --seed $s --sched ONE --penalty run1 --steps 12000 --outdir runs/r4_interrupt_s$s > logs/r4_interrupt_s$s.log 2>&1 & done
wait; echo "[r4-int] done $(date +%H:%M)"
