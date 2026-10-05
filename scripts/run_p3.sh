#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
for s in 0 1; do
  python analysis/z113_interrupt.py --seed $s --steps 25000 --mode interrupt --outdir runs/p3_interrupt_s$s > logs/p3_interrupt_s$s.log 2>&1 &
done
wait
for s in 0 1; do
  python analysis/z113_interrupt.py --seed $s --steps 25000 --mode spare --outdir runs/p3_spare_s$s > logs/p3_spare_s$s.log 2>&1 &
  python analysis/z113_interrupt.py --seed $s --steps 25000 --mode none --outdir runs/p3_none_s$s > logs/p3_none_s$s.log 2>&1 &
done
wait; echo P3 DONE
