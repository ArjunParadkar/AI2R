#!/bin/bash
# more seeds of the PRE-REGISTERED interrupt (x10 escalation, check every 500), 6000 steps each,
# plus one POST-HOC schedule on the seed that collapsed (x3 escalation, check every 250).
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"; source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for pid in "$@"; do while kill -0 "$pid" 2>/dev/null; do sleep 20; done; done
for s in 2 3 4; do
  python analysis/z113_interrupt.py --seed $s --steps 6000 --mode interrupt --outdir runs/p3_interrupt_s$s > logs/p3_interrupt_s$s.log 2>&1 &
done
python analysis/z113_interrupt.py --seed 1 --steps 6000 --mode interrupt --esc 3 --check-every 250 --max-rounds 5 --outdir runs/p3_interrupt_gentle_s1 > logs/p3_interrupt_gentle_s1.log 2>&1 &
wait; echo "[queue_f] done $(date +%H:%M)"
