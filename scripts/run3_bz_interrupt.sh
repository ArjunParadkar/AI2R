#!/bin/bash
# P3.3: five seeds, ONE process (stacked with vmap; host RAM was the constraint), one unit (wb-r3-bz). The training step is a CUDA graph
# (the eager stacked step was CPU-launch-bound at <0.75 steps/s on the shared GPU); falls back to eager if the graph path fails before step 100.
# Resumes from the last 500-step check on a crash (shared-GPU OOM).
cd /home/god/blindinglight && source .venv/bin/activate
export OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for try in 1 2 3 4 5 6 7 8; do
  G=1; [ -f logs/r3_bz_nograph ] && G=0
  python analysis/run3_blockP3_train.py --seeds 0,1,2,3,4 --graph $G >> logs/r3_bz_stacked.log 2>&1; rc=$?
  echo "[r3-bz] try $try graph=$G exit $rc $(date +%H:%M:%S)" >> logs/r3_bz_stacked.log
  [ $rc -eq 0 ] && break
  grep -q "^\[stacked\] step 100 " logs/r3_bz_stacked.log || touch logs/r3_bz_nograph
  sleep 15
done
