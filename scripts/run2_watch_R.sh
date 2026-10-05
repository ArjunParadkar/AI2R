#!/bin/bash
# stop wb-r2-D34 as soon as R seed 3 has written its step-12000 record (before the script's own G loop can overwrite anything)
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"
while true; do
  if grep -q '"step": 12000' runs/r2_align_R_s3/history.json 2>/dev/null; then systemctl --user stop wb-r2-D34; echo "[watch] stopped wb-r2-D34 $(date +%H:%M:%S)"; break; fi
  mkdir -p results/run2/G_backup; for s in 0 1 2 3 4; do cp -f runs/r2_align_G_s$s/history.json results/run2/G_backup/history_s$s.json 2>/dev/null; done
  sleep 2
done
