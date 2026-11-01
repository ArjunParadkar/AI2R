#!/bin/bash
cd "/home/god/Desktop/Arenda_Projects/WhiteBOX/WhiteBOX - BlindingLight"
source .venv/bin/activate
mkdir -p runs logs
launch () { # mode seed frac tag
  python systems/z113.py --mode $1 --seed $2 --steps 25000 --frac $3 \
     --outdir runs/z113_$4 > logs/z113_$4.log 2>&1 &
  sleep 3
}
launch clean     0 0.3 clean_s0
launch clean     1 0.3 clean_s1
launch scrambled 0 0.3 scrambled_s0
launch clean     2 0.5 cleanctrl_s2
launch shortcut  0 0.5 shortcut_s0
launch shortcut  1 0.5 shortcut_s1
wait
echo "ALL Z113 DONE"
