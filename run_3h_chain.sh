#!/bin/bash
cd /home/god/whitebox_experiments_clone
export PYTHONPATH=/home/god/whitebox_experiments_clone:/home/god/whitebox_experiments_clone/site_pkgs
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONWARNINGS=ignore
PY=/home/god/Desktop/whitebox/venv/bin/python
N="$1"
echo "[chain] starting B at $(date +%T)"
$PY -W ignore expB_crystallization.py --out "$N/raw/expB_crystallization.json" --ckpt "$N/raw/resnet_seed1_ep60.pt" --epochs 60 >> "$N/logs/expB.log" 2>&1
echo "[chain] B done; starting C at $(date +%T)"
$PY -W ignore expC_failure.py --ckpt "$N/raw/resnet_seed1_ep60.pt" --bjson "$N/raw/expB_crystallization.json" --out "$N/raw/expC_failure.json" >> "$N/logs/expC.log" 2>&1
echo "[chain] C done; starting D at $(date +%T)"
$PY -W ignore expD_crossmodal.py --out "$N/raw/expD_crossmodal.json" --epochs 30 >> "$N/logs/expD.log" 2>&1
echo "[chain] D done; generating outputs at $(date +%T)"
$PY -W ignore gen_3h_outputs.py --dir "$N" >> "$N/logs/outputs.log" 2>&1
echo "[chain] ALL DONE at $(date +%T)"
