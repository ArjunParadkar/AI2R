#!/bin/bash
set -e
cd /home/god/Desktop/whitebox
source venv/bin/activate

echo "=== Memorization (12 epochs x 3 noise levels) ===" 
python3 exp_memorization.py --epochs 12 --ckpt-every 1 --seed 0 > mem_log.txt 2>&1

echo "=== Language (TinyGPT, 5 epochs, smaller corpus) ==="
python3 exp_language.py --epochs 5 --batch_size 32 --seq_len 96 --corpus_size 6000 \
    > lang_log.txt 2>&1

echo "=== Generating final figures and tables ==="
python3 generate_results.py > gen_log.txt 2>&1
echo "=== ALL DONE at $(date +%H:%M:%S) ==="
