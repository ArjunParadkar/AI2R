#!/usr/bin/env bash
# Job manifest — one line per VM:  name|sku_class|command   (command runs from the repo root, after `source hpcc/env.sh`).
# Mirrors hpcc/sbatch/*.sbatch exactly (same scripts, same --out dirs) so hpcc/results/ and the PAUSE reports stay one namespace.
source "$(dirname "${BASH_SOURCE[0]}")/config.env"
jobs() {
  local i
  for ((i=0; i<F2_SHARDS; i++)); do
    echo "f2-s$i|T4|cd experiments/F_gauge && python rice_shower.py --gpu --min-hidden 30 --max-hidden 30 --procs 6 --valid-budget $F2_VALID_BUDGET --shard $i $F2_SHARDS --out ../../hpcc/results/f2_d3w10/shard_$i"
  done
  echo "k7|T4|python hpcc/mayano_top_gun.py --model google/gemma-2-2b --out hpcc/results/k7_gemma2_2b --resume"
  echo "m4|T4|python hpcc/admire_vega.py --train-vit --out hpcc/results/m4 --resume"
  echo "b5|T4|python hpcc/mejiro_ryan.py --out hpcc/results/b5 --resume"
  echo "v7-0|T4|python hpcc/mejiro_ardan.py --model EleutherAI/pythia-410m --out hpcc/results/v7 --resume"
  echo "v7-1|T4|python hpcc/mejiro_ardan.py --model EleutherAI/pythia-1b --out hpcc/results/v7 --resume"
  for i in 0 1 2 3 4 5 6 7; do echo "v1-s$i|T4|python hpcc/mejiro_dober_large.py --out hpcc/results/v1_large --shard $i 8"; done
  echo "v5|T4|cd experiments/V_legibility && python twin_turbo.py --all-ckpts --limit 0"     # needs the V1 ckpt tarball (provision.sh --with-ckpt)
  echo "k8|A100|python hpcc/biwa_hayahide.py --model google/gemma-2-9b --layers 9 20 31 --out hpcc/results/k8_gemma2_9b --resume"
  echo "c4|A100|python hpcc/sirius_symboli.py --model google/gemma-2-9b --out hpcc/results/c4_gemma2_9b --resume"
  echo "f12|A100|python hpcc/kitasan_black.py --model google/gemma-2-9b --layers 8 20 32 --n 20 --out hpcc/results/f12_gemma2_9b --resume"
  echo "l7|A100|python hpcc/kawakami_princess.py --model google/gemma-2-9b --out hpcc/results/l7"
}
# filter: `jobs | job_filter f2` keeps names starting with f2; `job_filter all` keeps everything
job_filter() { case "${1:-all}" in all) cat;; *) grep -E "^($(echo "$1" | tr ',' '|'))\|" ;; esac; }
[ "${BASH_SOURCE[0]}" = "$0" ] && jobs | job_filter "${1:-all}"
