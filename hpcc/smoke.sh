#!/usr/bin/env bash
# One tiny unit of work per cluster job, on an interactive GPU node (e.g. `srun --gres=gpu:1 --mem=64G --time=02:00:00 --pty bash`),
# from the checkout root after `source hpcc/env.sh`.  Prints PASS/FAIL per job to hpcc/logs/smoke.log.  Skip nothing: a job that
# fails here would fail an hour into its sbatch.  Order = cheapest first.
cd "$(dirname "$0")/.."; L=hpcc/logs/smoke.log; : > $L
run() { local name=$1; shift; echo "== $name: $*" | tee -a $L; if timeout 3600 "$@" >> hpcc/logs/smoke_$name.log 2>&1; then echo "PASS $name" | tee -a $L; else echo "FAIL $name (see hpcc/logs/smoke_$name.log)" | tee -a $L; fi; }
run m4   python hpcc/admire_vega.py --out hpcc/results/smoke/m4                                           # cached CNN + LM batteries, no ViT
run b5   python hpcc/mejiro_ryan.py --models gpt2 --ns 50 --deltas 0 0.5 --trials 2 --out hpcc/results/smoke/b5
run v7   python hpcc/mejiro_ardan.py --model EleutherAI/pythia-410m --revisions step0 step143000 --n-seq 4 --out hpcc/results/smoke/v7
run f2   bash -c 'cd experiments/F_gauge && python rice_shower.py --gpu --min-hidden 30 --max-hidden 30 --procs 2 --shard 0 449 --out ../../hpcc/results/smoke/f2'
run k7   python hpcc/mayano_top_gun.py --model google/gemma-2-2b --widths 16k --n-seq 8 --out hpcc/results/smoke/k7
run f12  python hpcc/kitasan_black.py --model google/gemma-2-2b --layers 8 --n 2 --n-tokens 250 --out hpcc/results/smoke/f12
run k8   python hpcc/biwa_hayahide.py --model google/gemma-2-9b --layers 20 --n-seq 8 --out hpcc/results/smoke/k8
run c4   python hpcc/sirius_symboli.py --model google/gemma-2-9b --n-seq 8 --out hpcc/results/smoke/c4
run l7   python hpcc/kawakami_princess.py --model google/gemma-2-9b --n 2 --out hpcc/results/smoke/l7
echo "== v1_large has no smoke flag: submit the array and check hpcc/logs/wb-v1_ensemble_large-*_0.out after 10 min" | tee -a $L
cat $L
