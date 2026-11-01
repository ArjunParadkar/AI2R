#!/usr/bin/env bash
# One tiny unit of work per cluster job, on an interactive GPU node (e.g. `srun --gres=gpu:1 --mem=64G --time=02:00:00 --pty bash`),
# from the checkout root after `source hpcc/env.sh`.  Prints PASS/FAIL per job to hpcc/logs/smoke.log.  Skip nothing: a job that
# fails here would fail an hour into its sbatch.  Order = cheapest first.
cd "$(dirname "$0")/.."; L=hpcc/logs/smoke.log; : > $L
run() { local name=$1; shift; echo "== $name: $*" | tee -a $L; if timeout 3600 "$@" >> hpcc/logs/smoke_$name.log 2>&1; then echo "PASS $name" | tee -a $L; else echo "FAIL $name (see hpcc/logs/smoke_$name.log)" | tee -a $L; fi; }
run m4   python hpcc/admire_vega.py --out hpcc/results/smoke/m4                                           # cached CNN + LM batteries, no ViT
run b5   python hpcc/mejiro_ryan.py --models gpt2 --ns 50 --deltas 0 0.5 --trials 2 --out hpcc/results/smoke/b5
run v7   python hpcc/mejiro_ardan.py --model EleutherAI/pythia-410m --revisions step0 step143000 --n-seq 4 --out hpcc/results/smoke/v7   # add --per-node for the V AMENDMENTS #15 repair (default OFF: the V7 rows are pre-registered with the per-layer gradient)
run o31  python experiments/O_opener/hishi_akebono.py --model EleutherAI/pythia-70m --smoke --out hpcc/results/smoke/o31   # 3 revisions, tiny probe: must write scorecard_pythia-70m.json
run f2   bash -c 'cd experiments/F_gauge && python rice_shower.py --gpu --min-hidden 30 --max-hidden 30 --procs 2 --shard 0 449 --out ../../hpcc/results/smoke/f2'
run k7   python hpcc/mayano_top_gun.py --model google/gemma-2-2b --widths 16k --n-seq 8 --out hpcc/results/smoke/k7
run f12  python hpcc/kitasan_black.py --model google/gemma-2-2b --layers 8 --n 2 --n-tokens 250 --out hpcc/results/smoke/f12
run k8   python hpcc/biwa_hayahide.py --model google/gemma-2-9b --layers 20 --n-seq 8 --out hpcc/results/smoke/k8
run c4   python hpcc/sirius_symboli.py --model google/gemma-2-9b --n-seq 8 --out hpcc/results/smoke/c4
run l7   python hpcc/kawakami_princess.py --model google/gemma-2-9b --n 2 --out hpcc/results/smoke/l7
run o2   python experiments/O_opener/satono_crown.py --smoke --n-gauge 2          # O2.1/O2.2 GPT-2 open(): 3 tasks end to end, tiny splits
run o41p0  python experiments/O_opener/bamboo_memory.py --p0 --model google/gemma-2-9b --layers 20 21 --n-tokens 250 --out hpcc/results/smoke/o41   # builds BOTH the pre-registered and the per-target-node EAP graph (#O4.1 (d)) + a 3-sample K-O-2 orbit on a 4-sequence probe; --no-per-node / --n-gauge 0 skip them, the P0 verdict is unaffected either way
# the assemble smoke deliberately runs LAST and points at hpcc/results/smoke: it then reads the f12/c4/k8/l7/o41 dirs the
# lines above just wrote, which is the real read path.  It must PASS even when some of them failed — every missing input is
# a not_reached cell, never an exception; check hpcc/results/smoke/o41/scorecard_gemma_2_9b.md for the reasons.
run o41asm python experiments/O_opener/bamboo_memory.py --assemble --model google/gemma-2-9b --results hpcc/results/smoke --out hpcc/results/smoke/o41
echo "== v1_large has no smoke flag: submit the array and check hpcc/logs/wb-v1_ensemble_large-*_0.out after 10 min" | tee -a $L
# o31 (O3.1 Hishi Akebono) is one unit of work in the same sense as the others: the smallest public Pythia, three
# revisions (step0 / step1000 / step143000), a 4-sequence probe and reduced nulls, run end to end through the same code
# path the cluster job uses — basis freeze, per-revision trajectory row, cloud cache, zigzag, change points, Scorecard.
# PASS means hpcc/results/smoke/o31/scorecard_pythia-70m.json exists with P3 and P7 carrying values and the other six
# cells `not_reached` with reasons.  --smoke also caps K-O-2 to --n-gauge 3 / --orbit-every 2, so the gauge-orbit
# path runs too (SPEC_O3 AMENDMENT #O3-1): on the laptop this made the smoke 226 s instead of 38 s, and it returned
# P3 `certified`, P7 `failed (artifact)` (the EAP legibility statistic moves under the gauge).  A `failed` cell here
# is a RESULT, not a smoke failure — only a traceback is. At three revisions the change-point rule is undefined (T < 6) and every trace reports
# p = 1.0 with that note: that is the correct smoke output, not a failure. It does NOT exercise --purge (which the real
# job needs for its 246 GB / 616 GB of checkpoints) — purge on one revision by hand once, before submitting:
#   python -c "import sys;sys.path.insert(0,'experiments/O_opener');from hishi_akebono import purge_revision;print(purge_revision('EleutherAI/pythia-70m','step1000'))"
# and confirm the snapshot is gone from $HF_HOME/hub. Laptop reference: 76 s wall for the smoke including downloads;
# 18.6-27.4 s/revision of compute at full settings on pythia-410m with the weights already on disk.
cat $L
