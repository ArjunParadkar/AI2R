#!/usr/bin/env bash
# Sequential GPU queue for the sprint (one GPU job at a time; the F2 d3w8 enumeration may still share the GPU).
set -u
V=/home/god/Whitebox/whitebox-v2/.venv/bin/python
TL=/home/god/Whitebox/v2-programme/.venv-tl/bin/python
E=/home/god/Whitebox/whitebox-v2/experiments
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stage() { echo "STAGE $1 $(date '+%F %T')"; }
# a stage with logs/<name>.done is skipped (idempotent relaunch after power loss — AMENDMENTS #10)
run() { local name=$1; shift; if [ -e "$E/../logs/$name.done" ]; then stage "SKIP $name (done marker)"; return 0; fi
        stage "START $name"; ( "$@" ) >> "$E/../logs/$name.log" 2>&1; local rc=$?; echo "STAGE END $name rc=$rc $(date '+%F %T')"
        [ $rc -eq 0 ] && touch "$E/../logs/$name.done"; return $rc; }
# wait for F7
while systemctl --user is-active --quiet wb-f7b || systemctl --user is-active --quiet wb-f2-gpu24c || ! $V $E/../wb_ready.py f7 || ! $V $E/../wb_ready.py f2gpu; do sleep 60; done
stage "F7 and F2 d3w8 finished"
cd $E/F_gauge      && run F11     $V -u run_F11.py --n 30
cd $E/K_darkmatter && run K1      $TL -u run_K1.py
cd $E/K_darkmatter && run K2_K6   $V -u run_K2_K6.py
# B2_B4 may be running standalone (wb-b2b4, 2026-09-08 rerun): wait for it, then the .done marker skips the stage
while systemctl --user is-active --quiet wb-b2b4; do sleep 60; done
cd $E/B_bias       && run B2_B4   $V -u run_B2_B4.py
# V1 in three disjoint shards: shard 0 here, shards 1-2 as units wb-v1-s1 / wb-v1-s2 (launched with the queue); wait for all
cd $E/V_legibility && run V1      $V -u run_V1.py --shard 0 3
while systemctl --user is-active --quiet wb-v1-s1 || systemctl --user is-active --quiet wb-v1-s2 || ! $V $E/../wb_ready.py v1; do sleep 120; done
stage "V1 all shards finished"
cd $E/V_legibility && run V2_V3   $V -u run_V2_V3.py
cd $E/M_memorization && run M2    $V -u run_M2_M3.py
cd $E/M_memorization && run M3    $V -u run_M2_M3.py --m3
cd $E/C_concepts   && run C1_C3   $V -u run_C1_C3.py
cd $E/Z_topology   && run Z3      $V -u run_Z1_Z3.py z3
cd $E/Z_topology   && run Z1      $V -u run_Z1_Z3.py z1
cd $E/Z_topology   && run Z2      $V -u run_Z1_Z3.py z2
cd $E/V_legibility && run V5      $V -u run_V2_V3.py --all-ckpts
cd $E/F_gauge      && run F2_d3w10 $V -u run_F2.py --gpu --min-hidden 30 --max-hidden 30 --procs 4
stage "QUEUE DONE"
