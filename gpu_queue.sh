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
while systemctl --user is-active --quiet wb-fuji-kiseki || systemctl --user is-active --quiet wb-rice-shower-gpu || ! $V $E/../wb_ready.py fuji_kiseki || ! $V $E/../wb_ready.py mejiro_mcqueen; do sleep 60; done
stage "F7 and F2 d3w8 finished"
cd $E/F_gauge      && run narita_brian     $V -u narita_brian.py --n 30
cd $E/K_darkmatter && run eishin_flash      $TL -u eishin_flash.py
cd $E/K_darkmatter && run smart_falcon   $V -u smart_falcon.py
# B2_B4 may be running standalone (wb-winning-ticket, 2026-09-08 rerun): wait for it, then the .done marker skips the stage
while systemctl --user is-active --quiet wb-winning-ticket; do sleep 60; done
cd $E/B_bias       && run winning_ticket   $V -u winning_ticket.py
# V1 in three disjoint shards: shard 0 here, shards 1-2 as units wb-mejiro-dober-s1 / wb-mejiro-dober-s2 (launched with the queue); wait for all
# (all three shards run as units wb-mejiro-dober-s0/1/2 so they start before B2_B4 finishes; the queue only waits)
while systemctl --user is-active --quiet wb-mejiro-dober-s0 || systemctl --user is-active --quiet wb-mejiro-dober-s1 || systemctl --user is-active --quiet wb-mejiro-dober-s2 || ! $V $E/../wb_ready.py mejiro_dober; do sleep 120; done
stage "V1 all shards finished"
cd $E/V_legibility && run twin_turbo   $V -u twin_turbo.py
cd $E/M_memorization && run vodka    $V -u vodka.py
cd $E/M_memorization && run daiwa_scarlet    $V -u vodka.py --daiwa-scarlet
cd $E/C_concepts   && run gold_ship   $V -u gold_ship.py
cd $E/Z_topology   && run agnes_tachyon      $V -u agnes_tachyon.py agnes_tachyon
cd $E/Z_topology   && run seiun_sky      $V -u agnes_tachyon.py seiun_sky
cd $E/Z_topology   && run manhattan_cafe      $V -u agnes_tachyon.py manhattan_cafe
cd $E/V_legibility && run silence_suzuka      $V -u twin_turbo.py --all-ckpts
cd $E/F_gauge      && run mejiro_mcqueen $V -u rice_shower.py --gpu --min-hidden 30 --max-hidden 30 --procs 4
stage "QUEUE DONE"
