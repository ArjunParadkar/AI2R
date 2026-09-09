#!/usr/bin/env bash
# CPU queue: waits for the F2 enumeration (CPU + d3w8 GPU cells) to finish, then the F4 reformulated tests,
# aggregation, and the paused CPU jobs.
set -u
V=/home/god/Whitebox/whitebox-v2/.venv/bin/python
E=/home/god/Whitebox/whitebox-v2/experiments
stage() { echo "STAGE $1 $(date '+%F %T')"; }
# a stage with logs/<name>.done is skipped (idempotent relaunch after power loss — AMENDMENTS #10)
run() { local name=$1; shift; if [ -e "$E/../logs/$name.done" ]; then stage "SKIP $name (done marker)"; return 0; fi
        stage "START $name"; ( "$@" ) >> "$E/../logs/$name.log" 2>&1; local rc=$?; echo "STAGE END $name rc=$rc $(date '+%F %T')"
        [ $rc -eq 0 ] && touch "$E/../logs/$name.done"; return $rc; }
# gate on completeness on disk, not only on unit liveness (a failed+relaunched unit let the queue start early once — AMENDMENTS #9)
# AMENDMENTS #11: the F2 tail unit runs under an 8 h wall budget (RuntimeMaxSec); if it ends with specimens still
# missing they are recorded in results/enumeration_excluded.csv (reason = wall budget) and the queue proceeds.
while systemctl --user is-active --quiet wb-rice-shower || systemctl --user is-active --quiet wb-rice-shower-gpu; do sleep 120; done
if ! $V $E/../wb_ready.py f2; then $V $E/../wb_ready.py f2 --exclude-remaining "wall budget 8h exceeded after 3 power-loss restarts (AMENDMENTS #11)"; fi
while ! $V $E/../wb_ready.py f2; do sleep 120; done
stage "F2 enumeration finished"
cd $E/F_gauge && run grass_wonder        $V -u grass_wonder.py
cd $E/F_gauge && run el_condor_pasa_kills $V -u el_condor_pasa.py --kills
cd $E/F_gauge && run el_condor_pasa       $V -u el_condor_pasa.py --procs 8 --max-pairs 20000
cd $E/F_gauge && run air_groove       $V -u air_groove.py
cd $E/F_gauge && run hishi_amazon       $V -u hishi_amazon.py
cd $E/F_gauge && run grass_wonder_aggregate   $V -u grass_wonder_aggregate.py
cd $E/F_gauge && run grass_wonder_aggregate_f3  $V -u grass_wonder_aggregate.py f3
cd $E/F_gauge && run special_week    $V -u special_week.py --procs 8
cd $E/F_gauge && run special_week_summary $V -u special_week_summary.py
cd $E/D_attribution && run sakura_bakushin_o  $V -u sakura_bakushin_o.py
cd $E/M_memorization && run haru_urara $V -u haru_urara.py 6
cd $E/Z_topology && run tokai_teio     $V -u tokai_teio.py
cd $E/Z_topology && run tokai_teio     $V -u tokai_teio.py z5
stage "CPU QUEUE DONE"
