#!/usr/bin/env bash
# CPU queue: waits for the F2 enumeration (CPU + d3w8 GPU cells) to finish, then the F4 reformulated tests,
# aggregation, and the paused CPU jobs.
set -u
V=/home/god/Whitebox/whitebox-v2/.venv/bin/python
E=/home/god/Whitebox/whitebox-v2/experiments
stage() { echo "STAGE $1 $(date '+%F %T')"; }
run() { local name=$1; shift; stage "START $name"; ( "$@" ) >> "$E/../logs/$name.log" 2>&1; echo "STAGE END $name rc=$? $(date '+%F %T')"; }
# gate on completeness on disk, not only on unit liveness (a failed+relaunched unit let the queue start early once — AMENDMENTS #9)
while systemctl --user is-active --quiet wb-f2-cpu2 || systemctl --user is-active --quiet wb-f2-gpu24c || ! $V $E/../wb_ready.py f2; do sleep 120; done
stage "F2 enumeration finished"
cd $E/F_gauge && run F3        $V -u run_F3.py
cd $E/F_gauge && run F4a_kills $V -u run_F4a.py --kills
cd $E/F_gauge && run F4a       $V -u run_F4a.py --procs 8 --max-pairs 20000
cd $E/F_gauge && run F4b       $V -u run_F4b.py
cd $E/F_gauge && run F4c       $V -u run_F4c.py
cd $E/F_gauge && run aggregate_F   $V -u aggregate_F.py
cd $E/F_gauge && run aggregate_F3  $V -u aggregate_F.py f3
cd $E/F_gauge && run F8_F10    $V -u run_F8_F10.py --procs 8
cd $E/F_gauge && run F10_summary $V -u summarize_F10.py
cd $E/D_attribution && run D1  $V -u run_D1.py
cd $E/M_memorization && run M1 $V -u run_M1.py 6
cd $E/Z_topology && run Z4     $V -u run_Z4_Z5.py
cd $E/Z_topology && run Z5     $V -u run_Z4_Z5.py z5
stage "CPU QUEUE DONE"
