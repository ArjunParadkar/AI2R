#!/usr/bin/env bash
# AC-aware queue supervisor.  usage: scripts/supervisor.sh <name> <grid.jsonl> <outdir> <nworkers>
# Runs <nworkers> resumable workers only while on AC power (GPU is throttled to ~390 MHz on battery and the
# machine would die).  On unplug -> workers stopped; on plug-in -> restarted; finished configs are never redone.
# Stops itself when every cfg in the grid has a result (or .err).
name=$1; grid=$2; out=$3; nw=${4:-4}
cd "$(dirname "$0")/.."
ac() { [ "$(cat /sys/class/power_supply/ACAD/online 2>/dev/null)" = 1 ]; }
todo() { .venv/bin/python - "$grid" "$out" <<'PY'
import sys,json,os
g,o=sys.argv[1:3]; n=0
for l in open(g):
    if l.strip():
        i=json.loads(l)['id']
        if not (os.path.exists(f'{o}/{i}.json') or os.path.exists(f'{o}/{i}.err')): n+=1
print(n)
PY
}
while true; do
  left=$(todo)
  if [ "$left" = 0 ]; then echo "$(date) done"; exit 0; fi
  if ac; then
    for s in $(seq 0 $((nw-1))); do
      systemctl --user is-active -q wb-$name-w$s || scripts/wbrun $name-w$s .venv/bin/python -u bench/worker.py "$grid" $s $nw "$out" >/dev/null 2>&1
    done
  else
    for s in $(seq 0 $((nw-1))); do systemctl --user stop wb-$name-w$s 2>/dev/null; done
  fi
  sleep 30
done
