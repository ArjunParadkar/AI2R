#!/usr/bin/env bash
# Pull cluster results back.  Usage: HPCC_HOST=user@login.host HPCC_ROOT=/scratch/$USER/whitebox-v2 bash hpcc/sync_down.sh
set -e; : "${HPCC_HOST:?}" "${HPCC_ROOT:?}"; cd "$(dirname "$0")/.."
rsync -avz --progress "$HPCC_HOST:$HPCC_ROOT/hpcc/results/" hpcc/results/; rsync -avz "$HPCC_HOST:$HPCC_ROOT/hpcc/logs/" hpcc/logs/
# the F2 d3w10 array writes one dir per shard: merge into the local F results (specimens already local are never re-run)
python - <<'PY'
import csv, glob, os
R = "experiments/F_gauge/results"; n = 0
for kind in ("enumeration_status", "explanations", "symmetry_f2"):
    dst = os.path.join(R, f"{kind}.csv"); have = set()
    if kind == "enumeration_status" and os.path.exists(dst): have = {r["specimen"] for r in csv.DictReader(open(dst))}
    for src in sorted(glob.glob(f"hpcc/results/f2_d3w10/shard_*/{kind}.csv")):
        rows = list(csv.DictReader(open(src)))
        if not rows: continue
        if kind == "enumeration_status": rows = [r for r in rows if r["specimen"] not in have]; have |= {r["specimen"] for r in rows}; n += len(rows)
        new = not os.path.exists(dst)
        with open(dst, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            if new: w.writeheader()
            w.writerows(rows)
print("merged F2 d3w10 shards:", n, "new specimens")
PY
