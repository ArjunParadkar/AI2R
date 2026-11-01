#!/usr/bin/env bash
# Pull results + logs from blob into hpcc/results + hpcc/logs, merge the F2 shards into experiments/F_gauge/results (dedup by specimen),
# and delete the VMs of finished jobs.  Safe to run repeatedly.  Usage: bash azure/collect.sh [--keep-vms]
set -euo pipefail; cd "$(dirname "$0")/.."; source azure/config.env; : "${AZ_SAS:?}"
[ -n "$AZ_SUB" ] && az account set --subscription "$AZ_SUB"
mkdir -p hpcc/results hpcc/logs azure/pull
az storage blob download-batch -d azure/pull -s "$AZ_CONTAINER" --pattern 'results/*' --account-name "$AZ_STORAGE" --account-key "$AZ_KEY" --overwrite -o none
az storage blob download-batch -d hpcc -s "$AZ_CONTAINER" --pattern 'logs/*' --account-name "$AZ_STORAGE" --account-key "$AZ_KEY" --overwrite -o none
for t in azure/pull/results/*.tgz; do tar xzf "$t" -C . ; done     # each tgz is hpcc/results/<job dirs> — overlays onto hpcc/results
python - <<'PY'
import csv, glob, os
R = "experiments/F_gauge/results"
have = {r["specimen"] for r in csv.DictReader(open(f"{R}/enumeration_status.csv"))} if os.path.exists(f"{R}/enumeration_status.csv") else set()
new = set()
for src in sorted(glob.glob("hpcc/results/f2_d3w10/shard_*/enumeration_status.csv")):
    new |= {r["specimen"] for r in csv.DictReader(open(src))} - have
for kind in ("enumeration_status", "explanations", "symmetry_f2", "enumeration_excluded"):
    dst = os.path.join(R, f"{kind}.csv"); n = 0
    for src in sorted(glob.glob(f"hpcc/results/f2_d3w10/shard_*/{kind}.csv")):
        rows = [r for r in csv.DictReader(open(src)) if r["specimen"].split(":")[0] in new]
        if not rows: continue
        fresh = not os.path.exists(dst)
        with open(dst, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            if fresh: w.writeheader()
            w.writerows(rows); n += len(rows)
    print(f"{kind}: +{n} rows")
print("merged F2 d3w10 shards:", len(new), "new specimens")
PY
if [ "${1:-}" != "--keep-vms" ]; then
  for m in $(az storage blob list -c "$AZ_CONTAINER" --prefix markers/ --account-name "$AZ_STORAGE" --account-key "$AZ_KEY" --query "[?ends_with(name,'.done')].name" -o tsv); do
    j=$(basename "$m" .done); az vm show -g "$AZ_RG" -n "wb-$j" -o none 2>/dev/null && { echo "deleting wb-$j"; az vm delete -g "$AZ_RG" -n "wb-$j" --yes -o none & }
  done; wait
fi
