#!/usr/bin/env bash
# Fill the cluster-specific placeholders in every hpcc/sbatch/*.sbatch once (run on the cluster or before syncing):
#   bash hpcc/configure.sh <partition> [account] [gres]      e.g.  bash hpcc/configure.sh gpu72 my_alloc gpu:a100:1
set -e; P=${1:?partition}; A=${2:-}; G=${3:-gpu:1}
for f in "$(dirname "$0")"/sbatch/*.sbatch; do
  sed -i "s/__PARTITION__/$P/; s|__GRES__|$G|" "$f"
  if [ -n "$A" ]; then sed -i "s/__ACCOUNT__/$A/" "$f"; else sed -i '/__ACCOUNT__/d' "$f"; fi
done; echo "configured: partition=$P account=${A:-none} gres=$G"
