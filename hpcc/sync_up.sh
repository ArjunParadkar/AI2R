#!/usr/bin/env bash
# Push the sprint checkout to the cluster.  Usage: HPCC_HOST=user@login.host HPCC_ROOT=/scratch/$USER/whitebox-v2 bash hpcc/sync_up.sh [--with-ckpt]
# Excludes venvs, caches and the 1.5 GB K1 cache (K8 regenerates it). --with-ckpt adds the 2.2 GB V1 checkpoints (only for v5_drift_rest).
set -e; : "${HPCC_HOST:?}" "${HPCC_ROOT:?}"; cd "$(dirname "$0")/.."
EX=(--exclude '.venv*' --exclude '__pycache__' --exclude '.git' --exclude 'experiments/K_darkmatter/cache' --exclude 'hpcc/results' --exclude 'hpcc/logs' --exclude 'iso_build' --exclude 'experiments/O_opener/results/ckpt' --exclude 'experiments/O_opener/results/open/*_clouds.npz')
[ "${1:-}" = "--with-ckpt" ] || EX+=(--exclude 'experiments/V_legibility/ckpt')
ssh "$HPCC_HOST" "mkdir -p '$HPCC_ROOT'"
rsync -avz --progress "${EX[@]}" ./ "$HPCC_HOST:$HPCC_ROOT/"
echo "synced to $HPCC_HOST:$HPCC_ROOT (batteries under benchmarks/results/battery_* and F specimens under experiments/F_gauge/results/networks included)"
