#!/usr/bin/env bash
# Submit every WHITEBOX HPCC job in dependency order. Run from the whitebox-v2 checkout root on the cluster AFTER:
#   source hpcc/env.sh; bash hpcc/configure.sh <partition> [account] [gres]; bash hpcc/smoke.sh   (all PASS)
# Usage: bash hpcc/launch_all.sh [--no-gated]   (--no-gated skips every job that needs HF_TOKEN with Gemma/Llama licences)
set -euo pipefail
cd "$(dirname "$0")/.."; GATED=1; [ "${1:-}" = "--no-gated" ] && GATED=0
sub() { local id; id=$(sbatch --parsable "$@"); echo "submitted $* -> $id"; echo "$id"; }
declare -A J
J[f2]=$(sub hpcc/sbatch/f2_d3w10_enumeration.sbatch)
J[m4]=$(sub hpcc/sbatch/m4_cross_family.sbatch)
J[v1]=$(sub hpcc/sbatch/v1_ensemble_large.sbatch)
J[v7]=$(sub hpcc/sbatch/v7_pythia_drift.sbatch)
J[b5]=$(sub hpcc/sbatch/b5_family_calibration.sbatch)
[ -f hpcc/sbatch/o31_pythia_open.sbatch ] && J[o31]=$(sub hpcc/sbatch/o31_pythia_open.sbatch)
[ -f hpcc/sbatch/o2_gpt2_open.sbatch ]   && J[o2]=$(sub hpcc/sbatch/o2_gpt2_open.sbatch)
if [ "$GATED" = 1 ]; then
  J[f12]=$(sub hpcc/sbatch/f12_frontier_audit.sbatch)
  J[k8]=$(sub hpcc/sbatch/k8_frontier_residue.sbatch)
  J[k7]=$(sub hpcc/sbatch/k7_width_scaling.sbatch)
  J[c4]=$(sub hpcc/sbatch/c4_scale.sbatch)
  J[l7]=$(sub hpcc/sbatch/l7_end_to_end.sbatch)
  [ -f hpcc/sbatch/o41_frontier_p0.sbatch ] && J[o41p0]=$(sub hpcc/sbatch/o41_frontier_p0.sbatch)
  if [ -f hpcc/sbatch/o41_assemble.sbatch ]; then
    DEP="afterany:${J[f12]}:${J[k8]}:${J[c4]}:${J[l7]}"; [ -n "${J[o41p0]:-}" ] && DEP="$DEP:${J[o41p0]}"
    J[o41]=$(sub --dependency="$DEP" hpcc/sbatch/o41_assemble.sbatch)
  fi
fi
echo; echo "job ids:"; for k in "${!J[@]}"; do echo "  $k = ${J[$k]}"; done; echo; squeue -u "$USER"
echo "when everything is done: bash hpcc/sync_down.sh on the laptop"
