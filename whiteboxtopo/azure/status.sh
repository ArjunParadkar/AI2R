#!/usr/bin/env bash
# Job board: marker state per job + VM power state; `bash azure/status.sh log k7` tails a job's log from blob.
set -euo pipefail; cd "$(dirname "$0")/.."; source azure/config.env; source azure/jobs.sh; : "${AZ_SAS:?}"
[ -n "$AZ_SUB" ] && az account set --subscription "$AZ_SUB"
if [ "${1:-}" = log ]; then curl -sS "$AZ_BLOB_BASE/logs/$2.log?$AZ_SAS" | tail -n "${3:-40}"; exit; fi
markers=$(az storage blob list -c "$AZ_CONTAINER" --prefix markers/ --account-name "$AZ_STORAGE" --account-key "$AZ_KEY" --query '[].name' -o tsv | sed 's#markers/##')
vms=$(az vm list -g "$AZ_RG" -d --query '[].[name,powerState]' -o tsv 2>/dev/null || true)
printf "%-8s %-9s %s\n" JOB STATE VM
while IFS='|' read -r name cls cmd; do
  st=pending; grep -qx "$name.started" <<<"$markers" && st=running; grep -qx "$name.failed" <<<"$markers" && st=FAILED; grep -qx "$name.done" <<<"$markers" && st=done
  vm=$(grep -P "^wb-$name\t" <<<"$vms" | cut -f2 || true); printf "%-8s %-9s %s\n" "$name" "$st" "${vm:--}"
done < <(jobs | job_filter "${2:-all}")
echo; echo "done: $(grep -c '\.done$' <<<"$markers" || true)  failed: $(grep -c '\.failed$' <<<"$markers" || true)  vms: $(wc -l <<<"$vms")"
