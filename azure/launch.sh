#!/usr/bin/env bash
# Create one GPU VM per job.  Usage: bash azure/launch.sh [job-filter]   e.g. `f2` (all shards), `f2-s3,k7`, `all` (default)
# Idempotent: skips jobs with a done marker or an existing VM.  Needs provision.sh first and HF_TOKEN exported.
set -euo pipefail; cd "$(dirname "$0")/.."; source azure/config.env; source azure/jobs.sh
: "${AZ_SAS:?run provision.sh}" "${HF_TOKEN:?export HF_TOKEN}"
[ -n "$AZ_SUB" ] && az account set --subscription "$AZ_SUB"
WB64=$(base64 -w0 azure/worker.sh); TARBALL="$AZ_BLOB_BASE/payload/repo.tgz?$AZ_SAS"; CKPT=""
az storage blob exists -c "$AZ_CONTAINER" -n payload/ckpt.tgz --account-name "$AZ_STORAGE" --account-key "$AZ_KEY" -o tsv | grep -q True && CKPT="$AZ_BLOB_BASE/payload/ckpt.tgz?$AZ_SAS"
existing=$(az vm list -g "$AZ_RG" --query '[].name' -o tsv | tr '\n' ' ')
done_markers=$(az storage blob list -c "$AZ_CONTAINER" --prefix markers/ --account-name "$AZ_STORAGE" --account-key "$AZ_KEY" --query "[?ends_with(name,'.done')].name" -o tsv | sed 's#markers/##; s#\.done##' | tr '\n' ' ')
mkdir -p azure/logs; n=0
while IFS='|' read -r name cls cmd; do
  vm="wb-$name"
  case " $done_markers " in *" $name "*) echo "skip $name (done)"; continue;; esac
  case " $existing " in *" $vm "*) echo "skip $name (vm exists)"; continue;; esac
  [ "$cls" = A100 ] && sku=$AZ_SKU_A100 || sku=$AZ_SKU_T4
  ck=""; [ "$name" = v5 ] && ck="$CKPT"
  JOB="$name" CMD="$cmd" TARBALL_URL="$TARBALL" CKPT_URL="$ck" BLOB_BASE="$AZ_BLOB_BASE" SAS="$AZ_SAS" WORKER_B64="$WB64" \
    python3 azure/render.py azure/cloud-init.yaml > "azure/logs/$name.cloud-init.yaml"
  spot=(); [ "$AZ_SPOT" = 1 ] && spot=(--priority Spot --eviction-policy Deallocate --max-price -1)
  ( az vm create -g "$AZ_RG" -n "$vm" -l "$AZ_LOC" --size "$sku" --image "$AZ_IMAGE" --admin-username "$AZ_ADMIN" --generate-ssh-keys \
      --custom-data "azure/logs/$name.cloud-init.yaml" --os-disk-size-gb 128 --nsg-rule NONE "${spot[@]}" -o none \
    && az vm extension set -g "$AZ_RG" --vm-name "$vm" -n NvidiaGpuDriverLinux --publisher Microsoft.HpcCompute -o none \
    && echo "launched $vm ($sku)" || echo "FAILED $vm" ) > "azure/logs/$name.launch.log" 2>&1 &
  n=$((n+1)); [ $((n % 8)) -eq 0 ] && wait   # 8 creates in flight
done < <(jobs | job_filter "${1:-all}")
wait; grep -h -E "launched|FAILED" azure/logs/*.launch.log
