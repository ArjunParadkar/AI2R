#!/usr/bin/env bash
# One-time: resource group + storage + payload upload.  Usage: bash azure/provision.sh [--with-ckpt]
# Needs: az login done (or AZ_SUB + service principal in env), HF_TOKEN exported.
set -euo pipefail; cd "$(dirname "$0")/.."; source azure/config.env
[ -n "$AZ_SUB" ] && az account set --subscription "$AZ_SUB"
echo "$AZ_STORAGE" | grep -qE '^[a-z0-9]{3,24}$' || { echo "bad AZ_STORAGE"; exit 1; }
grep -q AZ_STORAGE azure/config.local.env 2>/dev/null || echo "AZ_STORAGE=$AZ_STORAGE" >> azure/config.local.env   # pin the random name
az group create -n "$AZ_RG" -l "$AZ_LOC" -o none
az storage account create -n "$AZ_STORAGE" -g "$AZ_RG" -l "$AZ_LOC" --sku Standard_LRS --allow-blob-public-access false -o none
KEY=$(az storage account keys list -n "$AZ_STORAGE" -g "$AZ_RG" --query '[0].value' -o tsv)
az storage container create -n "$AZ_CONTAINER" --account-name "$AZ_STORAGE" --account-key "$KEY" -o none
# payload: same excludes as hpcc/sync_up.sh (venvs, git, K1 cache, cluster results); the laptop's F2 status CSV is INSIDE, so shards skip done cells
EX=(--exclude '.venv*' --exclude '__pycache__' --exclude '.git' --exclude 'experiments/K_darkmatter/cache' --exclude 'hpcc/results' --exclude 'hpcc/logs' --exclude 'iso_build' --exclude 'experiments/V_legibility/ckpt' --exclude 'azure/config.local.env')
tar czf /tmp/wb-repo.tgz "${EX[@]}" . && ls -lh /tmp/wb-repo.tgz
az storage blob upload -f /tmp/wb-repo.tgz -c "$AZ_CONTAINER" -n payload/repo.tgz --account-name "$AZ_STORAGE" --account-key "$KEY" --overwrite -o none
if [ "${1:-}" = "--with-ckpt" ]; then tar czf /tmp/wb-ckpt.tgz experiments/V_legibility/ckpt && az storage blob upload -f /tmp/wb-ckpt.tgz -c "$AZ_CONTAINER" -n payload/ckpt.tgz --account-name "$AZ_STORAGE" --account-key "$KEY" --overwrite -o none; fi
EXP=$(date -u -d "+$AZ_SAS_HOURS hours" +%FT%TZ)
SAS=$(az storage container generate-sas -n "$AZ_CONTAINER" --account-name "$AZ_STORAGE" --account-key "$KEY" --permissions racwdl --expiry "$EXP" -o tsv)
cat > azure/config.local.env <<L
AZ_STORAGE=$AZ_STORAGE
AZ_KEY='$KEY'
AZ_SAS='$SAS'
AZ_BLOB_BASE=https://$AZ_STORAGE.blob.core.windows.net/$AZ_CONTAINER
L
chmod 600 azure/config.local.env; echo "provisioned: rg=$AZ_RG storage=$AZ_STORAGE sas expires $EXP (azure/config.local.env, gitignored)"
