#!/usr/bin/env bash
# Delete EVERYTHING in the resource group (VMs, disks, storage incl. results not yet collected).  Run collect.sh first.
set -euo pipefail; cd "$(dirname "$0")/.."; source azure/config.env
[ -n "$AZ_SUB" ] && az account set --subscription "$AZ_SUB"
read -r -p "delete resource group $AZ_RG and all results still in blob? [type yes] " a; [ "$a" = yes ] || exit 1
az group delete -n "$AZ_RG" --yes --no-wait; echo "deleting $AZ_RG in the background"
