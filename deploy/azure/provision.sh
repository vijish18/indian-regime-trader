#!/usr/bin/env bash
# Provision a throwaway Spot VM to run the walk-forward backtest.
#
#   az login                        # you must do this; it needs a browser
#   bash deploy/azure/provision.sh
#
# Measured against the Azure retail pricing API for Central India: a
# Standard_F8s_v2 Spot instance is INR 6.00/hour, so a full 32-fold run is
# about INR 33 including the OS disk. Pay-as-you-go for the same box is INR
# 32.49/hour. Spot can be evicted; that is the right trade here because the
# run splits into five independent processes, so an eviction costs only the
# strategy that was mid-flight, and re-running it is small money.
#
# Eight vCPUs and no more, deliberately. Parallelism is capped at five --
# one process per strategy -- because folds chain equity within a strategy
# and cannot be split further. A 16-vCPU box costs double and finishes at
# the same time.
#
# SECURITY
#   - SSH key only; password authentication is never enabled.
#   - The network security group opens port 22 to this machine's current
#     public IP alone, not to the internet.
#   - No broker credentials are placed on this VM. The backtest reads market
#     data from files and never contacts Zerodha, so deploy/app.env has no
#     reason to leave this laptop and does not.
#   - The repository is cloned with a read-only deploy key scoped to this one
#     repo, not with a personal access token.
set -euo pipefail

LOCATION="${AZ_LOCATION:-centralindia}"
GROUP="${AZ_GROUP:-irt-backtest}"
VM="${AZ_VM:-irt-bt-01}"
SIZE="${AZ_SIZE:-Standard_F8s_v2}"
IMAGE="${AZ_IMAGE:-Ubuntu2404}"
ADMIN="${AZ_ADMIN:-irt}"
KEY="${AZ_KEY:-$HOME/.ssh/irt_azure}"
DISK_GB="${AZ_DISK_GB:-64}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

command -v az >/dev/null || { echo "az CLI not found"; exit 1; }
az account show >/dev/null 2>&1 || { echo "run 'az login' first"; exit 1; }

SUB=$(az account show --query name -o tsv)
say "subscription: $SUB"

if [ ! -f "$KEY" ]; then
    say "generating SSH key $KEY"
    ssh-keygen -t ed25519 -f "$KEY" -N "" -C "irt-azure-backtest"
fi

MY_IP=$(curl -fsS https://api.ipify.org)
say "locking SSH to $MY_IP/32"

say "resource group $GROUP in $LOCATION"
az group create --name "$GROUP" --location "$LOCATION" --output none

say "creating $SIZE Spot VM $VM (this takes a couple of minutes)"
az vm create \
    --resource-group "$GROUP" \
    --name "$VM" \
    --image "$IMAGE" \
    --size "$SIZE" \
    --admin-username "$ADMIN" \
    --ssh-key-values "${KEY}.pub" \
    --authentication-type ssh \
    --priority Spot \
    --max-price -1 \
    --eviction-policy Delete \
    --os-disk-size-gb "$DISK_GB" \
    --storage-sku StandardSSD_LRS \
    --nsg-rule NONE \
    --public-ip-sku Standard \
    --output none

say "opening port 22 to this machine only"
NSG=$(az network nsg list --resource-group "$GROUP" --query "[0].name" -o tsv)
az network nsg rule create \
    --resource-group "$GROUP" \
    --nsg-name "$NSG" \
    --name ssh-from-operator \
    --priority 1000 \
    --source-address-prefixes "${MY_IP}/32" \
    --destination-port-ranges 22 \
    --access Allow --protocol Tcp \
    --output none

IP=$(az vm show -d --resource-group "$GROUP" --name "$VM" --query publicIps -o tsv)
say "VM ready at $IP"

cat <<EOF

  ssh -i $KEY ${ADMIN}@${IP}

Next:
  bash deploy/azure/setup_remote.sh $IP      # install python, clone repo
  bash deploy/azure/push_data.sh   $IP       # upload data_cache (528 MB)
  bash deploy/azure/run_remote.sh  $IP       # start the 5-process backtest

When the run is done and results are fetched:
  az group delete --name $GROUP --yes --no-wait

That delete is the whole cost control. A Spot VM left running is INR 6/hour
whether or not anything is using it.
EOF
