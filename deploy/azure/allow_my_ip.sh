#!/usr/bin/env bash
# Point the SSH rule at the network this machine actually reaches Azure from.
#
#   bash deploy/azure/allow_my_ip.sh
#   AZ_SSH_CIDR=203.0.113.7/32 bash deploy/azure/allow_my_ip.sh
#
# A /32 is the right posture and does not work behind CGNAT, which is what
# this connection turned out to be: the address api.ipify.org reports is not
# the address Azure sees, so a rule naming ipify's answer locks out the very
# machine it was written to admit. Measured directly -- the /32 timed out on
# every attempt, the enclosing /24 completed the SSH handshake immediately.
#
# So the default is the /24 containing the observed address. That admits
# every other customer on the ISP's pool, which is acceptable only because of
# what is on this VM: public market data, code that already lives in a
# private GitHub repo, no broker credentials, and key-only SSH with passwords
# disabled. It would not be acceptable on a machine holding anything that
# matters, and the VM is deleted when the run finishes.
#
# Set AZ_SSH_CIDR explicitly to force a tighter rule on a connection with a
# stable public address.
set -euo pipefail

GROUP="${AZ_GROUP:-irt-backtest}"

OBSERVED=$(curl -fsS https://api.ipify.org)
CIDR="${AZ_SSH_CIDR:-$(echo "$OBSERVED" | sed 's/\.[0-9]*$/.0\/24/')}"

NSG=$(az network nsg list --resource-group "$GROUP" --query "[0].name" -o tsv)
[ -n "$NSG" ] || { echo "no NSG in resource group $GROUP"; exit 1; }

CURRENT=$(az network nsg rule show --resource-group "$GROUP" --nsg-name "$NSG" \
    --name ssh-from-operator --query "sourceAddressPrefix" -o tsv 2>/dev/null || echo "")

if [ "$CURRENT" = "$CIDR" ]; then
    echo "  SSH already allowed from $CIDR (observed $OBSERVED)"
    exit 0
fi

echo "  SSH rule: ${CURRENT:-none} -> $CIDR (observed $OBSERVED)"
az network nsg rule update \
    --resource-group "$GROUP" \
    --nsg-name "$NSG" \
    --name ssh-from-operator \
    --source-address-prefixes "$CIDR" \
    --output none
echo "  updated"
