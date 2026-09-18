#!/usr/bin/env bash
# Install the runtime on the VM and clone the repository.
#
#   bash deploy/azure/setup_remote.sh <public-ip>
#
# The repository is private, so it is cloned with a READ-ONLY deploy key
# scoped to this one repo. A personal access token would also work and is
# what most guides reach for, but a PAT carries every permission the account
# has to every repo it owns -- on a throwaway Spot VM that can be evicted and
# recycled, that is the wrong thing to leave lying on disk.
#
# Ubuntu 24.04 ships Python 3.12, which satisfies the project's >=3.11.
#
# Note on reproducibility: this is Linux and the laptop is Windows. Identical
# library versions still leave BLAS free to reassociate floating-point
# operations differently, so figures can differ in the last decimals. That
# does not move a CAGR or a Sharpe enough to change a conclusion, but it does
# mean the two runs are not expected to be bit-identical.
set -euo pipefail

IP="${1:?usage: setup_remote.sh <public-ip>}"
ADMIN="${AZ_ADMIN:-irt}"
KEY="${AZ_KEY:-$HOME/.ssh/irt_azure}"
REPO="${IRT_REPO:-vijish18/indian-regime-trader}"
DEPLOY_KEY="${DEPLOY_KEY:-$HOME/.ssh/irt_deploy}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
remote() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "${ADMIN}@${IP}" "$@"; }

if [ ! -f "$DEPLOY_KEY" ]; then
    say "generating read-only deploy key"
    ssh-keygen -t ed25519 -f "$DEPLOY_KEY" -N "" -C "irt-azure-deploy"
    say "registering it with GitHub (read-only)"
    gh repo deploy-key add "${DEPLOY_KEY}.pub" \
        --repo "$REPO" --title "azure-backtest-$(date +%Y%m%d)" \
        || echo "  (add ${DEPLOY_KEY}.pub as a deploy key manually if this failed)"
fi

say "waiting for sshd"
for _ in $(seq 1 30); do
    remote true 2>/dev/null && break
    sleep 5
done

say "installing python and tooling"
remote 'sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq &&
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
            python3.12 python3.12-venv python3-pip git tmux >/dev/null &&
        python3.12 --version'

say "installing the deploy key on the VM"
scp -i "$KEY" -q "$DEPLOY_KEY" "${ADMIN}@${IP}:~/.ssh/id_ed25519"
remote 'chmod 600 ~/.ssh/id_ed25519 &&
        ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null'

say "cloning $REPO"
remote "rm -rf ~/indian-regime-trader &&
        git clone --quiet git@github.com:${REPO}.git ~/indian-regime-trader &&
        cd ~/indian-regime-trader && git log --oneline -1"

say "building the virtualenv"
remote 'cd ~/indian-regime-trader &&
        python3.12 -m venv .venv &&
        ./.venv/bin/pip install --quiet --upgrade pip &&
        ./.venv/bin/pip install --quiet -e . &&
        ./.venv/bin/python -c "import pandas, numpy; print(\"pandas\", pandas.__version__, \"numpy\", numpy.__version__)"'

say "setup complete"
echo "  next: bash deploy/azure/push_data.sh $IP"
