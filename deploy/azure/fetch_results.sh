#!/usr/bin/env bash
# Bring the run's output back, then say what to do about the VM.
#
#   bash deploy/azure/fetch_results.sh <public-ip>
#
# Results are small -- per-strategy JSON, equity curves, trade logs, vetoes
# and holdings come to a few megabytes against the 528 MB that went up -- so
# this is quick, and Azure charges nothing for the first 100 GB out per month
# in any case.
#
# This refuses to report success on a partial set. Five JSON files means five
# strategies finished; four means one was still running or was evicted, and
# merging four into a comparison table would produce a result that reads as
# the answer while missing the row it is measured against.
set -euo pipefail

IP="${1:?usage: fetch_results.sh <public-ip>}"
ADMIN="${AZ_ADMIN:-irt}"
KEY="${AZ_KEY:-$HOME/.ssh/irt_azure}"

cd "$(dirname "$0")/../.."
mkdir -p state/wf_json state/wf_series logs/azure

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "fetching results"
scp -i "$KEY" -q -o StrictHostKeyChecking=accept-new \
    "${ADMIN}@${IP}:~/indian-regime-trader/state/wf_json/*.json" state/wf_json/ 2>/dev/null \
    || echo "  no JSON yet"
scp -i "$KEY" -q "${ADMIN}@${IP}:~/indian-regime-trader/state/wf_series/*" state/wf_series/ \
    2>/dev/null || echo "  no series yet"
scp -i "$KEY" -q "${ADMIN}@${IP}:~/indian-regime-trader/logs/*.log" logs/azure/ 2>/dev/null \
    || echo "  no logs yet"

COUNT=$(ls state/wf_json/*.json 2>/dev/null | wc -l)
say "$COUNT of 5 strategies returned"
ls -la state/wf_json/ 2>/dev/null | tail -n +2 | sed 's/^/  /'

if [ "$COUNT" -lt 5 ]; then
    cat <<'EOF'

  Incomplete. Do NOT merge yet -- the comparison is only meaningful whole,
  because the HMM is judged against the baselines and the shuffled control.
  Check which are still going:

    bash deploy/azure/status_remote.sh <ip>
EOF
    exit 1
fi

say "merging"
./.venv/Scripts/python.exe scripts/merge_walk_forward.py --json-dir state/wf_json \
    || python3 scripts/merge_walk_forward.py --json-dir state/wf_json

cat <<'EOF'

Rebuild the dashboard:
  python scripts/collect_dashboard_data.py
  python scripts/build_dashboard.py

Then delete the VM. A Spot instance left running bills at INR 6/hour whether
or not anything is using it:
  az group delete --name irt-backtest --yes --no-wait
EOF
