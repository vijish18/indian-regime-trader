#!/usr/bin/env bash
# What the run is doing right now.
#
#   bash deploy/azure/status_remote.sh <public-ip>
#
# Reports fold progress per strategy, CPU per process, and -- the reason this
# exists as more than a tail -- whether any process has died. On Spot capacity
# a process can vanish because Azure reclaimed the instance, and a strategy
# that is simply absent looks exactly like one that finished quietly.
set -euo pipefail

IP="${1:?usage: status_remote.sh <public-ip>}"
ADMIN="${AZ_ADMIN:-irt}"
KEY="${AZ_KEY:-$HOME/.ssh/irt_azure}"

ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "${ADMIN}@${IP}" bash -s <<'REMOTE'
set -uo pipefail
cd ~/indian-regime-trader

printf '%-26s %6s %9s %s\n' strategy folds running last
for s in buy_and_hold rolling_volatility moving_average_trend hmm shuffled_regime_control; do
    log="logs/$s.log"
    folds=$(grep -cE '^\[' "$log" 2>/dev/null || echo 0)
    alive=$(pgrep -fc -- "--strategy $s" 2>/dev/null || echo 0)
    done_json="no"
    [ -f "state/wf_json/$s.json" ] && done_json="JSON"
    last=$(grep -E '^\[' "$log" 2>/dev/null | tail -1 | cut -c1-58)
    state="yes"
    [ "$alive" -eq 0 ] && state="NO($done_json)"
    printf '%-26s %6s %9s %s\n' "$s" "$folds" "$state" "$last"
done

echo
if grep -lE 'Traceback|Error:' logs/*.log 2>/dev/null | grep -q .; then
    echo "ERRORS in:"
    grep -lE 'Traceback|Error:' logs/*.log | sed 's/^/  /'
else
    echo "no tracebacks"
fi
echo
uptime | sed 's/^/  /'
free -g | awk '/Mem:/ {print "  RAM GB used/total: " $3 "/" $2}'
REMOTE
