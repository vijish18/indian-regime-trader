#!/usr/bin/env bash
# Start the five-process walk-forward run on the VM and return immediately.
#
#   bash deploy/azure/run_remote.sh <public-ip> [--from 2015-01-01] [--to 2026-09-15]
#
# One process per strategy. The five are independent -- each chains equity
# only through its own folds and owns its circuit-breaker state file -- which
# tests/unit/test_walk_forward.py asserts directly, so this is an exact split
# rather than an approximation.
#
# Five processes and no more: folds chain equity within a strategy and cannot
# be divided further, so the eighth vCPU is headroom, not throughput.
#
# The frame cache is set high here because the VM has 16 GB to itself and
# nothing else to serve. The whole 2,205-instrument store is about 1 GB in
# memory, so every process can hold all of it and never re-parse a file. On
# the laptop this had to be throttled to 200 frames to avoid paging; that was
# a memory constraint, not a preference.
#
# nohup rather than tmux: nothing here needs an interactive terminal, and a
# detached process with a log file survives the ssh connection dropping, which
# over a multi-hour run it will.
set -euo pipefail

IP="${1:?usage: run_remote.sh <public-ip> [extra args]}"
shift || true
ADMIN="${AZ_ADMIN:-irt}"
KEY="${AZ_KEY:-$HOME/.ssh/irt_azure}"
FROM="${IRT_FROM:-2015-01-01}"
TO="${IRT_TO:-2026-09-15}"
CACHE="${IRT_FRAME_CACHE:-2205}"

# Overridable because the right split depends on the box. One strategy
# occupies one core for the whole run -- measured at 43.5 minutes per fold,
# so about 23 hours for 32 folds -- and that is a floor no amount of extra
# hardware moves, because folds chain equity within a strategy and cannot be
# divided. So the only question is how many strategies share how many cores.
# On a 4-vCPU VM, running all five oversubscribes and stretches every one of
# them; running four, one per core, finishes in the same 23 hours as a single
# strategy would, and the fifth belongs somewhere else.
STRATEGIES="${IRT_STRATEGIES:-buy_and_hold rolling_volatility moving_average_trend hmm shuffled_regime_control}"

ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "${ADMIN}@${IP}" bash -s <<REMOTE
set -euo pipefail
cd ~/indian-regime-trader

# A previous run's artifacts would merge into this one's results.
rm -rf state/wf_json state/wf_split state/wf_series logs
mkdir -p state/wf_json state/wf_series logs

for s in $STRATEGIES; do
    mkdir -p "state/wf_split/\$s"
    nohup ./.venv/bin/python -u scripts/run_walk_forward.py \
        --from $FROM --to $TO \
        --strategy "\$s" \
        --state-dir "state/wf_split/\$s" \
        --json-out "state/wf_json/\$s.json" \
        --series-dir state/wf_series \
        --frame-cache $CACHE \
        > "logs/\$s.log" 2>&1 &
done

sleep 20
echo
echo "started:"
ps -o pid,etime,%cpu,rss,args -C python3.12 --sort=-%cpu 2>/dev/null \
  | grep -o 'strategy [a-z_]*' | sort | uniq -c || true
nproc | xargs echo "  vCPUs available:"
free -g | awk '/Mem:/ {print "  RAM total/used/free GB: " \$2 "/" \$3 "/" \$4}'
REMOTE

echo
echo "  monitor : bash deploy/azure/status_remote.sh $IP"
echo "  fetch   : bash deploy/azure/fetch_results.sh $IP"
