#!/usr/bin/env bash
# Ship the market data the backtest reads.
#
#   bash deploy/azure/push_data.sh <public-ip>
#
# data_cache/ is gitignored -- 528 MB of bhavcopy archives, adjusted bars and
# reference tables -- so it does not travel with the clone and has to be sent
# separately.
#
# Sent as a compressed stream over ssh rather than with rsync, which Git Bash
# on Windows does not ship. These are CSVs, so they compress to roughly a
# third and the transfer is bounded by upload bandwidth, not by the disk.
#
# Regenerating the data on the VM instead was the alternative and is worse:
# scripts/backfill_bhavcopy.py would pull 2,886 archives from NSE, which takes
# hours and puts avoidable load on their servers to reproduce files that
# already exist here. The index history would additionally need a live Kite
# session, which would mean putting broker credentials on the VM -- and those
# stay on this laptop.
#
# What is deliberately NOT sent:
#   deploy/*.env       broker credentials; the backtest never contacts Zerodha
#   model_registry/    fitted models; each fold refits from scratch anyway
#   state/, .venv/     machine-local
set -euo pipefail

IP="${1:?usage: push_data.sh <public-ip>}"
ADMIN="${AZ_ADMIN:-irt}"
KEY="${AZ_KEY:-$HOME/.ssh/irt_azure}"
REMOTE_DIR="indian-regime-trader"

cd "$(dirname "$0")/../.."

[ -d data_cache ] || { echo "no data_cache/ here"; exit 1; }

SIZE=$(du -sh data_cache | cut -f1)
printf '\n\033[1m==> sending data_cache (%s uncompressed)\033[0m\n' "$SIZE"

tar czf - data_cache config/nse_holidays.csv \
  | ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "${ADMIN}@${IP}" \
        "cd ~/${REMOTE_DIR} && tar xzf - && echo '  extracted'"

printf '\n\033[1m==> verifying\033[0m\n'
ssh -i "$KEY" "${ADMIN}@${IP}" "cd ~/${REMOTE_DIR} &&
  echo -n '  bar files    : ' && ls data_cache/raw/equity_bars/*.csv 2>/dev/null | wc -l &&
  echo -n '  index series : ' && ls data_cache/raw/index/*.csv 2>/dev/null | wc -l &&
  echo -n '  reference    : ' && ls data_cache/reference/*.csv 2>/dev/null | wc -l &&
  echo -n '  holidays     : ' && wc -l < config/nse_holidays.csv"

echo
echo "  expected here: 2205 bar files, 2 index series, 3 reference tables"
echo "  next: bash deploy/azure/run_remote.sh $IP"
