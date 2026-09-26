#!/usr/bin/env bash
# Install or refresh the trading bot's systemd units on the VM.
#   sudo bash /home/irt/bot/deploy/bot/install.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
install -m 0644 "$here"/irt-bot-*.service "$here"/irt-bot-*.timer /etc/systemd/system/
systemctl daemon-reload
for t in "$here"/irt-bot-*.timer; do systemctl enable --now "$(basename "$t")"; done
systemctl list-timers 'irt-bot-*' --no-pager
