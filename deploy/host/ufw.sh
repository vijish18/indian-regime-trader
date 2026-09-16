#!/usr/bin/env bash
# Host firewall (Phase 23). Run once, as root, on the trading host.
#
#   sudo deploy/host/ufw.sh
#
# READ THIS BEFORE RUNNING IT. Two things will lock you out of a remote
# host if you get them wrong:
#
#   1. Enabling ufw with a default-deny incoming policy and no SSH rule.
#      The rule below is added before `ufw enable` for that reason. If
#      your SSH daemon listens on a non-default port, set SSH_PORT.
#   2. Assuming ufw protects your containers. It does not, by default --
#      see the DOCKER-USER note at the end, which is the part of this
#      file that actually matters.

set -euo pipefail

SSH_PORT="${SSH_PORT:-22}"
ADMIN_CIDR="${ADMIN_CIDR:-}"   # e.g. 203.0.113.4/32 -- your office/VPN address

[ "$(id -u)" = "0" ] || { echo "run as root" >&2; exit 1; }
command -v ufw >/dev/null 2>&1 || { echo "ufw is not installed" >&2; exit 1; }

ufw --force reset

ufw default deny incoming
ufw default allow outgoing   # the app must reach the broker and the data vendor

# SSH first, always.
if [ -n "${ADMIN_CIDR}" ]; then
    ufw allow from "${ADMIN_CIDR}" to any port "${SSH_PORT}" proto tcp comment 'SSH from admin network'
else
    echo "WARNING: ADMIN_CIDR is unset, so SSH will be open to the whole internet."
    echo "         Set ADMIN_CIDR to your own address range. A key-only SSH daemon"
    echo "         exposed to the internet is survivable; it is still an unnecessary"
    echo "         thing to expose on a host that can move money."
    ufw allow "${SSH_PORT}/tcp" comment 'SSH'
fi

# Nothing else is opened. In particular:
#
#   - Postgres (5432) is NOT opened. deploy/docker-compose.yml publishes
#     no port for it; it is reachable only from the compose network.
#   - No HTTP/HTTPS rule is added, because this deployment exposes no web
#     listener. Add 443 only when you actually put a reverse proxy in
#     front of something (deploy/host/Caddyfile), and never 80 except for
#     the ACME challenge.

ufw --force enable
ufw status verbose

cat <<'NOTE'

----------------------------------------------------------------------
IMPORTANT: ufw does not filter published container ports.

Docker inserts its own rules into the nat and DOCKER chains, which are
traversed before ufw's. A container published with `ports: - "5432:5432"`
is reachable from the internet even with `ufw default deny incoming`.
Every "we exposed our database by accident" story starts here.

This deployment's defence is structural: deploy/docker-compose.yml
publishes no ports at all. Keep it that way. If you ever add one, also
add a matching DOCKER-USER rule, which *is* traversed first:

    iptables -I DOCKER-USER -i eth0 -p tcp --dport 5432 -j DROP

and verify from another machine that the port is closed. Verify from
outside the host -- `ss -tlnp` on the host itself tells you what is
listening, not what is reachable.
----------------------------------------------------------------------
NOTE
