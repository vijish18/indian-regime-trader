# Operations

The routine. What to do each day, each week, each month, and what each command
actually tells you.

For putting the system on a host, see [DEPLOYMENT.md](DEPLOYMENT.md). For when
something is wrong, see [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md).

All times are IST (Asia/Kolkata). Logs are UTC — see §7.

---

## 0. Conventions

Every command below assumes:

```bash
ssh deploy@trading-host
cd /srv/indian-regime-trader
alias dc='docker compose -f deploy/docker-compose.yml'
```

---

## 1. The daily routine

NSE regular session: **09:15–15:30 IST**, pre-open 09:00–09:15.

### 08:45 — before the open

```bash
dc ps                          # both services Up (healthy)
dc exec app python -m app.cli health
df -h /var/lib/docker          # see §4
dc logs --since 24h app | grep -i 'fail_closed\|critical' || echo "clean"
```

What you are checking, in order of how much it would ruin the day:

1. **Is it halted?** A `HALTED` state from yesterday persists deliberately across
   restarts and must be cleared by a human before the open — see
   [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md#clearing-a-halt). It will not clear
   itself, and that is the design.
2. **Is the heartbeat fresh?** `health` says so.
3. **Is there disk?** A full disk stops the state file being written, which is a
   database failure, which means no trading (§4).

### 09:00–15:30 — during the session

Leave it alone. The system is designed to be watched, not steered. Watch the
terminal dashboard:

```bash
dc exec app python -m app.cli health     # quick
dc logs -f app                            # the stream
```

The Phase 20 dashboard (`monitoring/terminal_dashboard.py`) shows SYSTEM,
PORTFOLIO, REGIME, EXECUTION and RISK panels from a **single snapshot**
(`monitoring/snapshot.py`) — the screen and the alerts read the same one
gathering, so they can never disagree with each other.

Intervene only for the conditions in INCIDENT_RESPONSE.md. Restarting a running
trading process to "see if that helps" during market hours is how an in-flight
order becomes an order whose state nobody knows.

### 15:45 — after the close

```bash
dc logs --since 12h app | grep -c 'order_submitted'      # orders today
dc logs --since 12h app | grep -i 'rejected\|unknown\|reconcil'
dc exec app python -m app.cli health
```

Reconcile the day: every order should be in a terminal state, positions should
match the broker's own view, and no order should be `UNKNOWN`. An `UNKNOWN` order
left overnight blocks trading tomorrow — by design, see
[fail_closed.py](../orchestration/fail_closed.py) — so it is worth resolving
today, while the broker's support desk is still answering.

### 16:30 — automated

`deploy/backup.sh` runs from cron. Check it did:

```bash
ls -la backups/ | tail -5
cat "backups/$(ls backups | tail -1)/MANIFEST.txt"
tail -20 /var/log/irt-backup.log
```

A backup job that has been silently failing for three weeks is discovered at
exactly the wrong moment. Checking the manifest takes five seconds.

---

## 2. Weekly

**Monday before the open:**

```bash
dc images                                  # note the running image digest
docker system df                           # reclaimable space
dc logs --since 168h app | grep -i 'critical\|fail_closed' | sort | uniq -c
```

**During the week, off-hours:**

- Apply host security updates (`unattended-upgrades` for the OS; restart the
  stack only outside market hours).
- Review the audit log for anything surprising:
  ```bash
  dc exec app sh -c 'ls -la /app/logs'
  dc exec app sh -c 'grep -c fail_closed /app/logs/audit.log'
  ```
- Confirm log rotation is working: you should see `audit.log` plus `audit.log.1`
  … as volume grows, and `docker inspect irt-app` should show the `json-file`
  size caps.

**Friday after the close:** copy the week's backups off-host. A backup on the
host that dies is not a backup.

---

## 3. Monthly

### Restore drill

**This is the single most valuable recurring task in this document.** An untested
backup is a belief, not a backup, and the way you find out is always during an
incident.

```bash
# 1. Pick a backup and unpack it somewhere disposable
BACKUP="backups/$(ls backups | tail -1)"
mkdir -p /tmp/restore-drill && tar -xzf "$BACKUP/app-state.tar.gz" -C /tmp/restore-drill

# 2. Confirm it is what you think it is
cat /tmp/restore-drill/system_state.json | head -40

# 3. Start a throwaway container against it -- NOT the production stack
docker run --rm \
    --volume /tmp/restore-drill:/app/state \
    --env ENVIRONMENT=paper \
    indian-regime-trader:local health

# 4. Clean up
rm -rf /tmp/restore-drill
```

Record the date and the result. If step 3 reports anything but a parseable state,
the backup is not a backup and the reason needs finding *now*.

### Other monthly work

- Rotate credentials that are rotatable (database passwords; broker credentials
  when and if they exist). Update `deploy/app.env` / `deploy/db.env`, restart outside market
  hours.
- Review `docs/preflight_report.md` against reality: `python -m app.cli preflight`
  on the build host. Not to make it pass — to see whether anything that used to
  pass has stopped.
- Review SSH access: `sudo lastlog`, and check `~deploy/.ssh/authorized_keys`
  against the list of people who should still have it. People leave.
- Check disk growth trend (§4).

---

## 4. Disk

Disk exhaustion is the most likely *boring* failure of this deployment, and it is
not boring in its consequences: when the disk fills, the state file cannot be
written, which is a `DATABASE_FAILURE`, which means the system stops trading.

```bash
df -h /var/lib/docker
docker system df
dc exec app sh -c 'du -sh /app/logs /app/state /app/data_cache'
du -sh backups/
```

Three things grow: the market-data cache (fastest, least valuable), the audit log
(capped at ~550 MB by rotation), and backups (capped at 30 dated directories by
`BACKUP_KEEP`). Only the cache is unbounded.

Reclaim:

```bash
docker image prune -a          # old images; safe
docker builder prune           # build cache; safe
# The data cache is reconstructible from the vendor, so it can be cleared --
# but do it outside market hours, since the next run will re-fetch.
```

Never `docker volume prune` on this host without reading which volumes it is
about to remove. `app-state` is unrecoverable.

---

## 5. Common commands

| Task | Command |
|---|---|
| Status | `dc ps` |
| Health | `dc exec app python -m app.cli health` |
| Follow logs | `dc logs -f app` |
| Logs since the open | `dc logs --since 09:15 app` |
| Fail-closed events | `dc logs app \| grep fail_closed` |
| Shell in the container | `dc exec app sh` |
| Read persisted state | `dc exec app cat /app/state/system_state.json` |
| Audit log tail | `dc exec app tail -50 /app/logs/audit.log` |
| Stop (stays stopped) | `dc stop app` |
| Start | `dc start app` |
| Restart | `dc restart app` |
| Backup now | `deploy/backup.sh` |
| Preflight (build host) | `python -m app.cli preflight` |

Note `dc stop app` and `restart: unless-stopped`: a stopped container stays
stopped across a daemon restart and a host reboot. That is deliberate — see
[DEPLOYMENT.md §9](DEPLOYMENT.md#9-automatic-restart) — and it means you must
remember to start it again.

---

## 6. Reading the state file

```bash
dc exec app cat /app/state/system_state.json
```

The fields that answer real questions:

| Field | Question it answers |
|---|---|
| `system_state` | `halted` means a human is required; nothing else will clear it |
| `updated_at` | when the process last proved it was alive (UTC) |
| `portfolio_snapshot` | what this system believes it holds |
| `last_broker_event_id` | how far through the broker's event stream it got |
| `model_version` | which approved model artifact is in force |
| `app_version` | which build wrote this |

`app/service.py`'s heartbeat is a read-modify-write for a reason: everything
except `system_state` and `updated_at` is preserved from whatever a real
orchestrated run recorded. A heartbeat that flattened `portfolio_snapshot` or
`last_broker_event_id` would silently destroy the record of what the system had
already done, in the exact file whose entire purpose is to survive.

---

## 7. Reading the logs

JSON, one object per line, UTC timestamps.

```bash
dc logs app | grep '^{' | jq 'select(.level=="CRITICAL")'
dc logs app | grep '^{' | jq 'select(.event=="orchestrator_fail_closed")'
dc logs app | grep '^{' | jq -r '[.timestamp, .level, .component, .message] | @tsv'
```

**UTC, not IST — do the arithmetic deliberately.** IST is UTC+5:30, so the 09:15
open is `03:45Z` and the 15:30 close is `10:00Z`. A log line at `09:15Z` is 14:45
IST, mid-session, not the open. Getting this wrong during an incident review is
easy and expensive.

Events worth knowing by name:

| Event | Meaning |
|---|---|
| `service_started` | startup completed; mode and paths recorded |
| `service_not_trading` | no composition root wired (see DEPLOYMENT.md §1) |
| `service_fail_closed` | the service could not persist state; halted, still alive |
| `orchestrator_fail_closed` | the daily cycle refused to trade; the reason is in the record |
| `service_shutdown` | clean shutdown, with the signal that caused it |

---

## 8. Changing configuration

Non-secret settings live in `config/settings.yaml`, are versioned, and are baked
into the image. Changing one means rebuilding:

```bash
$EDITOR config/settings.yaml
python -m pytest -q                        # settings are validated by tests
git commit -am "..." && git push
# on the host:
git pull && dc up -d --build
```

Secrets and per-deployment values live in `deploy/app.env` and `deploy/db.env`
and need only a
restart:

```bash
$EDITOR deploy/app.env
dc up -d                                   # recreates with the new environment
```

Both, outside market hours. Configuration that fails to load is a
`CONFIGURATION_FAILURE` and the service refuses to start — which is the correct
behaviour (every limit this system respects is configured, so unreadable
configuration means no limits are known to be in force) and also means a typo
takes the system down until it is fixed. Validate before deploying:

```bash
python -c "from config.loader import load_settings; load_settings(); print('ok')"
```

---

## 9. Operational limits worth knowing

Things that are true today and would surprise someone who assumed otherwise:

- **The deployed service does not trade.** See
  [DEPLOYMENT.md §1](DEPLOYMENT.md#1-what-actually-gets-deployed). It logs this at
  WARNING on every start.
- **Postgres is provisioned but unused.** `storage/database.py` is a stub. The
  state that matters is the JSON file in `app-state`.
- **Preflight reports FAIL (15/18).** Correct and expected; the remaining items
  are real work.
- **There is no alerting transport configured by default.** Alerts are logged;
  `ALERT_WEBHOOK_URL` / `ALERT_EMAIL_TO` in `app.env` are empty. Until one
  is set, "monitoring" means a human reading logs, so the 08:45 and 15:45 checks
  above are not optional.
- **There is no multi-host failover.** One host, one process. A host failure is an
  outage, and the recovery path is DEPLOYMENT.md plus the latest backup.
