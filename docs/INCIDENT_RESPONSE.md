# Incident response

What to do when something is wrong.

This document assumes you are reading it under time pressure, so it is ordered by
what you need first. For the daily routine see [OPERATIONS.md](OPERATIONS.md);
for provisioning see [DEPLOYMENT.md](DEPLOYMENT.md).

```bash
ssh deploy@trading-host
cd /srv/indian-regime-trader
alias dc='docker compose -f deploy/docker-compose.yml'
```

---

## 0. The first thing to know

**A halted system is not an emergency. It is the system working.**

This system is built to fail closed: when it cannot establish that trading is
safe, it stops trading, says which of six named conditions applied, and stays
alive to be asked about it. `HALTED` in the state file means the safety machinery
did its job.

The emergencies are the opposite: a system that is *trading* when it should not
be, a position that does not match the broker's, an order whose state nobody
knows. Those are in §2.

**Never react to a halt by restarting until it clears.** A halt that clears on
restart has not been resolved — it has been hidden, and the next occurrence will
be in front of a real position. Find out why first.

---

## 1. Severity, and what it costs to get it wrong

| Sev | Definition | Response | Examples |
|---|---|---|---|
| **1** | Money is at risk right now | Immediate, drop everything | unknown order state with capital committed; position mismatch; system trading while it should be halted |
| **2** | Trading is stopped | Within the hour, during market hours | fail-closed halt; container unhealthy; broker unreachable |
| **3** | Degraded but trading safely | Same day | stale data recovered on its own; alert transport down; disk above 80% |
| **4** | No trading impact | Next business day | log noise; backup verification warning |

When in doubt, treat it as one level more severe. The cost of over-reacting is an
hour; the cost of under-reacting is a position nobody knows about.

---

## 2. Emergency: stop trading now

If you believe the system is doing something wrong with real money, in
decreasing order of preference:

### 2.1 The kill switch (preferred)

```bash
dc exec app python -c "
from risk.circuit_breaker import CircuitBreaker
# ...construct against the deployment's own state path...
"
```

`live.kill_switch.KillSwitch.engage(operator, reason)` →
`CircuitBreaker.force_halt(operator, reason)`, which halts **unconditionally**,
bypassing the threshold evaluation in `evaluate()`. It is recorded with the
trigger label `kill_switch`, your name, and your reason. It is the preferred
route because it stops trading while leaving the process alive, so you can still
see what it holds and what it did.

Nothing clears it but a human: `KillSwitch.release(operator, reason)`, after §5.

### 2.2 Stop the container

```bash
dc stop app
```

`restart: unless-stopped` means stopped is *stopped* — across a daemon restart
and a host reboot. That is deliberate. It also means you must remember to start
it again, and that until you do, nothing is reconciling.

### 2.3 Stop at the broker

If neither of the above is reaching the problem, go around the system: log into
the broker's own web terminal and cancel open orders or square off there. Then,
critically, **do not restart this system until you have reconciled** — it has a
record of positions that is now wrong, and restarting it against a broker state
it did not cause is exactly the situation §4 is about.

---

## 3. The six fail-closed conditions

Each is [`FailClosedReason`](../orchestration/fail_closed.py), named on the
`DailyCycleReport` and logged as `orchestrator_fail_closed` at CRITICAL.

```bash
dc logs app | grep '^{' | jq 'select(.event | test("fail_closed"))'
```

### `unknown broker state` → no more orders

**What it means.** An order's true state at the broker could not be established,
or the broker reports state this system has no record of.

**Why it stops everything.** Placing another order while an earlier one's outcome
is unknown risks doubling a position that may already exist. This is the one
condition where continuing is actively worse than stopping.

**Do:**
1. Open the broker's own order book. That is the authority, not this system.
2. Compare against `dc exec app cat /app/state/system_state.json`.
3. Resolve every ambiguous order at the broker — cancel or confirm.
4. Only then, §5.

**Read §4 before doing this one.**

### `stale market data` → do not trade

**What it means.** The data the system would decide on is older than the
configured freshness bound — or the freshness check itself could not run, which
is treated identically ("an unverifiable feed is a stale feed").

**Why.** A decision on stale data is not a conservative decision; it is a decision
about a market that no longer exists.

**Do:** check the vendor's status page, then the host's connectivity, then
`data.freshness` configuration. This condition is `DEGRADED`, not `HALTED` — it
self-heals when a clean iteration succeeds. If it does not self-heal, the feed
really is down.

### `risk engine failure` → do not trade

**What it means.** The independent risk layer could not produce a decision.

**Why.** Its veto is non-negotiable, so an absent verdict reads as *rejected*,
never as "nothing objected". A missing answer is not a yes.

**Do:** this is a code or configuration fault, not a market event. Read the
traceback, fix it, redeploy. Do not work around it.

### `database failure` → do not trade

**What it means.** Persisted state could not be read or written.

**Why.** Without it, a restart cannot know what this system already did — the
exact state Phase 18's reconciliation exists to avoid trading through.

**Do:**
```bash
df -h /var/lib/docker                             # full disk is the usual cause
dc exec app ls -la /app/state
dc exec app cat /app/state/system_state.json      # does it parse?
docker volume inspect indian-regime-trader_app-state
```
If the state file is corrupt, **do not delete it**. It is the only record of what
the system believed. Copy it aside, then restore from the most recent backup
(§7), then reconcile against the broker before starting.

### `configuration failure` → do not trade

**What it means.** Configuration failed to load or validate.

**Why.** Every limit this system respects is configured, so unreadable
configuration means no limits are known to be in force.

**Do:**
```bash
python -c "from config.loader import load_settings; load_settings(); print('ok')"
```
on the build host. The error names the failing key. Usually a bad edit or a
missing environment variable.

### `market-calendar uncertainty` → do not trade

**What it means.** Whether the exchange is open could not be determined —
typically an uncovered year in the holiday file.

**Why.** "I cannot tell" must never be read as "the exchange is closed". That
would look exactly like a normal quiet day, and the system would sit out a real
session without anyone noticing.

**Do:** update `config/`'s holiday file from the exchange's published circular —
not from memory and not from a third-party list. NSE publishes trading holidays
annually; muhurat sessions and unscheduled closures are the cases that bite.

---

## 4. The reconciliation caveat you need to know about

**`execution.order_manager.handle_ambiguous_response` treats "the broker does not
recognize this order" and "the broker could not be reached" identically: both
resolve the order to `REJECTED`.**

That is deliberately conservative in one direction — it never invents a fill —
but it means a **transient broker outage during reconciliation can mark an order
rejected when the order actually exists and may have filled.**

So: **before you accept "rejected" as meaning "no position was taken", verify
against the broker's own order book.** Especially if the rejection coincided with
a connectivity problem, a timeout, or a broker status incident. This is the single
most likely way for this system's view and reality to diverge silently.

This is a known, documented limitation rather than a bug report: changing the
behaviour would destabilise three phases of tested reconciliation logic, and the
conservative direction is the right one to err in. But it puts a manual
verification step in your hands, and this is where it is recorded.

---

## 5. Clearing a halt

A halt never clears itself. Two mechanisms, for two different things:

| Halted by | Cleared by |
|---|---|
| a circuit breaker threshold, or the kill switch | `CircuitBreaker.manual_reset(operator, reason)` |
| a startup/reconciliation failure | `StartupSequence.acknowledge_and_recover(operator, reason)` |

Both require your name and your reason, and both record them. That is not
bureaucracy: the record is what a post-incident review reads, and "cleared by
someone, for some reason" is not a reviewable statement.

**Before clearing anything, answer all four:**

1. Do I know which of the six conditions applied, and why?
2. Has the underlying cause been fixed — not worked around, not waited out?
3. Does this system's view of positions match the broker's own, checked at the
   broker?
4. Is there any order in a non-terminal state?

If you cannot answer all four, you are not ready to clear it. Stopped costs
opportunity; wrongly resumed costs capital.

---

## 6. Other incidents

### Container unhealthy or restarting

```bash
dc ps
docker inspect irt-app --format '{{json .State.Health}}' | jq
dc logs --tail 100 app
```

`unhealthy` means one of: no state file (startup never completed, or the volume
is not mounted), an unparseable state file, or a heartbeat older than 300s (the
process is wedged). Note that a *halted* system reports **healthy** on purpose —
see [DEPLOYMENT.md §8](DEPLOYMENT.md#8-health-checks) — so "unhealthy" genuinely
means the process is not working, not that trading stopped.

A restart loop is the one case where reading the logs beats any other action: the
process is dying before it can persist a reason, so stdout is all you have.
`dc logs app` retains it across restarts.

### Host full

See [OPERATIONS.md §4](OPERATIONS.md#4-disk). A full disk becomes a
`database failure`, so it stops trading. `docker image prune -a` and
`docker builder prune` are safe. `docker volume prune` is **not** — `app-state`
is unrecoverable.

### Suspected compromise

Treat the host as untrusted immediately.

1. **Do not restart anything.** A restart destroys volatile evidence.
2. Stop trading at the broker (§2.3) and revoke API credentials at the broker —
   from a different, known-good machine.
3. Isolate: `sudo ufw default deny outgoing` (accepting that this also stops the
   system reaching the vendor — that is the trade).
4. Preserve: copy `/app/logs/audit.log`, `/app/state/`, `docker logs`,
   `/var/log/auth.log` off-host before touching anything else.
5. Check `sudo last`, `sudo grep sshd /var/log/auth.log | grep -i accept` —
   `LogLevel VERBOSE` records the key fingerprint, so you can tell *which* key.
6. Rebuild the host from scratch. Do not clean it. Rotate every credential.

### Broker disconnected

Alerts fire (`monitoring/alerts.py`, rate-limited per alert type and subject).
The system stops submitting; open orders stay at the broker and continue to live
their own lives there. Check the broker's status page, then the host's outbound
connectivity, then §4 before trusting any order state you see afterwards.

---

## 7. Restoring from backup

```bash
dc stop app                                    # never restore under a running process
BACKUP="backups/<timestamp>"
cat "$BACKUP/MANIFEST.txt"                     # confirm it is what you think

docker run --rm \
    --volume indian-regime-trader_app-state:/target \
    --volume "$(pwd)/$BACKUP:/backup:ro" \
    alpine:3.20 sh -c "rm -rf /target/* && tar -xzf /backup/app-state.tar.gz -C /target"

dc start app
dc exec app python -m app.cli health
```

**Then reconcile against the broker before allowing any trading.** A restored
state file describes the world as of the backup, and the world has moved. What
the broker holds now is the truth; the restored file is a starting point for
comparison, not an answer.

**Never restore `app-logs` over a newer audit log.** Restore it to a separate
location for reading. An audit trail you have overwritten is no longer an audit
trail.

---

## 8. After the incident

Write it down while it is fresh, in `docs/incidents/YYYY-MM-DD-<slug>.md`:

- **Timeline** in IST, with the UTC log lines that support it (logs are UTC;
  IST is UTC+5:30, so the 09:15 open is 03:45Z — get this right or the timeline
  is fiction).
- **What was detected, by what, and how long after it started.** If a human
  noticed before the monitoring did, that is the most important finding in the
  report.
- **Money impact**, stated plainly. Including "none".
- **Root cause**, not the first plausible cause.
- **What is changing.** A guarantee is worth more than a resolution: if the
  condition could recur, the fix is a test that fails when it does. Every
  fail-closed condition in `orchestration/fail_closed.py` has one
  (`tests/unit/test_fail_closed.py`), and that is the standard to hold new ones
  to.

If the incident revealed a gap in *this* document, fix this document. It is the
one that gets read under pressure.
