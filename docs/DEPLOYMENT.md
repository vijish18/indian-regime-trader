# Deployment

How to put this system on a host, and why each piece is shaped the way it is.

**This document does not enable live trading, and following it will not.** The
deployed stack runs in paper mode and refuses to start in live mode. Turning
live trading on is a separate, deliberately-gated decision documented in
[PRE_LIVE_CHECKLIST.md](PRE_LIVE_CHECKLIST.md).

Companion documents: [OPERATIONS.md](OPERATIONS.md) for the daily and weekly
routine, [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) for when something is
wrong.

---

## 1. What actually gets deployed

Be clear about this before provisioning anything, because it changes what is
worth spending effort on.

The container runs `python -m app.cli serve` ([app/service.py](../app/service.py)),
which loads and validates configuration, configures logging including the durable
audit trail, refuses live mode, proves the persisted state store is writable and
parseable, and then heartbeats on an interval so an external health check can
distinguish a working process from a wedged one.

**It does not run a trading day.** Doing that requires a composition root — a
market-data provider, a populated holiday calendar, an approved model artifact,
and a constructed broker — and none of those exist until this deployment is
provisioned with a real data vendor and real credentials. The daily cycle itself
is written and tested ([orchestration/orchestrator.py](../orchestration/orchestrator.py),
`run_daily_cycle` / `run_forever`); what is missing is the wiring that hands it
real objects. `tests/unit/test_orchestrator.py`'s `Harness` shows exactly what
that wiring looks like.

The service logs this on every start, at WARNING, rather than presenting an idle
process as a trading one. When the composition root lands, the loop in
`app/service.py` becomes a call to `Orchestrator.run_forever` and nothing else in
this document changes.

Deploying now is still worth doing: it exercises the image, the volumes, the
health check, the backup path, and the least-privilege database role against a
process that is easy to reason about, rather than discovering all of them at once
on the first day that money is involved.

---

## 2. Host provisioning

A single Linux host (Debian 12 or Ubuntu 22.04+) with Docker Engine 24+ and the
Compose plugin. 2 vCPU / 8 GB RAM / 40 GB disk is comfortable for the resource
limits in the compose file.

**Location matters for a different reason than usual.** NSE and BSE are in
Mumbai. If this ever goes live, host latency to the broker's API endpoint is part
of the execution assumption the backtest was built on — a host in another
continent is not the same system. Mumbai or Bangalore region, from any major
provider.

### 2.1 Create an unprivileged account

```bash
sudo adduser --disabled-password --gecos "" deploy
sudo usermod -aG docker deploy
```

Note what the second line means: membership of the `docker` group is equivalent
to root on the host, because anyone in it can start a container that mounts `/`.
That is an accepted, understood trade — not an oversight. It is also why the SSH
configuration below matters as much as it does: the `deploy` account is
effectively an administrative account.

### 2.2 SSH: keys only

```bash
sudo cp deploy/host/sshd_config.hardened /etc/ssh/sshd_config.d/10-irt.conf
sudo sshd -t                    # MUST pass
sudo systemctl reload sshd
```

Read the comments in [deploy/host/sshd_config.hardened](../deploy/host/sshd_config.hardened)
before reloading, and keep your current session open while you test a new one in
a second terminal. On a remote host, a mistake here is unrecoverable the moment
you close the last working session.

Key-only rather than passwords because SSH is this deployment's *only*
administrative channel: the dashboard is a terminal UI, the kill switch is a
command, and everything in INCIDENT_RESPONSE.md assumes you can get a shell. An
SSH daemon that can be brute-forced is a trading system that can be operated by
someone else.

### 2.3 Firewall

```bash
sudo ADMIN_CIDR=203.0.113.4/32 deploy/host/ufw.sh
```

Default-deny inbound, SSH only, ideally restricted to your own address range.

**The important part of that script is the warning at the end.** `ufw` does not
filter published container ports — Docker inserts rules into the `nat` and
`DOCKER` chains that are traversed first, so a container published with
`ports: - "5432:5432"` is reachable from the internet even with
`ufw default deny incoming`. Every "we exposed our database by accident" story
starts there.

This deployment's defence is structural rather than procedural:
[deploy/docker-compose.yml](../deploy/docker-compose.yml) publishes **no ports at
all**, and `tests/unit/test_deployment.py::test_the_compose_stack_publishes_no_ports`
fails the build if that ever changes. The firewall is the second line, not the
first.

---

## 3. Secrets

No secret is in Git, in the compose file, or in an image layer.

An image layer is not erasable. A credential `COPY`ed in one layer and deleted in
the next is still in the image, still in the registry, and still readable by
anyone who can pull it. The same is true of a git object: a commit that adds a
secret and a later commit that removes it leave the secret permanently in the
history of every clone. So the defence has to be that they never arrive in the
first place, which is what [.dockerignore](../.dockerignore) and
[.gitignore](../.gitignore) enforce (`deploy/*.env`, `.env`, `*.pem`, `*.key`,
`state/`, `logs/`, `backups/`).

### 3.1 The environment files (start here)

```bash
cp deploy/app.env.example deploy/app.env
cp deploy/db.env.example  deploy/db.env
chmod 600 deploy/app.env deploy/db.env
$EDITOR deploy/app.env deploy/db.env
```

**Two files, not one, on purpose.** `deploy/db.env` holds the database owner's
credentials and is mounted only into the `db` container; `deploy/app.env` holds
the application's, including a `DATABASE_URL` for the *least-privilege* role. A
compromised application container therefore cannot read the credential that can
drop the schema — which is what makes the careful `GRANT`s in §6 worth writing.

**One trap worth knowing about**, because this deployment had it and it was
caught by `docker compose config` rather than by reading: Compose interpolates
`${VAR}` from your shell and from a `.env` in the project directory — **not** from
`env_file:`. A compose file that says `POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}`
while the value lives in an env_file yields an empty password, silently. So no
secret is named in an interpolation anywhere in `deploy/docker-compose.yml`; they
reach the containers through `env_file` alone, and the `db` healthcheck uses
`$$POSTGRES_USER` (escaped, so the *container's* shell expands it) rather than
a host-side `${...}`.

Generate the two database passwords with something that is not your memory:

```bash
openssl rand -base64 32
```

Leave every broker credential **empty**. The paper broker
([broker/adapters/paper_broker.py](../broker/adapters/paper_broker.py)) is an
in-process simulation: it makes no network call and needs no credential.
Credentials present in a paper deployment are pure downside — they can leak, and
they cannot help.

### 3.2 Moving to a secret manager

A `chmod 600` file on one host is an honest baseline and a real limitation: it is
readable by root and by anything in the `docker` group, it does not rotate, and
it leaves no record of access. Before this deployment holds real broker
credentials, move to one of:

| Option | Fits when | What it costs |
|---|---|---|
| **Docker secrets** (Swarm) | you already run Swarm | files under `/run/secrets`, no rotation story |
| **HashiCorp Vault** | multiple hosts, rotation and audit required | an agent and a server to keep alive |
| **Cloud KMS/Secrets Manager** (AWS/GCP/Azure) | already on that cloud | provider lock-in; IAM to get right |
| **systemd `LoadCredential`** | single host, no extra infrastructure | Linux-only, still local |

The application reads `os.environ` in every case, so nothing in the code changes.
Substitute the injection mechanism for `env_file:` in the compose file.

---

## 4. Bring the stack up

```bash
cd /srv/indian-regime-trader/deploy
docker compose up -d --build
docker compose ps                 # app should reach (healthy) within ~90s
docker compose logs -f app
```

You should see, as JSON on stdout:

- `service_started` — configuration loaded, state store verified, mode `paper`
- `service_not_trading` — the honest warning from §1

`docker compose ps` should show `irt-app` as `Up (healthy)` and `irt-db` as
`Up (healthy)`.

### 4.1 What the compose file provides, and why

| Requirement | How | Why this way |
|---|---|---|
| Application container | `Dockerfile`, non-root uid 10001 | see §5 |
| Database | `postgres:16-alpine`, unpublished | see §6 |
| Persistent state | `app-state` volume → `/app/state` | unrecoverable if lost; see §7 |
| Persistent logs | `app-logs` volume → `/app/logs` | the audit trail |
| Environment config | `env_file: app.env` / `db.env` | never in an image layer; split by privilege |
| Health checks | `python -m app.cli health` | see §8 — the design decision is in there |
| Automatic restart | `restart: unless-stopped` | not `always`; see §9 |
| Resource limits | `deploy.resources.limits` | a bound, not a target |
| Log rotation | two mechanisms | see §10 |
| Backup | `deploy/backup.sh` from cron | see §11 |

### 4.2 Verify the refusals

The deployment's most important behaviours are refusals, so verify them rather
than assuming:

```bash
# Refuses live mode, with a distinct exit code
docker run --rm --env ENVIRONMENT=live indian-regime-trader:local serve; echo "exit=$?"
# expect: FAIL CLOSED [configuration failure] ... exit=3

# Refuses to run as root
docker run --rm --user 0 indian-regime-trader:local serve; echo "exit=$?"
# expect: FAIL CLOSED ... refusing to run as root ... exit=1
```

The automated equivalents are `tests/integration/test_docker_smoke.py`
(`pytest -m integration`), which builds the real image and runs the real
container.

### 4.3 Verify the database privileges

Worth doing once per deployment, because "least privilege" is a claim until
something has actually been refused:

```bash
dc exec db psql -U irt_owner -d indian_regime_trader -c "CREATE TABLE probe (id int);"
dc exec db psql -U irt_app   -d indian_regime_trader -c "INSERT INTO probe VALUES (1);"   # works
dc exec db psql -U irt_app   -d indian_regime_trader -c "CREATE TABLE nope (id int);"     # denied
dc exec db psql -U irt_app   -d indian_regime_trader -c "TRUNCATE probe;"                 # denied
dc exec db psql -U irt_app   -d indian_regime_trader -c "DROP TABLE probe;"               # denied
dc exec db psql -U irt_owner -d indian_regime_trader -c "DROP TABLE probe;"
```

The three denials are the point. If any of them succeeds, the init script did
not run — most likely because the `db-data` volume already existed, since
`/docker-entrypoint-initdb.d` fires only on an empty data directory.

---

## 5. The application container

[Dockerfile](../Dockerfile), two stages.

The builder carries a compiler toolchain, because numpy and pyarrow occasionally
need one on a platform without a prebuilt wheel. The runtime does not: a trading
host should not carry a compiler it needed for ten minutes at build time.

Dependencies are installed into a virtualenv and then the project itself is
uninstalled (`pip install . && pip uninstall --yes indian-regime-trader`). That
looks odd and is deliberate. It resolves the dependency set from `pyproject.toml`
alone — no duplicated requirements file to drift — while leaving only *one* copy
of the source in the image, at `/app`, which is the copy that runs and the copy
you can read with `docker exec`. In particular there is no installed `config`
package whose `settings.yaml` could differ from `/app/config/settings.yaml`.

**Hardening, and what each piece buys:**

| Setting | Prevents |
|---|---|
| `USER app` (uid 10001) | a process rewriting its own source, so the source keeps describing what it did |
| source owned by `root:root` | the same, even for the app user |
| `read_only: true` | writes anywhere but the three mounted volumes |
| `cap_drop: ALL` | every Linux capability; the process needs none |
| `no-new-privileges:true` | a setuid binary escalating inside the container |
| `tmpfs /tmp` | needing a writable layer for scratch files |
| `PYTHONDONTWRITEBYTECODE=1` | `__pycache__` writes under a read-only root |

`deploy/entrypoint.sh` refuses live mode and refuses to run as root before Python
starts, then `exec`s Python so that **Python is PID 1**. That last detail matters:
without `exec`, `docker stop` signals a shell that does not forward SIGTERM, and
the trading process is SIGKILLed after the grace period having never run its
shutdown path.

---

## 6. The database

Postgres 16, on the internal compose network only, with **no `ports:` key**.

Two roles ([deploy/postgres-init/01-least-privilege.sql](../deploy/postgres-init/01-least-privilege.sql)):

- **`irt_owner`** owns the schema. Used for migrations, by a human, deliberately.
  The application never connects as this.
- **`irt_app`** is what the application connects as. `NOSUPERUSER NOCREATEDB
  NOCREATEROLE`, `GRANT SELECT, INSERT, UPDATE, DELETE` on rows — no `CREATE`, no
  `ALTER`, no `DROP`, and deliberately no `TRUNCATE`, which is the one row-shaped
  privilege that can erase a whole table's history in a single statement.

The distinction is what stands between a SQL injection or a mistaken migration
and the loss of the order and fill history this system reconciles against after a
restart. Phase 18 established that once that record is gone, the broker's view and
this system's view can never be compared again — it is not recoverable, only
regrettable.

### 6.1 Honest note: the application does not use it yet

[storage/database.py](../storage/database.py) is still a Phase 12 stub. The
compose stack provisions Postgres because the schema and the least-privilege role
should exist and be exercised before anything depends on them — but today, the
state that actually persists is the JSON file in the `app-state` volume
(`execution/system_state.py`) plus the model registry.

Two consequences, stated plainly so nobody is surprised later:

1. `deploy/backup.sh`'s Postgres dump will be nearly empty. The backup that
   matters right now is `app-state.tar.gz`.
2. Preflight check 16 ("database backups") reads **FAIL** and should keep reading
   FAIL until `storage/` is implemented, migrations exist, and a restore has been
   rehearsed. Do not mark it passed because `deploy/backup.sh` exists.

---

## 7. Persistence

Three volumes rather than one, because they have three different recovery
stories:

| Volume | Holds | If lost |
|---|---|---|
| `app-state` | what this system believes it already did | **unrecoverable.** Reconciliation has nothing to reconcile from |
| `app-logs` | the audit trail | the record of what happened is gone |
| `app-data-cache` | vendor market data | re-fetch it; not backed up on purpose |

`app-data-cache` is excluded from backups deliberately: it is the largest and
least valuable thing on the host, and including it would multiply the size of
every backup, which is how backups quietly stop being taken.

---

## 8. Health checks

`python -m app.cli health` ([app/health.py](../app/health.py)) — exit 0 healthy,
1 unhealthy. Docker restarts an unhealthy container, so what this decides is
precisely what gets restarted.

**A HALTED system is reported healthy.** This is the one design decision in this
document most likely to look like a bug, so here is the reasoning in full.

`HALTED` means the system met one of the [fail-closed
conditions](../orchestration/fail_closed.py) and correctly refused to trade. It
is doing exactly what it should. Restarting it would throw away the process that
knows why it halted, re-run startup into the same condition, and halt again — a
crash loop whose only visible symptom is a restart counter. That is the precise
outcome the whole fail-closed design exists to prevent, and a health check is a
very easy place to reintroduce it by accident. A halted container must stay up,
stay queryable, and wait for an operator.

The health output says `HALTED and alive` prominently so a dashboard or alert
rule can act on it — the supervisor just must not act on it first.

What *is* unhealthy: no state file (startup never completed, or the volume is not
mounted), a state file that will not parse (a database failure nothing can trade
through), or a heartbeat older than 300 seconds (the process is wedged). Those are
the cases where replacing the process is the correct and only remedy.

The 300-second bound is five times the 60-second heartbeat interval, so an
ordinary slow iteration never reads as death. If you raise
`monitoring.heartbeat_interval_seconds`, raise this too via
`--max-heartbeat-age-seconds`; a unit test asserts the bound stays at least twice
the interval.

---

## 9. Automatic restart

`restart: unless-stopped`, never `always`.

The difference matters exactly once, and it matters a lot: an operator who
deliberately stops a trading process during an incident must not have it come
back on the next daemon restart or host reboot. `always` would do that. The
whole of INCIDENT_RESPONSE.md assumes `docker compose stop app` means *stopped*.

Automatic restart is only safe *because* the system fails closed. A restart
re-runs startup, which re-runs reconciliation, which refuses to trade if broker
state is unknown. Without that, an automatic restart of a trading process is an
automatic re-attempt of whatever went wrong.

---

## 10. Log rotation — two mechanisms, and both are needed

| Stream | Rotated by | Configured in |
|---|---|---|
| stdout (JSON) | Docker's `json-file` driver, `max-size: 20m`, `max-file: 5` | `deploy/docker-compose.yml` |
| `/app/logs/audit.log` | the process itself, `RotatingFileHandler` | `config/settings.yaml` → `logging.audit_log_*` |

Neither covers the other. Docker rotates the stream it collects, because the
process does not own that file. The process rotates the file it opened itself,
because Docker does not know about it.

Both destinations carry the same records, by design. Splitting a separate "audit"
severity out was considered and rejected: the records an incident actually turns
on are ordinary INFO ones — which order, at what price, on whose risk decision —
and a filter deciding in advance which of those matter is a filter that will be
wrong during the one incident that matters.

The audit file is what survives. The stdout stream lives in the container
runtime's own storage and a `docker system prune` discards it; `/app/logs` is a
volume, backed up nightly. Default retention: 50 MB × 10 files ≈ 550 MB.

If the audit log cannot be opened, the service **refuses to start**
(`monitoring.logger.AuditLogError`). A system that runs without the audit trail
its operator believes it has is worse than one that refuses to start, because the
absence is only discovered during the incident where the trail was needed.

---

## 11. Backups

```bash
crontab -e -u deploy
30 16 * * 1-5 cd /srv/indian-regime-trader && deploy/backup.sh >> /var/log/irt-backup.log 2>&1
```

16:30 IST, after the 15:30 close. [deploy/backup.sh](../deploy/backup.sh) archives
`app-state` and `app-logs` from inside a helper container (volume paths under
`/var/lib/docker` are a storage-driver detail — reading them directly is how a
backup script works on one host and silently produces empty archives on another),
takes a logical `pg_dump` rather than a tarball of a running data directory (a
tarball of a live database is a torn copy that may or may not restore), verifies
every archive is non-empty and readable as gzip, writes a manifest, `chmod`s the
directory to owner-only, and prunes to 30 dated directories.

Backups contain positions, order history and the audit trail. They are exactly as
sensitive as the database. If you copy them off-host — and you should, since a
backup on the host that dies is not a backup — encrypt them in transit and at
rest.

**Restore is the other half, and it is only real if you have done it.** The
monthly restore drill is in [OPERATIONS.md](OPERATIONS.md#restore-drill). An
untested backup is a belief.

---

## 12. HTTPS for the dashboard — the honest answer

**This deployment exposes no web dashboard, so there is nothing for HTTPS to
protect.** The Phase 20 dashboard
([monitoring/terminal_dashboard.py](../monitoring/terminal_dashboard.py)) renders
to a terminal. It is reached by:

```bash
ssh deploy@trading-host
docker compose -f deploy/docker-compose.yml exec app python -m app.cli ...
```

which is already an encrypted, key-authenticated, logged channel — a better one
than an HTTPS page with a password would be, and one with no listener to attack.

[deploy/host/Caddyfile](../deploy/host/Caddyfile) is a template for the day a web
dashboard exists, kept in the repository so that day does not begin with someone
binding a development server to `0.0.0.0`. It is not installed and not running.
When that day comes: bind the listener to `127.0.0.1` only, run Caddy on the host
(not in the compose stack, so certificates survive an app redeploy), open 443 in
`ufw`, and keep the access log — a dashboard that can trigger the kill switch is
an administrative interface, and the application's audit log records what the
system did, not who asked it to.

---

## 13. Audit logging

Every log record, in JSON, to both destinations in §10. Records carry a UTC
timestamp, the component name, and structured `extra_fields` including the event
name — so `service_started`, `orchestrator_fail_closed`, `service_fail_closed`
and the execution-layer events are greppable rather than prose to be read.

Host-level audit is separate and complementary: `LogLevel VERBOSE` in the SSH
configuration records the *key fingerprint* used for each login, which is what
lets a review answer "which key, belonging to whom" rather than only "someone
logged in".

---

## 14. Upgrading

```bash
cd /srv/indian-regime-trader
deploy/backup.sh                         # before, not after
git pull
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml ps    # wait for (healthy)
docker compose -f deploy/docker-compose.yml logs --tail 50 app
```

Never during market hours. The volumes are external to the container, so state
and logs survive the replacement; that is the point of §7.

Rollback: `git checkout <previous-tag> && docker compose up -d --build`. If state
is also involved, restore the `app-state` archive taken *before* the upgrade —
which only exists if you ran the first line.

---

## 15. Going live — not from here

Live trading is gated by four independent checks in
[broker/factory.py](../broker/factory.py) (`execution.mode == "live"`,
`enable_live_trading=True`, `preflight_confirmed=True`, and a passing
`ComplianceGate`), and a deployment is not one of them.

This is deliberate. A container image is copied between hosts, promoted between
environments, and restarted by a supervisor; none of those events is a human
deciding to risk real money, so none of them may be what causes real money to be
risked. `deploy/entrypoint.sh` and `app/service.py` both refuse live mode
outright, and `docker run --env ENVIRONMENT=live ...` exits 3.

The path is [PRE_LIVE_CHECKLIST.md](PRE_LIVE_CHECKLIST.md) and
`python -m app.cli preflight`. As of this phase, preflight reports **FAIL** (15 of
18 checks), which is correct and expected — the remaining items are real work, not
paperwork.

Note that `tests/` is excluded from the image by `.dockerignore`, so preflight
runs on the build host or in CI, not inside the production container.
