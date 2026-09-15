# Pre-Live Checklist

**Live trading remains disabled until every condition below is
satisfied.** This is not a suggestion an operator can override by
reasoning "it's probably fine" — `broker.factory.build_broker` will not
construct a live-capable broker without four independent, explicit
confirmations (`execution.mode == "live"`, `enable_live_trading=True`,
`preflight_confirmed=True`, and a fully-configured `ComplianceGate`), and
`preflight_confirmed=True` is only honest to pass after this checklist's
automated command has actually reported PASS, freshly, in the current
session. See `broker/factory.py`'s own module docstring for exactly how
that gate is enforced in code.

## Run it

```
python -m app.cli preflight
```

Runs all eighteen conditions below, prints PASS or FAIL for each with a
concrete reason, writes `docs/preflight_report.md`, and exits `0` only if
every condition passed (`1` otherwise) — so it can gate a deploy step, not
just be read by a human. It submits no order and constructs no live
broker; the only network- or process-level activity it performs is
running this repository's own test suite as subprocesses.

Checks 1-5, 7, and 13-15 (and part of 8) run genuinely, as subprocesses,
every time this command is invoked — they are not read from a stale
record. `--skip-test-suites` exists only for fast local iteration on the
other checks; a run with it set can never report an overall PASS (see
`live.preflight.run_preflight`'s own docstring).

## The eighteen conditions

| # | Condition | How it is verified |
|---|---|---|
| 1 | Unit tests pass | `pytest tests/unit` (the whole suite), as a subprocess |
| 2 | Integration tests pass | `pytest tests/unit/test_orchestrator.py tests/unit/test_e2e_validation.py` |
| 3 | Look-ahead tests pass | `pytest tests/unit/test_no_lookahead_walkforward.py` |
| 4 | Walk-forward backtests complete | `pytest tests/unit/test_walk_forward.py` |
| 5 | Stress tests complete | `pytest tests/unit/test_stress_test.py` |
| 6 | Paper trading completed successfully | `docs/validation_report.md` (Phase 21) exists and records `**Result: PASSED**` |
| 7 | Broker reconciliation verified | `pytest tests/unit/test_reconciliation.py tests/unit/test_startup.py` |
| 8 | API configuration verified | `broker.provider` is a supported adapter, and `pytest tests/unit/test_kite_broker.py tests/unit/test_kite_mappings.py tests/unit/test_kite_ticker.py` |
| 9 | Compliance configuration verified | `broker.compliance.ComplianceGate(settings.compliance)` constructs without raising |
| 10 | Static-IP configuration verified where applicable | Not applicable if `broker.static_ip_required` is `False`; otherwise `compliance.static_ip_primary` must not be the `"0.0.0.0"` placeholder |
| 11 | Order-type configuration verified | `execution.order_type` is in `compliance.allowed_order_types`, `no_market_order_fallback` is `True`, `MARKET` is not an allowed type |
| 12 | Risk limits configured | `settings.risk` loads (schema-validated) and stays within this check's own sanity ceilings; every configured limit is listed for a human to review |
| 13 | Kill switch tested | `pytest tests/unit/test_kill_switch.py` (`risk.circuit_breaker.CircuitBreaker.force_halt` / `live.kill_switch.KillSwitch`) |
| 14 | Restart recovery tested | `pytest tests/unit/test_startup.py` |
| 15 | Monitoring tested | `pytest tests/unit/test_alerts.py tests/unit/test_terminal_dashboard.py tests/unit/test_monitoring_snapshot.py` |
| 16 | Database backups tested | `storage/database.py` must be a real implementation (not the Phase 12 stub) with a tested backup/restore procedure under `scripts/` |
| 17 | Secrets are not stored in source code | Every git-tracked file is scanned for embedded-secret-shaped patterns; `.env` must not be git-tracked |
| 18 | Live credentials are not present in development configuration | `BROKER_API_KEY`/`BROKER_API_SECRET` must be unset in the current process and, if `.env` exists, empty there too |

## What "PASS" proves, and what it does not

A handful of these conditions are things code can verify were *exercised
and passed their own tests* — not that a human has made the underlying
business judgment they stand in for:

- **Compliance (9-11)** verifies `ComplianceConfig` is internally
  consistent and not still the `settings.yaml` placeholder. It cannot
  verify that `broker_authorization_confirmed=True` is actually true —
  that a human confirmed with the broker, in the real world, that this
  account's API/algo setup is eligible for live trading. That attestation
  belongs to a human, not this script. See `docs/COMPLIANCE.md`.
- **Static IP (10)** verifies a real-looking value is configured. It
  cannot verify that value is the IP actually registered with the broker.
- **Risk limits (12)** verifies the configuration loads and stays under a
  wide sanity ceiling. It cannot tell you whether those limits are *right
  for your capital and risk tolerance* — that is a decision for whoever
  is putting capital behind this system, every time, not a one-time code
  check.
- **Database backups (16)** will read FAIL today, honestly: `storage/`
  remains an unimplemented stub (see `docs/ARCHITECTURE.md`'s phase
  plan), so there is no backup/restore procedure to have tested. This is
  not a false negative to work around — it is the correct signal that
  live trading is not ready, in this specific respect, until a real
  persistence layer and a tested backup procedure exist.

A PASS on every condition is **necessary, not sufficient**, for the
decision to actually go live. That decision itself is `enable_live_trading=True`
— a separate, explicit, human act at the call site, every time, never
inferred from a config file or a memory of a prior successful run.

## Before you flip `execution.mode` to `"live"`

1. Run `python -m app.cli preflight` and confirm it prints `OVERALL: PASS`.
2. Read `docs/preflight_report.md` in full — not just the last line. Every
   condition's detail is there, including the exact configured risk
   limits (condition 12) for a final human review.
3. Fill in `settings.compliance` with values a human has actually
   confirmed with the broker (`docs/COMPLIANCE.md`), not placeholders
   that merely satisfy the schema.
4. Set `BROKER_API_KEY`/`BROKER_API_SECRET` in the environment at deploy
   time — never in `settings.yaml`, never committed.
5. Pass `enable_live_trading=True` and `preflight_confirmed=True` (the
   result you just obtained, not a remembered one) to `broker.factory.build_broker`
   explicitly, in code you are looking at, at the moment you mean to go
   live.

If step 1 does not print `OVERALL: PASS`, stop. Live mode is not ready.
