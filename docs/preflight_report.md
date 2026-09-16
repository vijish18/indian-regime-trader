# Pre-Live Checklist Report

**Result: FAIL**

- Generated: 2026-09-16T03:23:17.870736+00:00
- Conditions satisfied: 14 / 18

See `docs/PRE_LIVE_CHECKLIST.md` for what each condition means and how it is verified. Regenerate with `python -m app.cli preflight`.

| # | Condition | Result | Detail |
|---|---|---|---|
| 1 | unit tests pass | PASS | pytest tests/unit: 1616 passed in 161.29s (0:02:41) |
| 2 | integration tests pass | PASS | pytest tests/unit/test_orchestrator.py tests/unit/test_e2e_validation.py: 44 passed in 88.11s (0:01:28) |
| 3 | look-ahead tests pass | PASS | pytest tests/unit/test_no_lookahead_walkforward.py: 7 passed in 10.23s |
| 4 | walk-forward backtests complete | PASS | pytest tests/unit/test_walk_forward.py: 14 passed in 36.56s |
| 5 | stress tests complete | PASS | pytest tests/unit/test_stress_test.py: 60 passed in 51.16s |
| 6 | paper trading completed successfully | PASS | C:\Users\Vijish\Downloads\indian-regime-trader\docs\validation_report.md records PASSED (- Generated: 2026-09-15T16:41:16.701554+00:00). Confirm this is recent enough before relying on it -- this check does not enforce a staleness window on its own. |
| 7 | broker reconciliation verified | PASS | pytest tests/unit/test_reconciliation.py tests/unit/test_startup.py: 34 passed in 6.35s |
| 8 | API configuration verified | PASS | broker.provider='paper' is a supported adapter; pytest tests/unit/test_kite_broker.py tests/unit/test_kite_mappings.py tests/unit/test_kite_ticker.py: 89 passed in 0.32s |
| 9 | compliance configuration verified | **FAIL** | missing broker authorization: broker_authorization_confirmed is False -- the broker has not been confirmed to have authorized this account for API/algo trading (docs/COMPLIANCE.md section 13). Refusing to operate a live-capable broker. |
| 10 | static-IP configuration verified where applicable | **FAIL** | broker.static_ip_required is True but compliance.static_ip_primary is still the unconfigured placeholder '0.0.0.0' |
| 11 | order-type configuration verified | PASS | order_type='limit', allowed_order_types=['LIMIT', 'SL', 'SL-M'], no_market_order_fallback=True |
| 12 | risk limits configured | PASS | max_gross_exposure=1.0, max_leverage=1.0, max_single_name_pct=0.15, daily_loss_halt_pct=0.03, rolling_loss_halt_pct=0.06, peak_to_trough_drawdown_halt_pct=0.1, max_concurrent_positions=10 -- review these values before enabling live trading; this check only confirms they loaded and are within a wide sanity ceiling, not that they are right for you. |
| 13 | kill switch tested | PASS | pytest tests/unit/test_kill_switch.py: 12 passed in 0.43s |
| 14 | restart recovery tested | PASS | pytest tests/unit/test_startup.py: 23 passed in 6.31s |
| 15 | monitoring tested | PASS | pytest tests/unit/test_alerts.py tests/unit/test_terminal_dashboard.py tests/unit/test_monitoring_snapshot.py: 110 passed in 0.70s |
| 16 | database backups tested | **FAIL** | storage/database.py is still an unimplemented stub (Phase 12) -- no real database exists yet, so no backup/restore procedure can have been tested |
| 17 | secrets are not stored in source code | **FAIL** | tests/unit/test_preflight.py: looks like an embedded private key; tests/unit/test_preflight.py: looks like a literal API key assigned as a string constant |
| 18 | live credentials are not present in development configuration | PASS | BROKER_API_KEY/BROKER_API_SECRET are unset in this process and (if present) empty in .env; config.models.BrokerConfig also has no field that could carry a credential -- settings.yaml is structurally incapable of holding one |

## Live mode remains disabled

The following condition(s) must be resolved before `build_broker` will accept `preflight_confirmed=True`:

- **compliance configuration verified**: missing broker authorization: broker_authorization_confirmed is False -- the broker has not been confirmed to have authorized this account for API/algo trading (docs/COMPLIANCE.md section 13). Refusing to operate a live-capable broker.
- **static-IP configuration verified where applicable**: broker.static_ip_required is True but compliance.static_ip_primary is still the unconfigured placeholder '0.0.0.0'
- **database backups tested**: storage/database.py is still an unimplemented stub (Phase 12) -- no real database exists yet, so no backup/restore procedure can have been tested
- **secrets are not stored in source code**: tests/unit/test_preflight.py: looks like an embedded private key; tests/unit/test_preflight.py: looks like a literal API key assigned as a string constant
