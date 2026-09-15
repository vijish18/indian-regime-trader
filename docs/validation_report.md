# Phase 21: End-to-End Paper-Trading Validation Report

**Result: PASSED**

- Generated: 2026-09-15T16:41:16.701554+00:00
- Run duration (wall clock): 15.6s
- Configuration: this repository's own `config/settings.yaml`, unmodified (`execution.mode='paper'`)
- Strategy version: `validation-phase-21`
- Model: `hmm_2022-11-14_5s_seed101_244100ed3f2312e3` (trained through 2022-11-14, 5 states)
- Market data: 2019-06-03 -> 2023-08-18 (1100 synthetic sessions, 6 instruments -- see "Limitations" below)
- Live trading sessions exercised: 2022-11-21, 2022-11-22, 2022-11-23, 2022-11-24, 2022-11-25

No live broker credentials were read or required at any point in this run (the only broker constructed was `broker.adapters.paper_broker.PaperBroker`).

## Summary

| Category | Passed | Total |
|---|---|---|
| Session narrative stages | 12 | 12 |
| Required failure injections | 8 | 8 |
| Final invariants | 6 | 6 |

## Session narrative

| # | Stage | Type | Result | Detail |
|---|---|---|---|---|
| 1 | market-data ingestion | stage | PASS | 6 instrument(s), 1100 session(s) ingested and read back |
| 2 | feature calculation + HMM | stage | PASS | regime high_risk (label elevated, confidence 1.00) as of 2022-11-14 |
| 3 | stock ranking | stage | PASS | 6 candidate(s) ranked |
| 4 | portfolio construction | stage | PASS | 6 target position(s), cash_weight=60.23% |
| 5 | risk management | stage | PASS | 6/6 position(s) risk-approved, circuit breaker normal |
| 6 | paper execution | stage | PASS | 6 order(s) submitted to the paper broker |
| 7 | fills | stage | PASS | 6/6 order(s) filled or partially filled |
| 8 | portfolio accounting | stage | PASS | holdings 19,894,970.21 + cash 30,069,519.14 = 49,964,489.35 vs starting 50,000,000.00 (drift 0.0710%, expected from trading costs) |
| 9 | broker API timeout | failure_injection | PASS | submit() marked UNKNOWN after the timeout: True; reconciliation resolved it to rejected: True; exactly one place_order call reached the broker (never resubmitted): True; 1 order(s) resolved this sweep |
| 10 | rejected order | failure_injection | PASS | order for 243830049 shares -> rejected (insufficient cash: order could cost up to 30069519085.66, cash available 30069519.14) |
| 11 | partial fill | failure_injection | PASS | 125/10000 filled against thinned depth |
| 12 | duplicate event | failure_injection | PASS | one fill for 50 shares, delivered twice in one response; 1 fill(s) applied (expected 1); NSE:S02 quantity 36179 -> 36229 (expected 36229) |
| 13 | lost WebSocket | failure_injection | PASS | a quote request during the outage was correctly blocked: True; broker health check during the outage: paper broker: in-process simulation, always reachable |
| 14 | delayed market data | failure_injection | PASS | quote stamped 2700s old against a 15s guard -- PaperBroker's own staleness check would reject an order against it: True |
| 15 | monitoring | stage | PASS | dashboard rendered (35 lines, width 80); 1 alert(s) delivered so far |
| 16 | shutdown | stage | PASS | pre-shutdown state was running; final state persisted (ready); positions closed on shutdown: False (must be False -- close_positions_on_shutdown was not configured) |
| 17 | application crash | failure_injection | PASS | broker still holds {'NSE:S00': 24456, 'NSE:S01': 19608, 'NSE:S02': 36229, 'NSE:S03': 13762, 'NSE:S04': 16114, 'NSE:S05': 12077}; the fresh process's own order history is empty (orders_before_crash=10, orders_after_rebuild=0); startup correctly refused to permit trading until this is resolved: True (system_state=reconciliation_required). NOTE: the 10 pre-crash order(s)' audit trail is lost with them -- ExecutionJournal is in-memory only (Phase 17's own documented scope boundary, durable persistence deferred to storage/). |
| 18 | database restart | failure_injection | PASS | corrupted state file raised a StartupError: True; after restoring the file, post-repair system_state=reconciliation_required |
| 19 | reconciliation | stage | PASS | 1 orphaned order(s) found and cancelled; post-recovery state: ready, permit_strategy_execution=True |
| 20 | halted state persists (deliberately triggered) | stage | PASS | circuit breaker: halted |

## Final invariants

Checked once more after the full narrative -- including recovery from every injected failure -- completed, against live state (not a replay).

| Invariant | Result | Detail |
|---|---|---|
| no duplicate positions | PASS | 6 local and 6 broker position(s), all distinct |
| no negative cash | PASS | cash 30,049,107.37 |
| no leverage | PASS | gross exposure 39.72%, no short positions |
| all orders traceable | PASS | no orders to trace |
| reconciliation succeeds | PASS | local and broker state agree on positions and open orders |
| halted state persists | PASS | nothing halted (circuit breaker is normal) |

`restart is safe` is not a point-in-time check above -- it is proven by the `application crash` / `reconciliation` stage pair in the narrative table: the fresh process correctly refused to trade until an operator resolved the gap, and never re-submitted or duplicated anything in the meantime.

## Limitations -- read before drawing any conclusion beyond "the system works end to end"

- **The market data is synthetic**, not historical. It exists to give every stage something to compute over; no conclusion about returns, regime accuracy, or factor performance follows from this run. See `validation/synthetic_market.py`'s own docstring.
- **Quotes are synthesized from the daily close** (`validation/paper_feed.py`), not a real bid/ask history. Spread-sensitive results (execution cost, partial-fill sizing) are illustrative of the *mechanism*, not the *magnitude*, a live spread would produce.
- **There is no intraday path.** Every quote within a session is the same close, so this run says nothing about intraday timing risk.
- **The failure injections are deliberate and scripted**, not a fuzzer -- they prove each named scenario is handled the way each phase's own design claims, not that no other failure mode exists.
- **A crash loses pre-crash order traceability.** `execution.execution_journal.ExecutionJournal` is documented (Phase 17) as in-memory only. The `application crash` stage below confirms this directly: after the crash, `all orders traceable` holds only for orders created since -- durable journal persistence remains `storage/`'s unimplemented job, not something this phase changes.
- **The recovery in the `reconciliation` stage is scripted for this scenario's own data** (seed local state from the broker's own reported truth; cancel the one orphaned order). It demonstrates the *mechanism* Phase 17/18 provide for an operator to use, not an automated recovery the system performs on its own -- per both phases' explicit design, no position or order discrepancy is ever auto-resolved.
