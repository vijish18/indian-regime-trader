# Corporate-action audit — 24 September 2026

**Result: source corrections established; backtest accounting validation is blocked.**
This is not a clean bill of health for the strategy or its reported returns.

## Evidence and scope

- Requested PR archives for all 2,890 dataset sessions, 2015-01-01 through
  2026-09-21. 2,889 contained usable BC records. The BC member for 2022-01-10
  was empty; an official NSE API query for that date returned the IPCALAB
  2-for-1 split. The fallback evidence is recorded in
  `config/corporate_action_verified_terms.json`.
- Audited saved daily holdings from `diagnostic_20260923`: four completed
  strategies and the partial rolling-volatility run, covering 431 instruments.
  This does not certify stocks the next corrected run may newly select.
- The original reference exposed 84 corporate-action dates. After reference
  corrections, 87 dates match PR or supplemental primary-source event evidence.
  Matching the existence of a demerger is not verification of its full lifecycle.
- The raw >10% price-loss screen produced 127 flags on held stock-days.
  These are investigation triggers, not evidence of 127 actual trade losses.
  Absence of an event in PR records does not automatically clear an anomaly.

## Corrections made

The reviewed, reproducible patch list in `config/corporate_action_corrections.json`
records source URLs, archive SHA-256 hashes and expected old amounts:

- INFY, 2018-06-14: dividend 20.50 to 30.50.
- ITC, 2023-05-30: dividend 6.75 to 9.50.
- MPHASIS, 2021-09-13: dividend 38 to 65.
- TATAELXSI, 2021-06-17: dividend 24 to 48.
- Add HEG dividend 30 on 2018-02-15, INFIBEAM dividend 0.10 on 2018-09-19,
  TATAMOTORS dividend 2 on 2023-07-28, and TCS dividend 29 on 2018-05-31.

`scripts/apply_action_corrections.py` writes a new reference file and rejects
unexpected old amounts or duplicates. The corrected local file is
`state/corporate_actions.verified_terms.csv`. Original prices, original references,
frozen VM inputs and previous results have not been overwritten.

The feed parser now preserves compound bonus/dividend entries and sums ordinary
and special dividend components. The engine snapshots dividend entitlement
before ex-date share adjustments and trading, so ex-date buyers do not receive
the dividend and sellers keep their entitlement. The previous diagnostic had
92 mismatched dividend-credit checks against its own reference amounts across
the five strategies. This count is independent of the eight reference repairs.

Only splits and bonuses can restate the same instrument's share count. A
demerger's price factor can no longer create fictitious parent shares.

## Unresolved accounting — do not certify results

- Dividend cash is still credited on ex-date. Payment dates and non-spendable
  receivables must be modelled, including carry across walk-forward boundaries.
- Bonus shares are treated as immediately tradable. Allotment/credit/availability
  dates and fractional entitlements remain unresolved. Five held strategy/event
  instances have fractional bonus entitlements (MANINFRA and WIPRO).
- TATACHEM's 2020 demerger required 114 Tata Consumer shares per 100 parent shares.
  Allotment occurred on 2020-03-11, after the 2020-03-05 record date. The engine
  does not book successor entitlements, cash for fractions or allocation of basis.
- HEXAWARE was suspended from 2020-11-02 and delisted from 2020-11-09. The INR 475
  offer required tendering. There is no automatic cash exit to invent on the
  failed 2020-11-13 fold boundary. The simulation's delisting-exit policy needs
  to be selected and implemented with point-in-time information.
- Unresolved adjusted-price fallbacks and remaining price-drop investigations
  prevent complete data/engine validation.

Preflight now fails on these lifecycle gaps instead of treating successful
fallback pricing as a passing corporate-action check. The dashboard explicitly
labels research results UNVALIDATED. The existing frozen diagnostic VM run is
not modified by this commit. A fresh run and renewed exposure audit are required
after the outstanding accounting is implemented.

## Reproduce the audit

1. `scripts/audit_held_corporate_actions.py`: supply `--run-root`, `--data-root`
   and `--output` to extract actual holdings, share adjustments and dividend checks.
2. `scripts/collect_action_evidence.py`: supply `--audit`, `--anomalies`,
   `--cache`, `--output`, and optionally `--sessions` with the NIFTY50 CSV.
   Cached archives are immutable evidence; failed sources remain explicit.
3. `scripts/reconcile_action_evidence.py`: supply `--audit`, `--evidence`,
   `--reference`, `--supplement config/corporate_action_verified_terms.json`,
   and `--output`. Source matches are separate from blocked accounting status.
4. `scripts/apply_action_corrections.py`: supply `--input`,
   `--corrections config/corporate_action_corrections.json` and a new `--output`.

Local evidence and detailed reports live in `state/held_action_audit*.json`,
`state/action_source_evidence.json`, `state/corporate_action_verification*.json`
and `data_cache/raw/pr`. These large artifacts are intentionally not committed.
Public source URLs and reviewed corrections are versioned in the repository.

Regression checks cover entitlement ownership, compound action parsing, source
deduplication, non-destructive reference patches, checkpoint restart equivalence,
next-session timing and rejection of incomplete lifecycle accounting.
