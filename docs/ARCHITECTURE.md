# Architecture

This document maps [SPECIFICATION.md](SPECIFICATION.md)'s nine required separations onto
this repository's packages, states the dependency rules that keep the boundaries real (not
just directories), and records the small number of places where an implementation decision
had to resolve something the specification left ambiguous.

## Layer -> package mapping

The specification asks for nine separated concerns; this repository organizes them into
thirteen top-level packages. Two packages each cover two related concerns because they
share the same causal/timing constraints:

| Specification concern | Package(s) |
|---|---|
| Market-regime detection | `core/regime/` |
| Feature engineering | `core/features/` |
| Stock selection | `universe/` (alongside universe construction — see below) |
| Portfolio construction | `portfolio/` |
| Risk management | `risk/` |
| Broker / execution | `execution/` + `broker/` |
| Backtesting | `backtest/` |
| Data storage | `storage/` (+ `data/` for the ingestion/quality side of "data") |
| Monitoring | `monitoring/` |

`core/` groups regime detection and feature engineering because both operate on the same
causal, market-level time series and share the "no look-ahead" discipline; they are still
separate subpackages (`core/regime/`, `core/features/`) with their own `__init__.py`, not a
flat namespace.

`universe/` groups point-in-time universe construction (`universe.py`) and stock selection
(`stock_selector.py`) because selection always operates over a specific universe snapshot;
keeping them in one package does not mean they share logic — `stock_selector.py` must never
take the current regime label as an input (see "Dependency rules" below).

`broker/` (the neutral interface + adapters) is split from `execution/` (order lifecycle,
position tracking, reconciliation) because the interface needs to be swappable without
touching the state machine or reconciliation logic that sits above it.

## Dependency rules

These are enforced by convention now (Phase 1) and should be enforced by an import-linter
rule or equivalent test once there is real code to check (Phase 5+):

- `core/` must not import from `universe/`, `portfolio/`, `risk/`, `execution/`, or
  `broker/`. It answers "what is the market doing", never "what should we own" or "how do
  we trade it". This is the structural guarantee that the HMM cannot become a stock-price
  or direction predictor by accident.
- `universe/stock_selector.py` must not import `core/regime/`. Regime state reaches
  portfolio construction only through `core/regime/regime_policy.py`'s exposure target, not
  through the selection layer — this is what lets a walk-forward run measure the regime
  layer's incremental value in isolation from selection.
- `portfolio/` proposes; it does not decide. `portfolio_constructor.py` produces
  `ProposedWeight` objects, not orders. `risk/risk_manager.py` always evaluates proposals
  and always has the final say — this is the structural form of specification section 8's
  "NON-NEGOTIABLE... independent veto".
- `risk/position_sizer.py` is the single place specification section 7.2's weight-based
  sizing formula and section 8.1's stop-distance risk-based sizing formula are reconciled
  into one final order quantity. No other module computes a final order quantity.
- `execution/` and `broker/` depend on `risk/`'s approved decisions; they never re-derive a
  target weight or quantity themselves.

## Resolved specification ambiguities

The architecture review that preceded this phase flagged several places where
[SPECIFICATION.md](SPECIFICATION.md) states a principle and a later section's concrete
detail doesn't fully match it, or leaves an implementation detail unstated. Re-verifying
the source PDF while building this scaffold cleared up one of those (the risk-number
"mismatch" was a PDF-table-extraction artifact, not a real inconsistency — corrected below).
The ones that are real are resolved here so Phase 5+ doesn't have to rediscover them
mid-implementation.

**Risk thresholds (not actually ambiguous).** An initial read of the PDF's risk-control
table (section 8) appeared to conflict with its suggested config block (section 21) because
a column-layout PDF extraction had scrambled which value belonged to which label. Re-read
against the PDF's natural reading order, section 8 and section 21 agree exactly (daily
reduce 2.0%, daily halt 3.0%, weekly reduce 4.0%, weekly halt 6.0%, peak-drawdown hard halt
10%). [config/settings.yaml](../config/settings.yaml) uses these values, and additionally
includes `daily_loss_warning_pct` (1.5%) and `max_concurrent_positions` (10), both present
in section 8's prose table but omitted from section 21's example config.

**Minimum holdings vs. single-name cap vs. calm-regime exposure (genuinely under-specified).**
Section 7.1 says "start with 5–10 positions"; section 8 caps single-name weight at 15%;
section 7 targets 85–100% gross exposure in the calm regime. Five positions at a 15% cap
can reach at most 75% gross exposure — short of the calm-regime floor, let alone its
ceiling. `config/settings.yaml` sets `selection.min_holdings: 7` (7 x 15% = 105%, enough to
reach 100%), and `config/models.py`'s `Settings` model enforces
`selection.min_holdings * portfolio.max_single_name_pct >= regime_policy.calm.max_gross_exposure`
as a validator, so this can't silently regress if either number changes later.

**Two position-sizing formulas (genuinely under-specified).** Section 7.2 sizes by
`selection_score / volatility` (a weight-based construction); section 8.1 sizes by
`floor((equity * max_risk_per_position) / |entry - stop|)` (a stop-distance risk-based
construction). Nothing in the source specifies how these combine. This repository puts
both as inputs to `risk/position_sizer.py::reconcile`, which is specified (Phase 7/8 stub)
to take the minimum of the two before applying single-name/sector/cash/liquidity caps, and
to record which constraint bound the final quantity for auditability. The `stop_distance`
input to the risk-based formula is documented as a sizing input (e.g. ATR-based), distinct
from a resting protective stop order, since section 1.2 demotes live stop orders to a
last-resort control while section 8.1's formula still needs a risk-distance estimate.

**Signed-return features in the HMM feature set (flagged, not yet resolved — Phase 4/5
decision).** Section 1.2 warns against "too many directional features" causing the HMM to
learn direction instead of risk state, but section 5's feature table still lists three
signed return features (1-day, 5-day, 20-day) alongside the volatility features. This
repository's `core/features/feature_engineering.py` docstring flags this explicitly; the
actual resolution (keep signed returns with justification, or switch to `|return|`/squared
return) is a Phase 4 decision informed by the section 6.3 validation gates, not a Phase 1
one, and is deliberately left open here rather than pre-decided.

## Phase plan

Follows [SPECIFICATION.md section 18](SPECIFICATION.md#18-codex-build-plan) with one
addition: this phase (Phase 1) also resolves the ambiguities above in configuration, so
Phase 5 (HMM) and Phase 7 (portfolio/risk) start from an unambiguous spec. Everything past
Phase 1 in this repository is currently a typed stub — a class/function with a docstring, a
type-hinted signature, and a `raise NotImplementedError("Phase N: ... is not implemented
yet.")` body — not working logic.

| Phase | Scope | Status |
|---|---|---|
| 1 | Repository, typed/validated configuration, logging, environment handling, test framework | **Done** |
| 2 | Market calendar, point-in-time instrument master | **Done** |
| 3 | Data ingestion, corporate actions, data quality | **Done** (local files; no vendor/broker feed) |
| 3b | Point-in-time universe construction (`universe/universe.py`) | **Done** |
| 4 | Causal feature engineering (`core/features/feature_engineering.py`) | **Done** (feature scaling for walk-forward fitting, `core/features/feature_scaler.py`, is still Phase 5) |
| 5 | HMM engine, model registry, causal feature scaling | Stubbed |
| 6 | Regime policy, stock selector | Stubbed |
| 7 | Portfolio constructor, position sizer, risk manager | Stubbed |
| 8 | Backtest engine, Indian cost/slippage model, performance metrics | Stubbed |
| 9 | Walk-forward validation, stress testing | Stubbed |
| 10 | Broker interface, paper adapter, order manager | Stubbed |
| 11 | Position tracking, reconciliation, live operational controls | Stubbed |
| 12 | Alerts, health checks, dashboard, production go-live gate | Stubbed |

## The data layer (phases 2-3)

`data/` is split into contracts and implementations so nothing above it knows
where data came from:

| Module | Role |
|---|---|
| `data/models.py` | `Instrument`, `DailyBar`, `Quote`, `TradingSession`, `CorporateAction`, `IndexObservation`, `IndexMembership` |
| `data/interfaces.py` | `MarketDataProvider`, `InstrumentRepository`, `TradingCalendar`, `CorporateActionProvider`, `IndexMembershipProvider` |
| `data/calendar.py` | `NSETradingCalendar` — holidays, weekends, special sessions |
| `data/instrument_master.py`, `data/corporate_actions.py`, `data/membership.py` | Point-in-time reference-data implementations |
| `data/market_data.py` | `LocalMarketDataProvider` — historical bars from files |
| `data/storage.py` | File layout, CSV/Parquet I/O, exact `Decimal` parsing |
| `data/data_quality.py`, `data/ingestion.py` | Validation and the validate-then-store pipeline |

Four decisions in this layer are load-bearing:

**Prices are `Decimal`.** Tick-size arithmetic, cost computation and price-band
checks need exact values; a 0.05 tick is not representable in binary floating
point. Analytics converts to float at the pandas boundary, never the reverse.

**Adjusted prices are computed as-of a date.** `cumulative_adjustment_factor`
uses only actions with `price_date < ex_date <= as_of`. The upper bound is the
look-ahead guard: a walk-forward fold ending on T sees the series as it looked
on T, so a split announced later cannot retroactively rewrite it.

**The calendar fails closed outside its coverage.** Asking about a year the
holiday dataset does not cover raises rather than assuming "no holidays" —
which would turn missing data into wrong data (trading on Republic Day).
`config/nse_holidays.csv` therefore ships empty and must be populated from
NSE's published list; fabricating dates would be worse than refusing to run.

**Parsing and validation are separate.** Models represent whatever the vendor
sent, including impossible bars, so `data_quality` can report them instead of a
parser silently dropping rows. Validation returns a report rather than raising,
so one pass surfaces every problem in a file — including dates it could not
check, which are warnings, not silent passes.

Point-in-time membership (`data/membership.py`) is the survivorship-bias
control: no constituent list is hardcoded anywhere, so a 2019 backtest sees the
index as it was in 2019, including companies later deleted.

## The universe engine

`universe/universe.py`'s `UniverseProvider.get_universe(as_of)` is where the
survivorship-bias control in `data/membership.py` becomes an actual eligible
universe, by joining three point-in-time sources:

1. **Index membership** — was the instrument a constituent on `as_of`.
2. **Instrument status** — was it active (not suspended/delisted), and under
   what symbol, on `as_of`.
3. **Corporate actions** (optional) — had a merger/demerger/delisting with
   `ex_date <= as_of` already completed.

Every check is bounded by `as_of`: membership uses `is_member_on(as_of)`,
instrument lookup uses `get(id, as_of)`, and the corporate-action cross-check
passes `end=as_of` to `actions_for(...)`, so a scheduled-but-not-yet-effective
action can never exclude an instrument early. That bound is exercised directly
by `test_future_corporate_action_does_not_exclude_early_because_that_would_be_look_ahead`
in `tests/unit/test_universe.py`, the same test file's
`test_future_constituents_cannot_enter_earlier_periods` proving the equivalent
property for membership itself.

A constituent is never silently dropped or silently kept: every exclusion
(`ExclusionReason.MISSING_INSTRUMENT_DATA`, `NOT_TRADABLE`, `CORPORATE_ACTION`)
is recorded on the `UniverseSnapshot` alongside the accepted constituents, so a
name missing from a given day's universe is an auditable decision rather than
a gap someone has to notice by its absence.

**Explicit scope boundary:** this module decides *eligibility*, not
*investability*. Liquidity filtering (`config.universe.min_avg_daily_value_inr`)
and trend/momentum scoring belong to `universe/stock_selector.py` (Phase 6,
still stubbed), applied *within* the eligible universe this module returns.
Keeping the two separate is what will let a later walk-forward run attribute
performance to selection vs. eligibility independently, rather than conflating
"was this a real historical constituent" with "did it pass today's factor
screen."

**Explicit data limitations** (also documented at the top of
`universe/universe.py`): correctness here is bounded by the completeness of
the membership and instrument datasets it is given — no logic in this module
can recover a historical add/remove date a vendor never recorded, and an
instrument whose delisting was never reflected in either the instrument
master's `status` or a corporate-action record will incorrectly remain
eligible. Snapshot persistence (so a later data correction cannot retroactively
change what a past rebalance "saw") is deferred to the storage layer, which is
not implemented yet.

## Feature engineering

`core/features/feature_engineering.py` computes the market-level inputs to
the HMM regime engine. The central discipline it enforces: the HMM is a
market-regime classifier, not a directional stock predictor, so every
feature is chosen for what it says about *how risky the market is*, not
*which way it is about to move* — see docs/ARCHITECTURE.md's earlier note
(B5) on the risk of signed-return features teaching the model direction.

Three mechanisms make the module's causality guarantee structural rather
than a convention someone has to remember:

- **`MarketFeatureInputs`** aligns NIFTY 50 and India VIX by inner join on
  session date — a date present in only one series is dropped, not
  interpolated, so a genuine vendor gap degrades the feature matrix (fewer
  rows) rather than fabricating a value.
- **`rolling_standardize`** and every feature's `compute` function use
  `pandas.Series.rolling(window=W, min_periods=W, center=False)` exclusively
  — trailing windows only, `min_periods` equal to the declared lookback so
  warm-up is NaN rather than a guess, and never a centered or expanding/
  full-history statistic. `tests/unit/test_feature_engineering.py` proves
  this behaviorally, not just by code inspection: it appends a wild future
  outlier to a computed series and asserts every prior value is bit-identical.
- **`FeatureDefinition`** pairs each feature's `compute` function with its
  own documentation (economic interpretation, calculation, required
  lookback, and a `known_at_decision_timestamp` flag) as structured data,
  not a comment that can drift out of sync with the code. `FeaturePipeline.audit()`
  turns this into a per-value provenance trail (`FeatureSnapshot`: name,
  timestamp, value, source observations, lookback) derived structurally from
  each definition's lookback and the row's position in the aligned date
  index — not hand-maintained per feature.

**Deliberately small and documented, not exhaustive.** Nine features, one per
bullet in the feature brief, each justified in
`core/features/feature_engineering.py`'s module docstring. Breadth stress and
generic overnight-gap/range-expansion features from
[SPECIFICATION.md section 5](SPECIFICATION.md#5-feature-engineering-for-the-hmm)
are explicitly deferred (documented, not silently dropped) — breadth needs
point-in-time index membership joined against every constituent's own price
history, which is a heavier lift only worth taking once Phase 6 stock-level
data flows exist.

**Explicit scope boundary:** `rolling_standardize` (a continuously-updating
trailing z-score, used inside two feature definitions) solves a different
problem from `core/features/feature_scaler.py`'s `CausalFeatureScaler` (fit
frozen parameters on one training window, apply them unchanged across an
entire out-of-sample walk-forward fold — Phase 5, alongside the HMM that
consumes it). Conflating the two would either bake a train/OOS split into
every single feature calculation, or lose the continuously-adapting
normalization that features like the VIX level actually need.

## Why the module boundaries matter for correctness, not just style

The specification's central risk — "the HMM must not secretly become a stock-price
prediction engine" — is not something a code review catches after the fact if the module
boundaries don't make the violation structurally awkward to write. Concretely:

- If `universe/stock_selector.py` could import `core/regime/`, a selection score could
  silently condition on regime state, and the walk-forward harness's "HMM vs. simple
  baseline" comparison (section 10.3) would no longer isolate the regime layer's
  contribution.
- If `risk/` could be bypassed by calling `broker/` directly from `portfolio/`, the
  "independent veto" claim in section 8 would be documentation, not a property of the code.

Phase 1 sets these boundaries up as package structure; later phases should keep them true
as import-boundary tests, not just as a convention people remember.
