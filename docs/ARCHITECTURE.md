# Architecture

This document maps [SPECIFICATION.md](SPECIFICATION.md)'s nine required separations onto
this repository's packages, states the dependency rules that keep the boundaries real (not
just directories), and records the small number of places where an implementation decision
had to resolve something the specification left ambiguous.

## Layer -> package mapping

The specification asks for nine separated concerns; this repository organizes them into
fourteen top-level packages. Two packages each cover two related concerns because they
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
| Application lifecycle / daily workflow | `orchestration/` |

`orchestration/` (Phase 11d) is the only package that is allowed to know about all the
others at once — it sequences them into a trading day and owns the process lifecycle. It
is deliberately the thinnest package in the repository: it contains no strategy
mathematics, and a test enforces that structurally (see the section on it below).

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
- `portfolio/` proposes; it does not decide. `portfolio_constructor.py` produces a
  `TargetPortfolio` of `TargetPosition`s, not orders. `risk/risk_manager.py` always
  evaluates that proposal and always has the final say — this is the structural form of
  specification section 8's "NON-NEGOTIABLE... independent veto". `risk/` never imports
  `core.regime`, so no regime label can influence a risk decision either.
- `risk/position_sizer.py` is the single place specification section 7.2's weight-based
  sizing formula and section 8.1's stop-distance risk-based sizing formula are reconciled
  into one final order quantity. No other module computes a final order quantity.
- `execution/` and `broker/` depend on `risk/`'s approved decisions; they never re-derive a
  target weight or quantity themselves.
- `broker/adapters/paper_broker.py`, `execution/position_tracker.py` and
  `risk/risk_state_builder.py` import from `backtest/` (`backtest.costs.CostModel`,
  `backtest.engine.market_liquidity_stats`) on purpose (Phases 10-11a, 11d): a paper fill
  must be priced through the identical Indian cost/slippage model a backtest fill uses,
  and a live risk state must measure liquidity the identical way a backtested one did, or
  neither pair is comparable. These are the places a later layer intentionally depends on
  an earlier one across the package boundary; it does not go the other way --
  `backtest/` never imports `broker/`, `execution/` or `orchestration/`.
- `orchestration/` may import from every other package; nothing may import from it. In
  particular `monitoring/health.py` must not import `orchestration/` even though the
  orchestrator is its main caller — the dependency runs orchestrator -> health checker,
  the same direction as every other module the orchestrator wires together.
- No module in `orchestration/` may import `numpy` or `pandas`. This is the structural
  form of "the orchestration layer must not contain strategy mathematics": every number in
  this system is computed by a dedicated module, so the layer that only sequences calls has
  no reason to reach for a numerical library.
  `tests/unit/test_orchestrator.py::test_the_orchestration_package_imports_no_numerical_libraries`
  enforces it.
- `validation/` (Phase 13) is the mirror image of `orchestration/`'s own rule: it may
  import from every other package (it exists to wire the whole system together and watch
  it run), and nothing in `broker/`, `execution/`, `risk/`, `portfolio/`, `universe/`,
  `core/`, `data/`, `monitoring/`, `backtest/` or `orchestration/` may import from
  `validation/` in return. A synthetic-market generator and a failure-injecting broker
  proxy belong nowhere near what a real deployment imports.
- `live/` (Phase 14) may import from `broker/`, `risk/`, `config/` and this repository's
  own test suite (as subprocesses); `broker/factory.py` deliberately does **not** import
  `live/` back, even though it is the one place `live/`'s own verdict matters most --
  see `build_broker`'s own docstring for why (an earlier layer depending on a governance
  layer above it would invert the intended direction; `preflight_confirmed` is a plain
  boolean instead, the same trust model `enable_live_trading` already uses). `app/cli.py`
  is the only thing that imports `live/` to actually run it.
- `broker/<broker_name>/` (Phase 10b: `broker/zerodha/`) is the only place any
  broker-specific detail -- endpoint paths, request/response field names, status
  vocabulary, WebSocket framing -- may appear anywhere in this codebase. `broker/factory.py`
  is the only module outside `broker/zerodha/` that imports from it (to construct one);
  strategy, risk, portfolio, and execution code depend only on `broker.base.Broker`.

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

**Minimum holdings vs. single-name cap vs. low-risk-tier exposure (genuinely under-specified).**
Section 7.1 says "start with 5–10 positions"; section 8 caps single-name weight at 15%;
section 7 targets 85–100% gross exposure in the calmest regime. Five positions at a 15% cap
can reach at most 75% gross exposure — short of that floor, let alone its ceiling.
`config/settings.yaml` sets `selection.min_holdings: 7` (7 x 15% = 105%, enough to
reach 100%), and `config/models.py`'s `Settings` model enforces
`selection.min_holdings * portfolio.max_single_name_pct >= regime_policy.low_risk.max_gross_exposure`
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

**Two regime vocabularies, deliberately (B6).** Section 7 names its exposure tiers Calm,
Normal, Elevated, Crisis-like — the same words section 6.1 uses for the HMM's own post-hoc
state labels. This repository keeps two separate enums instead of one:
`core.regime.hmm_engine.RegimeLabel` (calm/normal/elevated/crisis) is the HMM's relative,
per-model ranking of its own fitted states, for reporting only, exactly as Phase 5 requires.
`core.regime.allocation.AllocationRegime` (low_risk/normal_risk/high_risk/uncertain) is what
`RegimeAllocationEngine` actually acts on: an absolute, confidence-aware classification driven
by configured volatility thresholds, with a fourth category — UNCERTAIN — that has no
volatility-level analogue at all, since it answers "should this classification be trusted",
not "how risky does the market look". Reusing one vocabulary for both would have made it easy
to accidentally switch on `state.label` when computing exposure, exactly the failure mode
Phase 5's "names must not determine behavior" rule exists to prevent.
`config/settings.yaml`'s `regime_policy` section is keyed by the allocation vocabulary
(`low_risk`/`normal_risk`/`high_risk`/`uncertain`), not the spec's literal Calm/Normal/
Elevated/Crisis-like terms, and the configured bands are a fresh design for this phase
rather than a copy of section 7's example percentages — see `config/settings.yaml`'s comment
above `regime_policy:` for the actual numbers and reasoning.

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
| 5 | HMM engine, model registry, causal feature scaling | **Done** |
| 6 | Regime-aware allocation (`core/regime/allocation.py`, `regime_policy.py`, `baseline_policy.py`) | **Done** |
| 6b | Stock selection (`universe/stock_selector.py`, `factor_calculator.py`) | **Done** |
| 7 | Portfolio constructor (`portfolio/portfolio_constructor.py`) | **Done** (position sizer -- converting an approved target weight into a final order quantity -- is still Phase 7c) |
| 7b | Independent risk management (`risk/risk_manager.py`, `risk/circuit_breaker.py`, `risk/portfolio_risk_state.py`) | **Done** |
| 7c | Position sizer (`risk/position_sizer.py`) | Stubbed |
| 8 | Indian transaction-cost and execution-cost model (`backtest/costs.py`, `backtest/cost_schedule.py`) | **Done** |
| 8b | Backtest engine, performance metrics (`backtest/engine.py`, `backtest/performance.py`) | **Done** |
| 9 | Walk-forward validation (`backtest/walk_forward.py`) | **Done** |
| 9b | Stress testing (`backtest/stress_test.py`) | **Done** |
| 9c | Performance analytics (`backtest/comparison.py`, `backtest/robustness.py`, `backtest/report.py`) | **Done** |
| 10 | Broker interface, paper adapter, order manager (`broker/base.py`, `broker/adapters/paper_broker.py`, `execution/order_manager.py`) | **Done** |
| 10b | Zerodha Kite Connect v3 adapter, broker factory (`broker/zerodha/`, `broker/factory.py`) | **Done** |
| 10c | India API/algo operational controls (`config.models.ComplianceConfig`, `broker/compliance.py`, `docs/COMPLIANCE.md`) | **Done** |
| 10d | Production-grade order management (`execution/order_manager.py`'s `ExecutionStateMachine`, `execution/order_reconciler.py`, `execution/execution_journal.py`) | **Done** |
| 11a | Position tracking (`execution/position_tracker.py`) | **Done** |
| 11b | Position/cash reconciliation (`execution/reconciliation.py`) | **Done** |
| 11c | Restart recovery and broker reconciliation sequence (`execution/system_state.py`, `execution/startup.py`) | **Done** |
| 11d | Application lifecycle and daily workflow (`orchestration/`, `risk/risk_state_builder.py`, `monitoring/health.py`) | **Done** |
| 12 | Monitoring: terminal dashboard and alerts (`monitoring/snapshot.py`, `monitoring/terminal_dashboard.py`, `monitoring/alerts.py`) | **Done** |
| 12b | Operational analytics dashboard, production go-live gate (`monitoring/dashboard.py`) | Stubbed |
| 13 | End-to-end paper-trading validation (`validation/`) | **Done** |
| 14 | Live-trading safety gate (`live/`, `app/cli.py`, `docs/PRE_LIVE_CHECKLIST.md`) | **Done** (live order submission still disabled -- see below) |
| 15 | Production deployment and fail-closed hardening (`Dockerfile`, `deploy/`, `orchestration/fail_closed.py`, `app/service.py`, `app/health.py`, `docs/DEPLOYMENT.md`) | **Done** (deployed service does not yet run a trading day -- see below) |

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
and trend/momentum scoring belong to `universe/stock_selector.py`, applied
*within* the eligible universe this module returns.
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

## The regime engine

`core/regime/` answers one question -- *how risky is the market right now* --
and hands the answer to a risk budget. It never emits buy/sell signals, never
sees a security, and never learns what the portfolio holds.

The package is split so the most correctness-critical routine can be reviewed
without strategy context attached:

| Module | Role |
|---|---|
| `gaussian_hmm.py` | Pure inference math: parameters, Gaussian emissions, the forward filter, Baum-Welch, BIC/AIC |
| `hmm_engine.py` | Candidate selection, validation gates, measured state statistics, labelling |
| `model_registry.py` | Versioned JSON artifacts, approval gate |
| `allocation.py` | `AllocationRegime`, `AllocationTarget`, `RegimeAllocationEngine` — volatility-tier classification, confirmation, flicker, confidence scaling |
| `regime_policy.py` | `RegimePolicy` — allocation tier → configured exposure band (pure lookup) |
| `baseline_policy.py` | `RollingVolatilityBaseline` — the non-HMM comparison strategy |

**Filtered inference, never smoothed, never Viterbi.** The live regime call is
`P(state_t | observations_1..t)` and nothing else. Two standard routines
silently answer a different question, and both are easy to reach for:
forward-backward smoothing (`gamma`, which many libraries expose as
`predict_proba`) conditions on the *whole* sequence, and Viterbi returns a
full-sequence most-likely path — appending tomorrow's bar can retroactively
change which state yesterday was in. `forward_filter` is therefore a dedicated
implementation built from `filter_step`, whose signature takes only the
previous belief and one new observation; there is no argument through which
future data could arrive. Smoothing exists only inside Baum-Welch, named so it
cannot be mistaken for the live path. Tests assert the property behaviorally
by appending a wild future observation and checking earlier rows are
bit-identical, and separately assert that filtered and smoothed posteriors
actually *differ* — so the filter silently becoming a smoother would fail
loudly.

**Names are reporting; measurements drive behavior.** A fitted HMM's state IDs
are arbitrary (refit with another seed and "state 0" moves), and a label like
"crisis" is a name someone chose, not a measurement. So `StateStatistics`
carries annualized expected volatility, expected return, downside volatility,
empirical occupancy, expected duration and self-transition probability —
computed from the *actual return series* weighted by each state's
responsibility, not from the standardized feature space, which would be
uninterpretable. `RegimeLabel` is assigned afterwards by ranking states on
measured volatility, for reporting only. `RegimeAllocationEngine` (below)
never reads it — every allocation decision comes from `expected_volatility`
and `confidence` alone. A test permutes a fitted model's state IDs and
asserts every risk-relevant output is unchanged session by session while the
IDs demonstrably change.

**Selection fails closed.** Every (candidate state count × seed) pair is
fitted, then candidates are rejected for non-convergence, degenerate
(near-empty) states, or near-singular covariance before the lowest-BIC
survivor is chosen. If nothing survives, `fit` raises rather than returning a
"best effort" model: an unvalidated regime model is worse than none, because
it would size real positions from noise. Every candidate — accepted or not —
is retained with its rejection reason so a selection decision can be audited
later.

**Artifacts are JSON, not pickle.** A model file is loaded by the process that
places orders, so it must not be able to execute code; it must also be
readable by a human during an incident, and survive a library upgrade. Models
are a handful of states over a handful of features, so there is no size
argument against it. Saving a model does not make it live: `approve()` is a
separate step and `load_current_approved()` raises when nothing is approved,
rather than falling back to the newest fit.

## Regime-aware allocation (Phase 6)

`core/regime/allocation.py` turns a `RegimeState` history into an
`AllocationTarget` — a gross-exposure band and a point target within it. It
has no access to any security's price, score, or candidacy; it hands its
output to `portfolio/portfolio_constructor.py` (Phase 7, still stubbed),
which is the only place stock-level weights get decided.

**Never `state.label`.** `RegimeAllocationEngine` reads only
`expected_volatility` and `confidence` from each `RegimeState` — the same
discipline `hmm_engine.py` already enforces for its own labelling, applied
one layer up. `volatility_tier` classifies a bare number against two
configured thresholds (`config.allocation.low_risk_volatility_threshold`,
`high_risk_volatility_threshold`) into LOW_RISK / NORMAL_RISK / HIGH_RISK;
it can never return UNCERTAIN, because that category isn't a volatility
level — it's whether the classification itself should be trusted, which a
bare number can't answer. A test constructs two otherwise-identical regime
histories differing only in `RegimeLabel` and asserts identical
`AllocationTarget`s.

**Confirmation and flicker are pure functions of parallel arrays.**
`confirmed_tier_sequence` takes a list of raw tiers and confidences (not
`RegimeState` objects) and returns, position by position, the most recently
*confirmed* tier — `None` before anything has ever confirmed. A candidate
tier confirms after `hmm.confirmation_bars` consecutive agreeing
observations, or in one bar if confidence is at or above
`allocation.extreme_confidence_threshold` (docs/SPECIFICATION.md section 6,
"2 consecutive observations unless confidence is extreme"). While a
transition is unconfirmed, the *previous* confirmed tier is held rather than
acted on — tested directly by constructing a tier sequence with a dissenting
observation that reverts before confirming, and checking the confirmed
sequence never moved. `count_transitions` then measures how often the
confirmed tier actually changed within the trailing
`hmm.flicker_window_sessions`; too many changes forces UNCERTAIN regardless
of what the latest single reading says. Keeping both as pure functions over
plain arrays (rather than methods needing a fitted model) means the
confirmation and flicker logic is tested with hand-constructed sequences, not
only through a full HMM fit.

**Confidence scales continuously within a tier, not just on/off.** Once a
tier is confirmed and trusted, the final target is
`band.min + (band.max - band.min) * scale`, where `scale` maps confidence
linearly from `hmm.min_confidence` (→ 0) to 1.0 (→ 1), clamped to `[0, 1]`.
`AllocationTarget.__post_init__` re-validates `0 <= min <= max <= 1` and that
the target sits inside its own band — defense in depth on top of
`ExposureBand`'s own `Percent` fields, since floating-point arithmetic can
land a hair outside a mathematically-guaranteed range (`_build` clamps for
exactly this reason, caught by a property-style test sweeping volatility and
confidence across their full ranges).

**The baseline is not a toy.** `RollingVolatilityBaseline` exists because
"the HMM must beat a simpler alternative after costs" is not verifiable
without the alternative existing as running code
(docs/SPECIFICATION.md section 10.1, 10.3). It classifies trailing realized
volatility — using the exact same population-std, `sqrt(252)` annualization
as `core.features.feature_engineering`'s realized-vol feature, so a
comparison reflects the classification logic and not a different volatility
estimator — into the *same* configured bands `RegimeAllocationEngine` uses,
and returns the identical `AllocationTarget` shape, so a later walk-forward
comparison (Phase 8/9) can run the same downstream pipeline against either
one's output. It deliberately cannot express "uncertain" (`confidence` is
always reported as 1.0, `allow_new_positions` always True) — a documented
limitation, and itself part of what the HMM has to justify by doing better.

## Stock selection (Phase 6b)

`universe/stock_selector.py` decides *which* stocks receive the risk budget
`core/regime/allocation.py` has already set — it has no access to the current
regime, exposure target, or any risk state, and it places no orders. For V1
it is deliberately not a machine-learning model: six transparent,
individually-interpretable factors (`universe/factor_calculator.py`) combined
by configured, non-negative weights into one composite score, so every number
behind a ranking decision can be traced back to the raw price history that
produced it.

**Six steps, two auditable stages.** `StockSelector.select(as_of)` runs, in
order: (1) obtain the point-in-time eligible universe
(`universe.universe.UniverseProvider`), (2) drop instruments with fewer than
`selection.min_history_days` bars or no local data at all, (3) drop
instruments below `universe.min_avg_daily_value_inr` average traded value,
(4) compute factors from adjusted price history ending at `as_of`, (5)
cross-sectionally standardize each factor *within that date's surviving
candidate set* and combine by `selection.factor_weights`, (6) return the top
`selection.max_holdings`. Steps 1-3 are captured as one `CandidateUniverse`
(mirroring `universe.universe.UniverseSnapshot`'s pattern of recording every
exclusion with a reason, never silently dropping a name); steps 4-6 produce
ranked `StockScore`s.

**Why liquidity can safely use adjusted prices.** `data.models.DailyBar.adjusted()`
scales volume inversely to the price factor specifically so `close * volume`
(traded value) is invariant under adjustment — a 5-for-1 split scales price by
1/5 and volume by 5, and the product is unchanged. That single invariant means
every factor, including the liquidity/traded-value one, can fetch one
uniformly adjusted bar series instead of juggling raw prices for volume-based
factors and adjusted prices for everything else — a much easier bug to
introduce than it looks, since only the *momentum/trend/drawdown* factors are
obviously wrong on raw prices (a split reads as an 80% crash); a raw-price
liquidity factor would fail silently, just producing a traded-value number
that's off by the split ratio.

**Cross-sectional, not temporal, standardization.** `_cross_sectional_zscore`
compares candidates against *each other* on one date — a different problem
from `core.features.feature_engineering.rolling_standardize`, which compares
one instrument against *its own history* over time. Confusing the two would
either dilute the ranking signal (standardizing against irrelevant history) or
break the "no future information" guarantee if implemented carelessly across
dates. Exclusions happen before this step, so an instrument that fails
liquidity or history checks can never distort the surviving candidates'
relative scores — checked directly by a test that adds an obviously-illiquid
extra instrument to the universe and confirms the other candidates' scores
are bit-identical with and without it.

**Sign convention lives in one place.** Every factor is defined so "higher is
better" except `volatility`, which is reported as the real annualized number
(economically meaningful on its own) — the sign flip needed to treat "calmer
is better" as a ranking input happens exactly once, explicitly, where factors
are combined (`StockSelector._standardize`), not hidden inside
`factor_calculator.py`'s arithmetic.

**What's not here.** No fundamentals/quality factor: `data/` has no
fundamentals data source, so docs/SPECIFICATION.md section 7.1's allowance
for one "when clean, point-in-time fundamental data is available" does not
yet apply, and this phase deliberately does not fabricate one or build a new
data pipeline to get there. The gap is documented in
`universe/factor_calculator.py`'s module docstring, not silently absent.

## Portfolio construction (Phase 7)

`portfolio/portfolio_constructor.py` is where the two upstream decisions
finally meet: `core.regime.allocation.AllocationTarget` (how much risk the
regime layer permits, right now) and `universe.stock_selector.StockScore`
(which names are worth holding, and in what order) combine with
`config.models.PortfolioConfig`'s position limits into one `TargetPortfolio`
of `TargetPosition`s. "Market regime" and "risk budget" are the same input
here, not two — the regime *is* what determines the risk budget in this
system, so `construct()` takes exactly one `exposure_target: AllocationTarget`
parameter rather than a separate risk-budget argument. This module produces
weights only; it never computes an order quantity (`risk/position_sizer.py`,
Phase 7b) and never touches a broker.

**The weighting waterfall.** `construct()` runs a fixed sequence of pure
reductions — each step only ever shrinks a weight, never grows one, which is
what lets the whole pipeline converge in a single pass with no iteration: (1)
select the top `selection.max_holdings` ranked candidates, restricted to
currently-held names only when `exposure_target.allow_new_positions` is
False (the UNCERTAIN regime never opens a new position); (2) compute a raw,
risk-adjusted weight per candidate, `(score - floor) / volatility`, shifted
to be strictly positive first since composite scores are cross-sectional
z-score sums and can be negative; (3) apply a correlation penalty, halving
the lower-ranked half of any pair of selected candidates whose trailing
return correlation exceeds `portfolio.max_pairwise_correlation`; (4)
normalize to sum to 1.0 and scale by `exposure_target.target_gross_exposure`;
(5) clip against the single-name cap, then the liquidity cap
(`execution.max_participation_adv_pct` of average daily traded value), then
the sector cap, in that fixed order; (6) drop anything left below
`portfolio.min_position_weight_pct` to cash. A final defense-in-depth check
re-validates the result against every configured limit and raises
`PortfolioConstructionError` rather than returning a portfolio that quietly
violates its own configuration.

**Why capped weight is never redistributed.** When a cap trims a position,
the freed weight becomes cash — it is never handed to another candidate.
Redistribution would need another pass (a name that absorbs freed weight can
itself now breach a cap, cascading), and it would mean a single-name or
liquidity limit indirectly *increases* another position's risk, which is
backwards for a control meant to reduce it. Clipping to cash instead means
every cap violation degrades gracefully toward less exposure, never more,
and the waterfall provably terminates in one pass.

**Target vs. current vs. required trades.** `TargetPortfolio` is used
structurally for both roles — there is no separate `CurrentPortfolio` type —
because "what should be held" and "what currently is held" are the same
shape of fact at two different points in time. `required_trades(target,
current)` is a module-level function, not a `PortfolioConstructor` method: it
needs no config and no market data, only the two portfolios being diffed
into `RequiredTrade`s (BUY/SELL/EXIT/HOLD). No orders are sent from here;
sizing a `RequiredTrade` into an actual order quantity is `risk/position_sizer.py`'s
job, one layer down.

**A correlation-data gap degrades, it never crashes.** `_correlation_matrix`
fetches each candidate's adjusted price history independently and skips (not
raises) any instrument the market data provider has no data for at all,
exactly like one with too few overlapping bars — a data gap here must weaken
the correlation estimate, never take down the whole construction. This
window is deliberately a separate, independently configured lookback from
`StockSelector`'s own history check, so the two can legitimately disagree
about how much history is "enough" for their different purposes.

**What's not here.** No sector taxonomy exists in `data/`, so the sector cap
takes an optional `sector_map: dict[str, str]` parameter; an instrument with
no entry defaults to its own `instrument_id` as a singleton sector (the cap
becomes a no-op for it) rather than fabricating a classification. Converting
an approved target weight to a final order quantity is Phase 7c.

## Independent risk management (Phase 7b)

`risk/risk_manager.py`, `risk/circuit_breaker.py`, and
`risk/portfolio_risk_state.py` are the layer specification section 8 calls
"NON-NEGOTIABLE... independent veto": every `TargetPortfolio`
`portfolio/portfolio_constructor.py` produces passes through here before
anything downstream can act on it, and this layer never imports
`core.regime` -- it never sees a `RegimeState` or `AllocationRegime`, only
measured numbers. A regime that looks calm cannot talk this layer out of a
check.

**`PortfolioRiskState` is the one snapshot every check reads from** --
equity, per-instrument liquidity/staleness/spread facts (`PositionRisk`),
P&L and drawdown numbers, and two operational health flags (system health,
broker connectivity). `risk/` assembles none of this itself: the module has
no dependency on `data.interfaces.MarketDataProvider` or `broker.base.Broker`,
which keeps every check a pure, deterministic function of plain data and
keeps `risk/` from acquiring a dependency on `broker/` or `execution/`,
which sit downstream of it. Whatever orchestrates a trading cycle (the
backtest engine, later the live loop) is responsible for building this
snapshot.

**Two-stage evaluation.** `RiskManager.evaluate()` first asks
`CircuitBreaker.evaluate()` for the current `CircuitState`:

- **HALTED** rejects *every* proposed position outright, with no further
  checks run and no exception for a trade that would only reduce risk -- a
  halt driven by broker/system failure means no order is safe to route,
  sell or buy alike (specification section 19's kill-switch requirement:
  "make new order creation impossible, not merely discouraged").
- **REDUCED_RISK** tightens `max_gross_exposure`, `max_single_name_pct`, and
  `max_sector_pct` by `RiskConfig.reduced_risk_exposure_multiplier`, and
  forbids opening any position not already present in `current`. Fail
  closed: with no `current` portfolio supplied at all, every proposed
  position is treated as new and rejected, rather than silently skipping
  this protection because the caller omitted the comparison.
- **NORMAL** runs every check against the configured limits unmodified.

Only then do the portfolio- and position-level checks run: gross exposure,
position count, sector concentration, correlation concentration,
single-name exposure, liquidity/ADV participation, stale data, abnormal
spread, daily turnover, and the V1 long-only/no-leverage/no-borrowing
invariants -- re-checked here as defense in depth even though
`TargetPortfolio`/`TargetPosition` already enforce most of them
structurally, because this layer must never simply trust what it is handed.
A position with no matching `PositionRisk` entry is rejected outright
(`MISSING_RISK_DATA`) -- fail closed rather than approve something this
layer has no data to judge.

**Veto, never resize.** `RiskManager` never adjusts a weight; it approves or
rejects each proposed position outright (`RiskDecision.approved`).
Converting an approved weight into a final order quantity is
`risk/position_sizer.py`'s job (Phase 7c) -- conflating "should we do this
at all" with "how many shares exactly" would blur the line specification
section 8 draws between risk control and execution mechanics.

**Attribution.** A portfolio-level breach (gross exposure, daily turnover,
or a structural no-leverage/no-borrowing violation) rejects every proposed
position, since none of them is individually at fault. A position-count
breach rejects only the lowest-ranked excess positions
(`TargetPosition.rank`, ascending = best). A sector-concentration breach
rejects every position in the offending sector. A correlation breach
rejects only the lower-ranked member of the flagged pair -- the same
"who gets blamed" rule the portfolio constructor's correlation *penalty*
uses, except here it is an outright rejection, not a half-weight.

**The circuit breaker: NORMAL / REDUCED_RISK / HALTED, and why HALTED is
sticky.** `CircuitBreaker.evaluate()` checks broker connectivity and system
health first -- either failing forces HALTED outright, bypassing the loss
tiers entirely. Otherwise, each of three drawdown measures (`daily_pnl_pct`,
`rolling_pnl_pct`, `peak_to_trough_drawdown_pct`) is checked against its own
"halt" threshold, then its "reduce" threshold; crossing only the "warning"
threshold (`daily_loss_warning_pct`) is logged but leaves the state
unchanged. Once HALTED, the state is persisted to a single JSON file and
`evaluate()` returns that persisted status unchanged on every subsequent
call, no matter what the current numbers say -- there is no automatic
recovery path out of a halt, and a fresh `CircuitBreaker` pointed at the
same state file after a process restart picks up exactly where the last one
left off. The only way out is `CircuitBreaker.manual_reset()`, an explicit,
separately logged action requiring an operator and a reason. REDUCED_RISK
has no such stickiness: it clears back to NORMAL on its own once the
triggering metric recovers, since it was never a "critical" halt to begin
with. Every transition, and every manual reset, is logged through
`monitoring/logger.py` with structured `extra_fields` (`event`,
`previous_state`, `new_state`, `triggered_by`, `reason`).

## Indian transaction-cost model (Phase 8)

`backtest/cost_schedule.py` and `backtest/costs.py` are what
docs/SPECIFICATION.md section 9 means by "do this before trusting any
backtest": a zero-commission or single-flat-fee assumption silently
flatters every Indian delivery-equity strategy, since CNC trading carries
several independent statutory charges on top of brokerage, each set by a
different authority on its own schedule.

**Rates are versioned data, never a Python constant.** NSE, SEBI, and
CDSL/NSDL publish and revise their own levies independently of this
repository's release cycle, so `config/cost_schedules.yaml` -- not
`config/settings.yaml`, and definitely not code -- is the single source of
truth for brokerage/STT/exchange/SEBI/GST/stamp-duty/DP rates. Each entry
is a dated `CostSchedule`; `CostScheduleRepository.schedule_as_of(trade_date)`
selects the most recent entry on or before that date, the same
point-in-time pattern `data/corporate_actions.py` and
`core/regime/model_registry.py` already use for their own versioned
records. A rate change is *added* as a new dated entry, never edited into
an existing one -- editing in place would silently reprice every backtest
that already ran against it. A trade date earlier than the earliest known
schedule raises `MissingCostScheduleError` rather than guessing with the
oldest (or newest) rates on file.

**Deterministic vs. estimated, made machine-readable.** `TradeCost` is the
fully deterministic side -- brokerage plus every statutory levy, computed
from a `CostSchedule` and the trade's own quantity/price/side, bit-for-bit
reproducible given the same inputs. `ExecutionCostEstimate` adds the
estimated side on top: slippage, from the section 9.1 research model
(`max(min_bps, 0.5 * spread_bps + impact_bps(...))`), where `impact_bps`
uses a square-root participation model
(`impact_coefficient * volatility * sqrt(order_value / avg_daily_value)`).
`CostCategory` (`DETERMINISTIC` / `BROKER_DEPENDENT` / `EXCHANGE_DEPENDENT`
/ `ESTIMATED`) tags every `TradeCost` line item via
`TradeCost.CATEGORY`, so "which costs are deterministic, broker-dependent,
exchange-dependent, or estimated" is a queryable class attribute
(`TradeCost.components_by_category()`), not something only a docstring
claims. STT/SEBI fee/GST/stamp duty are `DETERMINISTIC` (uniform, set by
law, same regardless of broker or exchange); brokerage and DP charges are
`BROKER_DEPENDENT` (DP charges are levied by the Depository Participant,
typically the broker's own DP arm); exchange transaction charges are
`EXCHANGE_DEPENDENT` (NSE and BSE rates differ); slippage is `ESTIMATED` --
never a charged amount, this system's best guess at market impact.

**Side-dependent charges are modeled per leg, not averaged.** Delivery
(CNC) equity STT applies on *both* the buy and sell leg (unlike intraday,
which is sell-side only) -- this system is delivery-only long-only, so
both legs apply, and `CostSchedule` carries `stt_buy_pct`/`stt_sell_pct`
separately since the regulatory history has not always been symmetric.
Stamp duty, under the Finance Act 2019's unified exchange-collected regime,
applies on the buy leg only; DP charges apply on the sell leg only, since
they are levied only when securities actually leave the demat account.
GST applies to brokerage + exchange transaction charges + SEBI turnover
fee only -- never to STT or stamp duty, which are themselves taxes/duties
rather than a taxable service charge.

**Rounding matches a real contract note.** Every individual charge is
rounded to the nearest paisa (`ROUND_HALF_UP`, not Python's banker's-rounding
default) *before* being summed, and GST is computed on the already-rounded
brokerage/exchange/SEBI figures. This guarantees the displayed total is
always exactly the sum of the displayed line items -- computing GST or the
total from unrounded intermediates can silently produce a total that
doesn't match its own breakdown by a paisa.

**What's not here.** `net_pnl()` and `cost_pct_of_turnover()` are the only
arithmetic this module provides toward docs/SPECIFICATION.md's "backtests
must report gross P&L, costs, net P&L, cost as % of turnover" requirement
-- pairing buy and sell legs into a gross P&L in the first place needs
position tracking across time, which is `backtest/engine.py`'s job (Phase
8b -- see "The backtest engine and walk-forward validation" below), not
this module's; that engine's `PerformanceCalculator` is where gross/net
P&L and cost-as-%-of-turnover are actually reported end to end. The rates
shipped in `config/cost_schedules.yaml` are illustrative approximations
assembled from publicly documented STT/GST/SEBI-fee rules and a representative
zero-brokerage discount-broker plan -- verify against the current
NSE/SEBI/CDSL circulars and your actual broker's rate card before using
this for anything beyond research backtests.

## The backtest engine and walk-forward validation (Phases 8b-9)

`backtest/engine.py` and `backtest/walk_forward.py` are where every prior
phase gets wired together and run day by day, and where
docs/SPECIFICATION.md's central walk-forward requirement gets enforced
mechanically rather than by convention:

    historical data
        v
    training window            <- fit HMM + scaler on this window ONLY
        v
    fit HMM / factors / normalizers
        v
    freeze model                <- ScalerParams and FittedRegimeModel never change again
        v
    OOS simulation               <- BacktestEngine, test window only
        v
    advance window                <- roll forward by roll_step_sessions
        v
    retrain
        v
    next OOS period

**The twelve-step per-session pipeline.** For every session `T`,
`BacktestEngine.run()` does, in order: (1-2) `StockSelector.select(T)` --
causal by construction (Phase 6b); (3-4) not repeated per session -- the
regime/HMM step already ran once, up front, when `WalkForwardValidator`
built the fold's whole `exposure_targets` series (see "Filtering the
whole test window in one call" below); (5) that day's `AllocationTarget`
is read from the precomputed series; (6-7) `PortfolioConstructor.construct(...)`
combines the risk budget, rankings, and every configured limit into one
proposed `TargetPortfolio`; (8) `RiskManager.evaluate(...)` approves or
vetoes each position, and `_apply_risk_decisions` folds that into the
portfolio actually acted on; (9) the weight deltas between what is
currently held and that risk-approved target become hypothetical orders
(`OrderRecord`) -- still no price, no quantity; (10) orders are sized (a
simple weight-based `floor(notional / fill_price)`, not
`risk/position_sizer.py`'s stop-distance reconciliation, still Phase 7c)
and filled at the *next* session's opening price; (11) `CostModel` prices
every fill's full Indian cost breakdown, deducted from cash immediately;
(12) cash, holdings, and every recorded series are updated.

**No same-bar execution.** The signal for session `T` is built from
information available at `T`'s close -- the most recent price stock
selection and portfolio construction ever see for that decision. The
resulting orders execute at session `T + 1`'s **opening** price, never at
`T`'s own close or open. This is `docs/SPECIFICATION.md` section 10's
"signal at close of day T, execution at day T+1" rule, made concrete as
one specific, named fill assumption rather than left implicit. A order
that cannot fill at `T+1`'s open (the instrument didn't trade) rests,
searching forward a bounded number of sessions (`max_fill_search_days`) --
not a look-ahead, since the order's terms were already fixed before this
price is read; only *when* it fills is being resolved. A full
guarded-limit-order microstructure model (partial fills, price-guard
rejection) remains `execution/`/`broker/`'s job.

**Filtering the whole test window in one call.** `HMMRegimeEngine.filter()`
is forward-only: the state for row `t` depends on observations `1..t`
only, proven directly by a dedicated test (Phase 5) that appending later
rows never changes an earlier one. That property is exactly what lets
`WalkForwardValidator._hmm_exposure_targets` call `filter()` *once* over
an entire fold's test window instead of incrementally re-running it every
session -- mathematically identical, and far cheaper. It also runs
inference a little before `test_start` (`feature_warmup_buffer_days`) so
`RegimeAllocationEngine`'s confirmation/flicker logic has real context on
the test window's first sessions, instead of manufacturing an artificial
"not enough history" gap at every fold boundary. This is not a leak: the
model is already frozen by this point, so extending how far back it is
*run* changes nothing about what it *learned*.

**Rolling, not expanding, training windows.** Each fold's training window
is the same length as every other's (`backtest.training_window_sessions`),
sliding forward by `backtest.roll_step_sessions` each time -- never
growing. This isolates "did the regime genuinely change" from "did the
model just see more data than last time," and means a later fold's model
is never trusted more just because its training set happened to be
bigger.

**Five strategies, one identical downstream pipeline.**
`WalkForwardValidator.run_all_strategies` runs buy-and-hold, the
rolling-volatility baseline, the moving-average trend baseline
(`core/regime/baseline_policy.py::MovingAverageTrendBaseline`, the "simple
200-day moving-average risk filter" docs/SPECIFICATION.md section 10.1
requires), the HMM, and a shuffled-regime control over the *same* fold
sequence and the *same* stock selector, portfolio constructor, risk
manager, and cost model -- the only thing that ever differs is which
`AllocationTarget` series each one produces. This is what makes "the HMM
beats the simple baseline after costs" (section 10.3) a checkable claim
rather than an assumed one. The shuffled-regime control re-derives targets
from the *same* states the HMM actually produced for a fold, with their
date assignment randomly permuted (`WalkForwardValidator._shuffled_exposure_targets`)
-- the same total time spent in each regime, just reordered, to test
whether *when* the HMM called a regime mattered, not merely that it called
some regime some of the time.

**Folds chain into one continuous ledger.** Each fold's
`BacktestEngine.run()` starts from the *previous* fold's ending equity,
not a fresh `initial_equity` every time. Holdings themselves do not carry
across a fold boundary -- a retrain is treated as a flatten-and-reassess
point, since the new fold's stock selector may not even rank the same
instruments; this is a documented simplification, not an attempt to model
a real zero-cost liquidation.

**Dates, not `pandas.Timestamp`.** The Phase 1 stub this module replaces
typed every date as `pandas.Timestamp`; that was a placeholder guess made
before the rest of the system's date conventions existed. Every date here
is `datetime.date` instead, matching `TradingCalendar`,
`AllocationTarget.as_of`, and `RegimeState.as_of` -- corrected rather than
preserved for its own sake, the same treatment earlier phases gave
`ProposedWeight` and other early guesses that got superseded once the
surrounding design solidified.

**What's not here.** `risk/position_sizer.py`'s stop-distance
reconciliation (Phase 7c) remains stubbed; `backtest/stress_test.py`'s
failure-injection scenarios are Phase 9b, covered below. No per-fold model
persistence through
`core/regime/model_registry.py`: a walk-forward run fits and discards many
models in sequence for research purposes, which is a different use case
from the registry's single-approved-production-model workflow (Phase 5),
so each fold's model is kept in memory only, identified by a `model_id`
label (`core/regime/model_registry.py::build_model_id`) for audit, not
persisted to disk.

## Performance analytics (Phase 9c)

`backtest/comparison.py`, `backtest/robustness.py`, and `backtest/report.py`
turn `WalkForwardValidator.run_all_strategies`'s raw `PerformanceReport`s
into the comparison docs/SPECIFICATION.md section 10.3 actually asks for --
and are built around one structural commitment: **nothing in this layer
ever emits a verdict**. There is no `success: bool` field anywhere across
the three modules, on purpose. A high Sharpe ratio has repeatedly been
enough, in this domain, to convince someone a strategy "works" when it was
really a short sample, one lucky regime, or an unaccounted-for cost --
so this layer never lets a single number stand in for that judgment.

**`backtest/performance.py`'s own extensions.** `PerformanceReport` now
also reports `recovery_duration_days` (sessions from the *worst*
drawdown's own trough back to its prior peak -- a different question from
`drawdown_duration_days`, which is the longest single streak spent
anywhere below the running peak, not necessarily the worst one specifically),
`average_holding_period_days` (reconstructed by simulating a running
per-instrument quantity across `trade_log`'s fills, in execution-date
order, timing each *closed* round trip -- a position still open at the
end of the window is excluded, the standard convention), and
`pct_invested`/`pct_cash` (needs `cash_history`, an optional third
argument to `compute()` sourced from
`backtest.engine.BacktestResult.cash_history` -- not derivable from
`equity_curve` alone, which conflates cash and invested value into one
number). All three degrade to `float("nan")`/`None` when the inputs
needed to compute them are not supplied, never a fabricated zero.

**Regime and confidence breakdowns align positionally, not by date
label.** `BacktestResult.regime_history`/`confidence_history` are indexed
by *signal* date; `equity_curve` (and `cash_history`) by *execution* date
-- one trading day later, by this engine's own next-session execution
rule. The two therefore never share date labels to reindex against in the
first place, which was a real bug in this module's first draft: `by_regime`/
`by_confidence` now require `regime_history`/`confidence_history` to have
exactly one entry per `equity_curve` point in the same chronological
order, drop the series' last entry (nothing follows the final signal
date's regime within this window to attribute a return to), and attribute
each remaining entry to the return that follows it -- a purely positional
correspondence, deliberately independent of whatever date convention
either series happens to use.

**`backtest/comparison.py`** computes a `MetricDelta` (`hmm - baseline`,
uniformly signed regardless of whether the metric is higher-or-lower-is-better)
per metric, and attaches a set of programmatically-generated `caveats` to
every `BaselineComparison` -- never empty. One caveat is standing on every
comparison, unconditionally: a reminder to check the robustness
diagnostics before treating any outperformance as durable. The others are
conditional: a low HMM `trade_count` (statistical unreliability), a high
Sharpe alongside a large `max_drawdown` (a ratio that hides the loss an
investor actually lived through), an HMM return advantage that comes with
a *worse* drawdown than the baseline (return that may just be
compensation for extra risk, not skill), an infinite `profit_factor` (no
losing days in-window -- a short or one-sided sample, not an edge), and
transaction costs consuming an outsized share of gross P&L.

**`backtest/robustness.py`** runs a batch of already-built,
zero-argument variant closures (the caller assembles each one -- "build a
validator with this one setting changed, run it, return the aggregate
performance" -- since doing that requires config/data/calendar
dependencies this module deliberately has none of) and measures each key
metric's `relative_range` (`(max - min) / |mean|`) across variants.
`RobustnessDimension` names all seven required dimensions (parameter
perturbation, training window, rebalance threshold, transaction cost,
slippage, universe size, market period); six map directly onto existing
config knobs the caller already has (`backtest.training_window_sessions`,
`CostScheduleRepository`, `backtest.slippage_*`, universe/selection size,
the `[start, end]` window) and the seventh --
`BacktestEngine.min_rebalance_weight_delta`, new this phase -- reverts any
position whose weight would move by less than the threshold back to its
current weight (or never opens it) instead of trading it, trimming
needless turnover from tiny drifts as a real V1 option in its own right,
not only a robustness knob. `RobustnessReport.is_stable(metric, tolerance)`
is an explicit, named, opt-in threshold check, not an automatic verdict: a
"stable" metric by this check can still describe a bad strategy, and an
"unstable" one is not automatically disqualifying, just something that
needs explaining.

**`backtest/report.py`** formats what the other two modules computed --
CSV (one row per strategy or per robustness variant, one column per
metric, for a spreadsheet), and Markdown/HTML (every comparison keeps its
caveats printed directly beneath its numbers; the robustness section is
never silently omitted when robustness results are supplied). It performs
no analysis of its own.

## Stress testing (Phase 9b)

`backtest/stress_test.py` answers a different question from Phases 8b-9c:
not "how did the strategy perform", but "does the system still behave
safely when something goes wrong" -- across the 20 Indian equity-market
failure scenarios named in the phase brief (`StressScenario`), grouped
into market/data shocks, execution/infrastructure failures, and
model/decision failures. The central claim the whole module exists to
check: **risk controls must limit damage even if the HMM is wrong.**

**Fault injection without touching every collaborator by hand.**
`ShockedMarketDataProvider` wraps a real `MarketDataProvider` and applies
a deterministic `MarketShock` (price crash, open gap, volume collapse,
index spike, or outright unavailability, each scoped to an instrument set
and date range) on top of whatever the base provider returns, delegating
everything else unchanged. The harder problem this module had to solve:
`StockSelector` and `PortfolioConstructor` each capture their own
`market_data` reference at construction time, independent of
`BacktestEngine`'s -- so swapping only the engine's copy would leave
selection and construction reading the original, unshocked feed.
`StressTestContext` solves this by holding factory callables
(`stock_selector_factory`, `portfolio_constructor_factory:
Callable[[MarketDataProvider], ...]`) rather than fixed instances;
`StressTestContext.engine()` rebuilds all three collaborators fresh
against whichever provider a given scenario needs. A regression test
(`test_context_engine_uses_the_shocked_provider_for_stock_selection`)
guards this specifically: it asserts both collaborators' `market_data`
identity, then confirms selecting against a fully-unavailable feed
returns no candidates (this pipeline's fail-closed response to missing
data is candidate exclusion, not an exception).

**"Even if the HMM is wrong" is made concrete, not just asserted.**
`StressTestContext.full_exposure_targets()` builds a trivial "always
fully invested" exposure signal -- deliberately not the HMM -- fed
through a real market shock. `HMM_REGIME_MISCLASSIFICATION` and
`SUDDEN_MARKET_CRASH` both run this signal against a crashing market: the
circuit breaker, which watches realized P&L and never consults the
regime label, must still halt or reduce risk. This is the one property
the module cannot compromise on, because it is the phase's actual
requirement rather than a nice-to-have coverage checkbox.

**Fail-closed infrastructure failures are a pass, not a crash.** When
`BacktestEngine.run()` itself raises `BacktestEngineError` -- missing
mark-to-market data, an empty or malformed signal-date sequence -- the
suite reports `system_failed_closed=True` via `_failed_run_result` rather
than propagating the exception. A stress test's job is to observe a
failure mode, not to crash alongside it; refusing to proceed on bad data
is the system working as designed, matching every earlier phase's
fail-closed convention.

**Duplicate-order prevention closed a real input-validation gap.**
`BacktestEngine.run()` previously checked only that `signal_dates` was
ascending, which a duplicate adjacent date trivially satisfies (it sorts
into itself) -- a duplicated broker order response would replay the same
session's decision twice, past the check. The phase now rejects
`signal_dates` containing duplicates outright, tightening the engine's
own input validation, not just a test fixture.

**Scope is stated honestly, not implied.** No live broker, order
manager, or database exists yet (Phases 7c/10/11 are still stubbed), so
`BROKER_API_OUTAGE`, `PARTIAL_FILL`, `ORDER_REJECTION`, `DELAYED_FILL`,
`APPLICATION_RESTART`, and `DATABASE_FAILURE` each test the specific
mechanism that already exists for that failure mode instead of
simulating a broker that isn't built: `PortfolioRiskState.broker_connected`
for outages, `BacktestEngine._apply_fill`'s ledger arithmetic for
partial-size fills, `_execute`/`_next_open` returning `None` for
undeliverable orders, `max_fill_search_days` for delayed fills, and
`CircuitBreaker`'s on-disk JSON state file (corrupted directly, to
simulate a database/disk failure) for restart/persistence scenarios.

**Monte Carlo: deterministic despite being randomized.**
`MONTE_CARLO_SCENARIOS` names the 8 scenarios (crash, gap, VIX spike,
liquidity deterioration, wide spread, HMM misclassification, wrong
ranking, sudden drawdown) whose severity is meaningfully continuous
rather than binary. `StressTestSuite.run_monte_carlo` seeds trial `i`
with `seed + i` and uses `numpy.random.default_rng` for magnitude
sampling, so a fixed seed reproduces the exact same 100+ trials byte for
byte. `test_monte_carlo_crash_scenario_reliably_triggers_risk_controls`
checks the requirement statistically rather than in one hand-picked case
-- across a uniform 10%-40% crash-magnitude sweep, most trials trigger
some risk-control response. The bar is "most", deliberately not "all":
at this environment's single-name weight cap, a mild ~10-20% cut
legitimately stays under the daily-loss reduce threshold, so the risk
layer correctly staying quiet on the mildest sampled shocks is expected
behavior, not a gap. `risk_controls_fired` itself checks both an
explicitly rejected decision *and* the circuit breaker having left
`CircuitState.NORMAL` at any evaluated decision -- catching a
REDUCED_RISK state that never rejected a new order because the affected
position was already open, not only an outright halt.

**What's not here.** No scenario asserts a specific dollar loss ceiling;
the requirement is that controls *engage* under stress, not a promised
maximum drawdown, which depends on position sizing decisions outside
this module's scope. Corporate-action and exchange-holiday scenarios
reuse `CorporateActionProvider`/`TradingCalendar` directly rather than
inventing a parallel data path, consistent with the project's
reuse-not-reimplement discipline for this phase (`_drawdown_stats`,
`_recovery_duration`, `RiskManager`, and `CircuitBreaker` are reused
from Phases 9 and 7b outright, not reimplemented).

## Paper trading engine (Phase 10-11a)

`broker/base.py`'s `Broker` ABC, `broker/adapters/paper_broker.py`'s `PaperBroker`,
`execution/order_manager.py`'s `OrderManager`, and `execution/position_tracker.py`'s
`PositionTracker` together answer this phase's actual requirement: strategy code must never
be able to tell whether it is talking to a paper broker or a future live one, and an order
request that gets sent twice must never become a position twice.

**Idempotency is enforced at two independent layers, not one.** `OrderManager.create()` is
keyed by a caller-supplied `idempotency_key` (a trading intent — "this signal, this
instrument, this side" — not the same thing as `client_order_id`, which `OrderManager`
generates fresh): a resubmitted identical request returns the already-created order and
never calls the broker again. `PaperBroker.place_order()` separately deduplicates by
`client_order_id` itself, mirroring how a real broker treats a repeated `clOrdID` — this
matters because a caller could reach the broker directly, bypassing `OrderManager`, and
because it is the realistic place a live adapter would enforce the same guarantee. A
`client_order_id` reused for a *different* order payload is treated as a caller bug and
raises, rather than silently keeping whichever payload arrived first.

**The order-state vocabulary is this system's own, not section 12.1's literal list.**
`execution.order_manager.OrderState` has ten values — `CREATED`, `SUBMITTED`, `OPEN`,
`PARTIALLY_FILLED`, `FILLED`, `CANCEL_REQUESTED`, `CANCELLED`, `REJECTED`, `EXPIRED`,
`UNKNOWN` — broader than docs/SPECIFICATION.md section 12.1's sketch, because treating a
partial fill as a first-class event (section 12.2) needs `OPEN` and `PARTIALLY_FILLED` to be
distinct states, and a resting order that never fills needs a terminal state of its own
(`EXPIRED`) rather than staying `SUBMITTED` forever. `broker.base.BrokerOrder.status` is a
raw string every adapter reports in its own vocabulary; `OrderManager` is the one place that
turns it into this typed state (`_adopt_broker_order`), so no caller above it pattern-matches
on adapter-specific strings. The allowed-transition table
(`execution.order_manager._ALLOWED_TRANSITIONS`) deliberately lets `CANCEL_REQUESTED` still
resolve to `FILLED`/`PARTIALLY_FILLED` — a cancel request can race a fill already in flight
broker-side, and a real broker does not guarantee the cancel wins that race.

**`UNKNOWN` is resolved by querying, never by retrying.** `OrderManager.submit()` catches
any exception `Broker.place_order` raises (a lost response, a timeout) and transitions the
local record to `UNKNOWN` instead of propagating or blind-retrying — exactly
docs/SPECIFICATION.md section 12.1's "UNKNOWN state -> RECONCILIATION REQUIRED", made
concrete. `OrderManager.handle_ambiguous_response()` resolves it by calling the new
`Broker.get_order(client_order_id)` (added to the ABC this phase specifically to make this
resolution possible — `get_open_orders()` alone cannot distinguish "filled", "rejected", and
"cancelled" for an order that is no longer open). `PaperBroker` itself never produces an
ambiguous response on its own (it is synchronous and in-process, so nothing is ever actually
lost) — the test suite exercises this path by wrapping a real `PaperBroker` in a small
`_FlakyBroker` test double that drops exactly one response after the real broker has already
processed the order underneath, proving `OrderManager` recovers the broker's true state
rather than losing track of what actually happened.

**`PaperBroker` prices every fill through the identical model a backtest fill uses.**
`backtest.engine.market_liquidity_stats` (extracted this phase from
`BacktestEngine._market_stats`, which now delegates to it, so both callers share one
implementation rather than two copies of the same formula) supplies the trailing
avg-daily-value/volatility estimate; `backtest.costs.CostModel.estimate_execution_cost` prices
the fill exactly as it would in a backtest. The one deliberate improvement over the
backtest: `PaperBroker` prices spread from a real live `Quote`
(`data.interfaces.MarketDataProvider.get_quote`), not the single assumed constant the
backtest falls back on for lack of real historical bid/ask data.

**Fills are matched against real depth, not assumed infinite liquidity.** A marketable limit
order fills against the quote's own `bid_quantity`/`ask_quantity`, capped per match by
`paper_trading.max_fill_participation_pct` — an order larger than one match's share of the
book rests `PARTIALLY_FILLED` and needs a later `PaperBroker.process_resting_orders()` call
(the periodic "tick" a live/paper trading loop is expected to make) to fill further as fresh
quotes arrive. A quote with no depth information at all fills in full, a documented
simplification in the same spirit as this codebase's other stated-not-hidden simplifications
(`backtest/engine.py`'s assumed spread constant, `PaperTradingConfig.order_expiry_seconds`
being duration-based rather than session-aware).

**Broker-side checks are defense in depth, not a re-derivation of risk's decision.**
`PaperBroker._validate()` rejects an unsupported order type (NSE algo orders may not use
market orders — section 13), a stale or crossed quote, a limit price outside
`execution.order_price_guard_bps` of the mid, a sell beyond the held quantity (no shorting,
structurally re-checked here as it is everywhere else in this codebase), and a buy beyond
available cash. None of this re-derives a target weight or quantity — by the time an order
reaches `PaperBroker`, `risk.risk_manager.RiskManager` has already approved it; this layer
only checks whether *execution* itself can still proceed safely.

**`PositionTracker` is the one portfolio-state shape both today's paper broker and a future
live adapter produce.** Weighted-average cost basis on a buy, realized P&L on a sell (using
that same cost basis, unaffected by the sell itself — the standard convention), and
unrealized P&L from the latest mark — all computed the identical way regardless of which
`Broker` produced the fill. Realized P&L survives a full close-and-reopen of a position
(never reset just because quantity returned to zero), and selling more than is held raises
rather than silently going short, the same V1 long-only invariant enforced structurally
elsewhere in this codebase. `PaperBroker.get_account()`/`get_positions()` proactively
mark every held instrument to its current quote before reporting, so `unrealized_pnl`
reflects the market a position is *currently* held in, not the price it last traded at.

**What's not here.** Corporate-action identity changes and forced exits
(`PositionTracker.apply_corporate_action`/`force_exit`) are not this phase's concern — this
phase's `PositionTracker` is the source of truth for a single run; reconciliation against a
broker's own state (`execution/reconciliation.py`, Phase 11b) and the restart-recovery
sequence that calls it (`execution/startup.py`, Phase 11c) are what cross-check it against
anything external. No `BacktestBroker` adapter was built to retrofit
`backtest.engine.BacktestEngine` onto the `Broker` interface: the engine's own historical
next-session-open execution model is a different, already thoroughly tested design, and nothing
in this phase's requirement needed it rewritten — the requirement was that `PaperBroker` and
a future `LiveBroker` be interchangeable beneath the same interface, which `Broker` already
guarantees structurally.

## Broker abstraction and the Zerodha Kite Connect adapter (Phase 10b)

This phase's own instruction was explicit and is worth restating exactly
because it shaped the whole design: *do not guess the broker API*. Before
any adapter code was written, the user chose Zerodha Kite Connect v3, and
every endpoint path, request parameter, response field, status string,
header, and WebSocket byte layout in `broker/zerodha/` was then fetched
from Zerodha's own published documentation
(https://kite.trade/docs/connect/v3/) and verified against the live pages
-- not recalled from training data, not inferred from a similar broker's
API, and not filled in with a plausible-looking guess where the docs were
thin. Where the docs did not give a precise answer (index-instrument
WebSocket packets, a dedicated server-time endpoint), that gap is
documented at the point it matters in `broker/zerodha/kite_broker.py` and
`kite_ticker.py`'s own module docstrings, not papered over.

### The generic interface grew to earn its "generic" label

Phase 10's original `Broker` ABC (built against `PaperBroker` alone) was
extended, not replaced, to actually cover every required capability
against a second, real adapter:

- `BrokerCapabilities` (new) lets a caller ask what an adapter supports
  *before* calling it, rather than discovering a gap by catching
  `BrokerCapabilityError` -- `KiteBroker.capabilities()` reports the
  authenticated account's own actual exchange/product/order-type
  entitlements (fetched once from `GET /user/profile` at `authenticate()`
  time), not just what Kite the API supports in the abstract.
- `Account` was renamed to `BrokerAccount` and gained `account_id`, so
  "account information" and "funds" -- two separate items in this
  phase's required-capabilities list -- are both satisfied by one
  `get_account()` call and one type, rather than inventing a seventh
  dataclass this phase's own instructions did not ask for.
- `BrokerFill` (new) and `Broker.get_trades()` answer "trade/fill
  history" -- a fill is not the same event as an order (one order can
  have many fills), a distinction `PaperBroker` already modeled
  internally (`PaperFill`) but had never exposed through the generic
  interface until now.
- `Broker.get_order`'s existing "regardless of whether it is still open"
  contract is honored by `KiteBroker` even though Kite's own API has no
  client-side-ID lookup at all (see "The client_order_id bridge" below).
- `Broker.subscribe_market_data` (new) answers "market data subscription
  where available" literally: it is `KiteBroker`'s and `PaperBroker`'s
  *only* required-but-optional capability -- `PaperBroker` raises
  `BrokerCapabilityError` (no live feed exists in paper mode, by design),
  `KiteBroker` raises the same exception only when no `KiteTicker`
  transport was actually configured, and delegates to it otherwise.
- `HealthStatus` gained `session_active`/`login_time` so "connection
  health" and "broker time/session information" -- two more separate
  required-capabilities list items -- are both satisfied by one
  `health_check()` call. Kite Connect has no dedicated server-time
  endpoint (verified absent, not assumed absent); session state is the
  honest substitute, documented as such rather than left unexplained.

### The client_order_id bridge

`Broker`'s contract promises callers can query any order by the
client-generated `client_order_id`. Kite's API has no such concept --
`POST /orders/:variety` returns only Kite's own broker-assigned
`order_id`, and every other order endpoint addresses by that same ID.
`KiteBroker` bridges the two with an in-memory `client_order_id ->
kite_order_id` map populated at `place_order` time (plus a truncated
`client_order_id` in Kite's own `tag` field, a human-visible breadcrumb
in the Kite order-book UI, not the actual lookup mechanism). This mapping
is **not persisted** -- an order this process did not place itself has no
known `client_order_id`, and `get_order`/`get_trades` fall back to
reporting Kite's own `order_id` as the identifier in that case,
documented in `kite_broker.py`'s own module docstring rather than hidden.
Closing this gap with real reconciliation against the broker's own state
at startup is Phase 11b's job, not this adapter's -- consistent with
every other place this codebase has already deferred reconciliation.

### Live trading is gated twice, independently

Per this phase's explicit instruction ("default mode must remain PAPER";
"do not enable live trading"), nothing in this repository can place a
real order without two separate, explicit confirmations, neither
sufficient alone:

1. `broker/factory.py`'s `build_broker()` only ever constructs a
   `KiteBroker` when `settings.execution.mode == "live"` *and* the caller
   passes `enable_live_trading=True` explicitly -- a config-file value by
   itself (easy to leave set from a prior session, easy to typo) is
   deliberately not enough.
2. `KiteBroker` itself defaults `enable_live_trading=False` in its own
   constructor and checks it again, independently, inside every
   order-placing method (`place_order`, `modify_order`, `cancel_order`,
   `close_position`, `close_all_positions`) before making any network
   call -- so even a caller that bypasses the factory and constructs
   `KiteBroker` directly cannot place a real order by accident.

Credentials (`BROKER_API_KEY`/`BROKER_API_SECRET`) are read from the
environment inside `build_broker()`, never from `settings.yaml` -- the
`.env.example` placeholders these names came from were planted in Phase
1, anticipating exactly this phase. Read-only calls (account, positions,
orders, quotes, health) are not gated -- authenticating and observing a
real account carries none of the risk placing a real order does.

### What Kite's own docs left genuinely ambiguous, and how that was handled

- **WebSocket index-instrument packets.** The verified byte table covers
  equity LTP/quote/full packets precisely; Kite's own docs describe the
  shorter index-quote-mode variant only in prose, not a byte table.
  `kite_ticker.py`'s `decode_binary_ticks` implements only the verified
  equity layouts and documents the index gap explicitly rather than
  guessing a layout -- this system's regime features already read
  NIFTY/VIX through `data.interfaces.MarketDataProvider` (Phase 4), not
  through this streaming path, so the gap costs nothing today.
- **No real WebSocket transport ships.** `KiteTickerTransport` is a
  `Protocol`; connecting a real `wss://` socket needs a WebSocket client
  this codebase does not otherwise depend on, and this phase's own scope
  is explicit that nothing should touch a real connection. URL/message
  construction and binary decoding (the parts Kite's docs actually
  specify precisely) are fully implemented and tested; opening a live
  socket is left to whoever wires in a real transport when live
  streaming is actually enabled -- documented in `kite_ticker.py`'s
  module docstring, not silently absent.
- **No automated browser login.** Kite's login step is a human completing
  a form at `kite.zerodha.com` and Kite redirecting back with a
  `request_token`; there is no documented API for automating it, and
  attempting to script it would mean interacting with an undocumented,
  unstable surface -- exactly what this phase's instruction forbids.
  `KiteBroker.authenticate()` takes the resulting `request_token` as
  input; `KiteBroker.login_url()` builds the URL a human visits to get
  one.

## India API/algo operational controls (Phase 10c)

`config.models.ComplianceConfig` and `broker/compliance.py` answer this
phase's own instruction as directly as the phase name states it: "do this
before connecting the live account." The user's exact closing instruction
was also explicit about method -- "Do not make regulatory assumptions" --
so before any line of this section's code was written, the applicable NSE
circular, the SEBI framework it implements, and Zerodha's own published
material on it were fetched and read (`docs/COMPLIANCE.md` is the full
research record, including what could **not** be verified and why). This
section is the design that research produced, not a description of an
assumption.

**Three independent gates, not one.** `broker/factory.py` already required
`execution.mode == "live"` and an explicit `enable_live_trading=True`
(Phase 10b). This phase adds a third: `ComplianceGate(settings.compliance)`
is constructed before a `KiteBroker` ever is, and its constructor is
itself the "refuse to operate" gate -- `broker_authorization_confirmed`
not `True`, `static_ip_primary` still the `"0.0.0.0"` placeholder, or
`algo_identifier` still `"UNSET"` each raise `ComplianceError` immediately,
before any network call is even possible. `settings.yaml`'s own default
`compliance:` section ships with exactly these placeholders -- the default
configuration is *engineered* to fail this gate, not merely undocumented
as unready.

**Every required control maps onto a specific, cited mechanism**, not a
generic "compliance" checkbox:

| Requirement | Mechanism |
|---|---|
| Missing static IP / algo identifier | `ComplianceGate.__init__` refuses to construct |
| Missing broker authorization | Same -- `broker_authorization_confirmed` |
| Unsupported order type / validity | `ComplianceGate.check_order_type`/`check_validity`, checked independently of (in addition to, not instead of) `broker.zerodha.kite_mappings.SUPPORTED_ORDER_TYPES` |
| Expired authentication | `ComplianceGate.check_session`, computed from `Broker.health_check().login_time` (Phase 10b) against `session_max_age_hours` |
| Order-per-second limit | `ComplianceGate.check_rate_limit`, a client-side sliding one-second window in front of Kite's own server-side 10 OPS ceiling |
| Algo-order tagging | `ComplianceGate.check_order` overwrites `BrokerOrder.tag` with `algo_identifier` before any order reaches an adapter |

**Never substitutes a prohibited value for an allowed one** -- every one of
the checks above either returns the (possibly tag-rewritten) order
unchanged or raises `ComplianceError`; there is no code path that swaps a
rejected `order_type`/`validity` for a permitted one and proceeds. This is
tested directly (`test_check_order_never_substitutes_a_prohibited_type_it_just_refuses`):
the order passed in is asserted unmodified after the raise, because
nothing inside the gate ever had the chance to modify it before deciding
to refuse.

**`ComplianceGuardedBroker` applies the gate uniformly, including to
`close_position`.** A naive wrapper that simply delegated
`close_position`/`close_all_positions` to the inner adapter would silently
bypass every check above -- both `PaperBroker` and `KiteBroker` build their
closing sell order and call their *own* internal `place_order`, never the
wrapper's. `ComplianceGuardedBroker` therefore reimplements the identical
order-construction logic (get the position, get a quote, build a marketable
SELL `BrokerOrder`) and routes it through its own `place_order`, so a
position close is tagged and gated exactly like any other order --
documented in the class's own docstring as *why* it duplicates roughly ten
lines already present twice elsewhere, rather than leaving the duplication
unexplained.

**What Phase 16's own research could not verify, and how the code reflects
that honestly:** `docs/COMPLIANCE.md`'s "What was not verified" section
lists, among other gaps, that Kite Connect's own API documentation --
re-checked directly during this phase -- makes no mention of the NSE-cited
algo-tag digit format at all. Rather than hard-code an unconfirmed byte
pattern into `KiteBroker`, the tag mechanism this system actually uses is
the one field Kite's docs *do* document (`tag`, generic and pre-existing
since Phase 10b) carrying `ComplianceConfig.algo_identifier` -- an honest,
stated choice pending the broker's written confirmation, not a guess
dressed up as a verified fact. The same section documents that the
5-year default for `audit_log_retention_years` is drawn from general SEBI
stock-broker record-keeping norms, not a retail-algo-specific primary
source, for the same reason: `docs/COMPLIANCE.md` is where a reader finds
out which numbers in this codebase are broker-confirmed and which are
still open questions, rather than that distinction being lost once the
research becomes code.

## Production-grade order management (Phase 10d)

Phase 14 built a working order lifecycle (`OrderManager`, its ten-state
`OrderState`, idempotent `create()`, and `handle_ambiguous_response` for a
lost-response order). This phase's brief asked for that to become
production-grade: an explicit, independently-testable state machine; a
dedicated reconciler with stale-order, timeout, and broker-reconnect
handling on top of single-order resolution; an append-only journal so
"every signal must be traceable"; and — stated as the one CRITICAL
requirement — proof that a broker accepting an order but losing the
response never causes an automatic resubmission.

**`ExecutionStateMachine` is now a standalone class, not an inline
table.** The `_ALLOWED_TRANSITIONS` dict `OrderManager` used internally
since Phase 14 is unchanged in spirit but now owned by its own class
(`validate_transition`, `is_terminal`, `is_open`, `allowed_next_states`),
composed by `OrderManager` rather than embedded in it — "what states can
an order legally move through" is readable and testable independently of
"how this system actually manages one," which is what the phase brief's
"build an explicit order state machine" asked for literally.

**The transition table itself had to widen, and the reason is a genuine
correctness finding, not a convenience.** Every non-terminal,
non-`CREATED` state can now reach *any* terminal state directly, on top
of its normal forward-progress edges (`_ALLOWED_TRANSITIONS`'s own
comment explains this in place). This was discovered, not designed in
advance: an integration test reconciling a locally-`OPEN` order that the
broker reported as `CANCELLED` (cancelled through another channel while
disconnected) failed against the original table, which only allowed
`OPEN -> CANCEL_REQUESTED -> CANCELLED` -- the step-by-step path *this
system's own actions* take, which is not the only path a real broker's
state can arrive at `CANCELLED` by. Reconciliation's entire premise is
that the broker is authoritative and may reflect events this system never
individually observed while disconnected; a state machine that could only
accept locally-driven transitions would defeat that premise the first
time reality diverged from the happy path. This is exactly the kind of
result the phase's own closing instruction ("run failure-injection
integration tests") exists to surface — a test that only used the
already-passing paths would never have found it.

**`execution/order_reconciler.py`'s `OrderReconciler`** builds a full
reconciliation sweep on top of `OrderManager.handle_ambiguous_response`
(kept, unchanged, and still what single-order `UNKNOWN` resolution
delegates to):

- **stale-order detection** (`detect_stale_orders`) flags `OPEN`/
  `PARTIALLY_FILLED` orders this system hasn't heard an update about
  recently, paired with `refresh_from_broker` to actually resync them.
- **order timeout** (`detect_and_handle_timeouts`) treats an order stuck
  in `SUBMITTED` past a configured age exactly like a submission call
  that raised -- transitioned to `UNKNOWN`, never assumed successful,
  never silently retried.
- **broker reconnect** (`reconcile_after_reconnect`) is the one entry
  point a live/paper trading loop calls after (re)establishing a
  connection: times out stuck submissions, resolves every `UNKNOWN`,
  refreshes every stale order, and surfaces orphans (broker-reported open
  orders with no local record -- flagged for manual review, never acted
  on automatically, since this system cannot safely manage an order whose
  signal/risk lineage it does not know).
- **`RetryPolicy`** is deliberately narrow: it retries a read (`get_order`,
  `get_open_orders`) up to a bounded number of times with backoff, and is
  never used for `place_order`'s initial submission -- `OrderManager.submit`
  does not import or reference `RetryPolicy` at all, which is the
  structural form of "no unsafe blind retry" rather than a comment saying
  so.

**Traceability is structural, not a convention a caller has to
remember.** `OrderManager.create()` now requires `signal_id` and
`risk_decision_id` (both opaque strings from this module's point of
view -- a future live trading loop mints them when it calls
`RiskManager.evaluate` and decides to act on the result); every
`create`/`transition` call writes to an `execution.execution_journal.ExecutionJournal`
this manager owns, so an order created without a traceable identity
chain is not possible to construct, not merely discouraged. A second,
new protection sits on top of the existing idempotency-key mechanism:
reusing a `signal_id` under a *different* `idempotency_key` raises
`DuplicateSignalError` rather than silently creating a second order --
a legitimate retry reuses the same key; a fresh key for an already-seen
signal is treated as a caller bug (a retry path that regenerated its key
instead of reusing the original), exactly the kind of duplicate this
phase's brief asks to detect beyond simple idempotency-key matching.

**What "fills" and "position" mean in the traceability chain.** A
`FILL_OBSERVED` journal entry is written whenever `transition` reports a
higher `filled_quantity` than the order previously had -- satisfying
"fills" directly. "Position" is satisfied structurally rather than
re-derived: every fill this journal observes is the identical fill
`PaperBroker`/`KiteBroker` already apply to `PositionTracker` (Phase
14/15); correlating this journal against `PositionTracker`'s own state by
timestamp and instrument to answer "which position resulted from which
order" as one automated query is not built this phase -- documented as
the honest boundary in `execution_journal.py`'s own module docstring,
not left for a reader to assume was covered.

**The CRITICAL scenario, proven end to end against a real broker, not a
stub.** `tests/unit/test_order_reconciler.py`'s
`test_critical_scenario_broker_accepts_order_but_response_is_lost` runs a
real `PaperBroker` (real cost model, real position tracker) wrapped in a
broker double that drops exactly one `place_order` response after the
real broker has already filled the order underneath -- the literal
"accepts an order but times out before returning the order ID" case. It
asserts, in order: `submit` does not raise (marks `UNKNOWN`); the broker
was called exactly once; the order was genuinely filled broker-side
already (proving the ambiguity is real, not something this system could
have inferred on its own); the local record reads `UNKNOWN` *before*
reconciliation runs (a caller checking state has no way to mistake this
for a known outcome); `OrderReconciler.resolve_unknown` determines the
true state by querying the broker; and afterward there is still exactly
one fill and one position -- the ambiguous submission was never
duplicated by either the original attempt or the reconciliation that
followed it. A second test proves reconciling an already-resolved order a
second time is a harmless no-op, not a second query or a second write.

## Restart recovery and broker reconciliation (Phases 11b-11c)

Phase 11b had been left stubbed since Phase 7b's own note that
position-level reconciliation "belongs with a broker connection, not
before one exists" — order-level reconciliation arrived first, in Phase
10d, as `OrderReconciler`. Phase 11c's brief asked for the other half:
everything a real (re)start must do, in order, before this system is
ever allowed to place an order again, plus persistence across the
restart itself. Both are implemented together here because 11c's
13-step sequence is the only caller 11b's engine has.

**`execution/reconciliation.py`'s `ReconciliationEngine`** is finally the
real thing, not the Phase 11b stub. `reconcile_positions()` compares
`PositionTracker.current_positions()` against `Broker.get_positions()` by
instrument, both directions — an instrument the broker reports that local
state has never seen ("missing local record") is exactly as much a
mismatch as a quantity that merely disagrees ("quantity mismatch"), and a
position local state holds that the broker no longer reports is a third,
distinct case, each worded differently in the mismatch detail so a reader
of a reconciliation report knows which of the three actually happened
without re-deriving it from the numbers. `reconcile_open_orders()`
delegates entirely to `OrderReconciler.reconcile_after_reconnect`
(Phase 10d) and reports as a mismatch only what that sweep could not
safely resolve on its own — `orphaned_broker_orders` — because resolving
an `UNKNOWN` order, refreshing a stale one, and timing out a stuck
submission are each already a safe, broker-confirmed *resolution*, not a
discrepancy this phase needs to re-report.

**A mismatch is quarantined, never guessed at.** Every mismatched
instrument this phase finds is a genuine ambiguity — the broker and this
system's own records disagree about how much of something exists, and
nothing in `broker.base.Broker`'s interface can say *why*, only *that*.
Per the phase's own explicit instruction, none of that is resolved
automatically: `StartupSequence.run()` returns a report with
`system_state=RECONCILIATION_REQUIRED` and
`permit_strategy_execution=False`, and the only way out is
`StartupSequence.acknowledge_and_recover(operator, reason)` — an
explicit, non-empty, `logger.critical`-logged, human-invoked call that
simply re-runs the sequence. This mirrors
`risk.circuit_breaker.CircuitBreaker.manual_reset`'s own established
"explicit, separately logged, never automatic" pattern for exactly this
kind of decision, deliberately reused rather than inventing a second one.
`acknowledge_and_recover` does not itself change anything about the
world — it is the operator's own out-of-band action (fixing local state,
confirming the broker's numbers are correct, whatever the investigation
concluded) that makes the next `run()` come back clean; the method's
whole job is making that re-run explicit and logged rather than silent.

**`execution/system_state.py`'s `SystemStateStore` stands in for "the
database."** `storage/database.py` remains an unimplemented Phase 12
stub, so this phase needed a real answer to "verify database" (step 3)
without fabricating a database layer that does not exist yet. The honest
answer, stated plainly in the module's own docstring rather than hidden:
a single JSON file, using the exact persistence shape
`CircuitBreaker` already established (`to_dict()`/`from_dict()`,
`json.dumps(..., indent=2, sort_keys=True)`, fail-closed on a corrupted
or wrong-shaped file via `SystemStateStoreError`) — reused deliberately
as the project's one precedent for "state that must survive a restart,"
not reinvented. `verify_accessible()` additionally probes that the
directory is actually writable (writes and removes a throwaway file)
before startup ever gets further, since a store that can be read but not
written would otherwise only fail much later, at the final persist.

**Two version concepts, deliberately kept separate.** `APP_VERSION`
(resolved via `importlib.metadata.version(...)`, falling back to
`"unknown"` if the package metadata is unavailable) is informational —
changing on every release, its mismatch against the previously-persisted
value only logged as a message, never blocking. `STATE_SCHEMA_VERSION`
is structural — this module's own control over the *shape* of what it
persists, and a mismatch is a hard `StartupError`: misreading an
incompatibly-shaped persisted file could silently corrupt this system's
understanding of its own state, which is a materially different risk
than merely running newer application code against old data.

**Step 9 ("resolve discrepancies") does not mean "make them go away" —
it means "resolve what is safe to resolve, and never touch what isn't."**
This distinction is the crux of the whole phase and is documented
prominently in `execution/startup.py`'s own module docstring so it
cannot be missed: order-level ambiguity (an `UNKNOWN` order) genuinely
can be resolved safely, because the broker is always authoritative for
its own order state — that is exactly what `OrderReconciler` already
does. A position-quantity mismatch cannot be resolved the same way,
because there is no query that explains *why* two numbers differ, only
confirms *that* they do — so it is never auto-resolved, on principle, not
as a missing feature.

**Steps 5-7 (positions, open orders, fills) treat a redelivered fill as
data, not as a new event.** Fills retrieved from the broker are
deduplicated by `trade_id` (`{fill.trade_id: fill for fill in fills}`)
before anything downstream sees them, and a message is logged whenever
deduplication actually removed something — proving the "duplicate broker
event" scenario is handled, not merely assumed away by a broker that
happens not to redeliver in testing.

**Step 10 (rebuild portfolio state) only runs once reconciliation is
clean**, and is an honest simplification rather than a full live-quote
refresh: `_mark_to_market_from_broker()` marks every held position to the
broker's own reported `avg_price`, which is the only price this phase has
without also standing up a live market-data connection — a real
last-traded-price refresh belongs to whatever live trading loop runs
after startup, not to the recovery sequence itself.

**Step 11 (verify risk state) reuses `core.regime.model_registry.ModelRegistry.approved_model_id()`
as-is** rather than inventing a new versioning concept — its natural
`None` default (nothing approved yet) is itself the correct "block
execution" signal, and a deployment that does not require an approved
model at all simply does not pass a registry, in which case the check is
skipped rather than treated as a failure. Step 12 reads
`CircuitBreaker.current_status()` directly rather than duplicating its
persisted content into `SystemStateStore` — a `HALTED` breaker forces
`SystemState.HALTED`, a state distinct from `RECONCILIATION_REQUIRED`
since the two require different operator responses (a halted breaker
needs `CircuitBreaker.manual_reset`; a reconciliation discrepancy needs
`StartupSequence.acknowledge_and_recover`).

**The named recovery scenarios are proven against both a fully
controllable stub broker and, for the highest-stakes one, a real broker.**
`tests/unit/test_startup.py` covers all eight scenarios the phase brief
names by name (clean restart, crash during order submission, crash after
fill, database restart, broker disconnect, duplicate broker event,
missing local record, unknown local order) plus the remaining steps not
covered by a named scenario (a `HALTED` circuit breaker, no approved
model, `acknowledge_and_recover`'s own validation and re-run behavior,
schema-version incompatibility, a merely-changed app version,
config-load failure, and `checkpoint_market_data`'s field-preserving
update). "Crash during order submission" is additionally proven against a
real `PaperBroker` wrapped in the same broker double
`test_order_reconciler.py` already established for Phase 10d's own
CRITICAL scenario — reusing it rather than only asserting the same
outcome against a stub, so the reconciliation this phase adds is shown
working through the same real fill/position mechanics a live run would
actually use.

## Application lifecycle and daily workflow (Phase 11d)

Every layer before this one answered a question in isolation: what is the
market doing, which stocks rank highest, what should the book look like,
is this trade allowed, did the order reach the broker, does local state
match the broker's. Nothing ran a *day*. `orchestration/` is that
missing piece, and its whole design brief is a negative one: sequence the
existing modules and own the process lifecycle, without acquiring any
opinion of its own about what the numbers should be.

**The lifecycle is a second, higher-level state machine, deliberately
distinct from `execution.system_state.SystemState`.** That enum
(Phase 11c) is internal bookkeeping for one `StartupSequence.run()` call;
`orchestration.orchestrator_state.OrchestratorState` describes what the
whole *process* is doing: `STARTING -> HEALTH_CHECK -> RECONCILING ->
READY -> RUNNING`, with `DEGRADED`, `HALTED` and `SHUTTING_DOWN` as the
exits. The two are related only where the orchestrator maps one onto the
other when persisting state.

**`DEGRADED` and `HALTED` mean genuinely different things, and the
difference is "can this resolve itself?".** `DEGRADED` is for conditions
that can clear on their own — market data that hasn't caught up yet,
insufficient trailing history to compute a regime, a mid-session
reconciliation break, a health check that came back degraded. Order
submission pauses; monitoring and reconciliation keep running; the next
clean loop iteration returns the process to `RUNNING` automatically.
`HALTED` is for conditions that by design cannot clear themselves: a
tripped circuit breaker, a broker that is not connected, a reconciliation
discrepancy at the startup gate, an unusable or missing model. Those
require an explicit operator action (`CircuitBreaker.manual_reset`,
`StartupSequence.acknowledge_and_recover`) exactly as Phases 7b and 11c
established. The phase brief's eight states have no separate
"reconciliation required" state, so a reconciliation break at the startup
gate maps onto `HALTED` and the report says which kind of halt it was,
rather than inventing a ninth state the brief did not ask for.

**Steps 1-6 are not re-implemented — they are `StartupSequence`.** The
daily workflow's "verify broker connectivity" and "reconcile portfolio"
are steps 4-9 of Phase 11c's own thirteen-step sequence in everything but
name, so `Orchestrator` constructs and runs a `StartupSequence` rather
than writing a second, subtly-different version of the same fail-closed
checks. It adds only the two checks that sequence has no reason to know
about: whether today is a trading day at all
(`TradingCalendar.is_trading_day`) and whether market data is fresh enough
to decide on (`HealthChecker.check_market_data_freshness`). Both run
*before* the broker is contacted, so a holiday or a stale data directory
costs nothing and touches nothing.

**`monitoring/health.py` stopped being a stub.** It was written in Phase 1
as a typed placeholder for "Phase 11/12", with exactly the five checks
this phase needed — broker, market-data freshness, instrument-master
freshness, model freshness, heartbeat. Implementing it there rather than
inlining the checks in the orchestrator is the whole "keep business logic
in dedicated modules" instruction applied literally: the orchestrator asks
"is the system healthy?" and reacts to the answer; it does not know what
makes market data stale. Every threshold comes from existing config
(`DataConfig.instrument_master_max_age_days`,
`MonitoringConfig.heartbeat_interval_seconds`,
`HMMConfig.retrain_interval_sessions`) rather than a new knob invented for
this phase.

**The four genuinely new pieces, and why each is its own module.**

- `orchestration/regime_computation.py` (`RegimeComputer`) pulls the
  trailing index window, computes features, applies the model's *frozen*
  scaler, filters, and asks `RegimeAllocationEngine` for today's exposure
  target. This is the identical inference recipe
  `WalkForwardValidator._hmm_exposure_targets` already ran out-of-sample;
  it exists as its own module so live orchestration does not have to
  import a backtesting class to get it. It re-fits nothing — a live
  decision uses the approved artifact's parameters and scaler exactly as
  persisted, which is what makes a live regime call reproducible from the
  audit log.
- `orchestration/trade_sizing.py` converts weight deltas into whole-share
  orders. It is explicitly *not* `risk/position_sizer.py` (Phase 7c, still
  stubbed), which will reconcile this weight-based formula against the
  stop-distance risk-based one into the single canonical quantity. This
  module uses the same simple `floor(notional / price)` shortcut
  `backtest/engine.py` already documents for its own fills, so the paper
  and backtest paths size trades the same way until 7c replaces both.
- `orchestration/fill_tracker.py` applies each broker fill to the
  canonical `PositionTracker` exactly once, deduplicated by `trade_id`.
  This closes a real asymmetry: `PaperBroker` updates whichever
  `PositionTracker` it was constructed with as a side effect of filling,
  while a live adapter updates nothing. The orchestrator therefore never
  shares its canonical tracker with the broker — the broker gets its own
  for order validation, and fills reach the canonical one through this one
  path for every adapter type. Without that separation, a paper fill would
  be counted twice and a live fill not at all.
- `risk/risk_state_builder.py` assembles the `PortfolioRiskState` the risk
  engine needs. `BacktestEngine` builds the equivalent privately for
  historical replay; that code was left untouched (refactoring a
  completed, tested phase's internals is not this phase's job), so this is
  a live reimplementation of the same approach that reuses the one piece
  already public — `market_liquidity_stats`. The one thing a live system
  cannot reconstruct the way a backtest does is its own equity curve, so
  `EquityHistory` is an explicit, append-only object the orchestrator
  maintains across the run.

**Step 17 is honest about not existing yet.** "Update stops/risk rules
where applicable" has no module to call: there is no trailing-stop or
resting protective-stop concept anywhere in this codebase (section 1.2 of
the specification demotes live stop orders to a last-resort control, and
`position_sizer.py`'s future `stop_distance` is a *sizing* input, not a
resting order). Rather than invent a stop-loss algorithm — which would be
exactly the strategy mathematics this layer is forbidden to contain — the
step is the defined seam such a module would plug into, and today does the
one thing that is genuinely applicable: re-marks positions to the broker's
latest prices and re-evaluates the circuit breaker, so the drawdown rule
that *does* exist reflects the fills just observed rather than this
morning's picture.

**Shutdown never liquidates by default.** `SIGINT`/`SIGTERM` set a flag;
the loop finishes its current iteration, transitions to `SHUTTING_DOWN`,
persists final state, and returns. Positions are left exactly as they are
unless `close_positions_on_shutdown=True` was passed at construction —
per this phase's explicit instruction, and for the obvious reason that a
process restart is not a trading decision. The previous signal handlers
are restored on the way out, so an orchestrator that ran inside a larger
process (or a test suite) leaves no trace in the process's signal table.

**Marketable limits, not last-price limits.** Orders are priced at the
touch — buy at the ask, sell at the bid — rather than resting at the last
traded price. This is execution plumbing rather than strategy: a limit
resting at the last trade frequently never fills, and for a
daily-rebalanced system that means silently drifting away from the
risk-approved target portfolio while believing it traded. How far through
the touch an order may be priced is not this layer's call either — the
broker's own `execution.order_price_guard_bps` check rejects anything
outside the configured band.

**The integration tests run the real pipeline, not mocks of it.**
`tests/unit/test_orchestrator.py` wires a real `StockSelector`,
`PortfolioConstructor`, `RiskManager`, `CircuitBreaker`, a genuinely
fitted and approved `ModelArtifact`, `OrderManager`, `PositionTracker`,
`StartupSequence`, `ReconciliationEngine` and a real `PaperBroker` against
the synthetic multi-year market the walk-forward suite already uses, then
runs the actual twenty-step workflow through them: a clean day reaches
`RUNNING` with orders that reach the broker, fills that reach the
canonical tracker exactly once, state persisted, and every order traceable
back to a signal and a risk decision. Each blocking condition gets its own
test proving *which* state it lands in and that nothing downstream ran —
a holiday, stale data, a disconnected broker, a reconciliation break, a
missing model, stale model metadata, a config failure, a tripped breaker.
The rest cover the ongoing loop (degrade, self-heal, halt, persist every
iteration) and shutdown (handlers installed and restored, the loop ends on
request, positions untouched by default and closed only when configured).

## Monitoring: the terminal dashboard and alerts (Phase 12)

Phase 11d gave the system a lifecycle; this phase gives it a face. Two
consumers, one source of truth: `monitoring/snapshot.py` gathers the
numbers once into a `MonitoringSnapshot`, and both the dashboard and the
alert rules read only that.

**Why the snapshot exists at all, rather than each consumer querying for
itself.** If the dashboard asked the broker for cash and the alert rules
asked again a moment later, an operator could be looking at a screen that
says one thing while an alert fires about another — and the resulting
"the dashboard was fine" argument would be unfalsifiable. Collecting once
makes what is displayed and what is alerted on provably the same reading.
It also makes both testable: `render_dashboard` is a pure function from
snapshot to string, and `evaluate_alerts` a pure function from one or two
snapshots to a list of alerts, so every field and every condition is
asserted directly rather than simulated.

**The dashboard renders; it does not compute.** No query, no state, no
arithmetic beyond presentation. It is plain ASCII at a fixed 80 columns,
with no colour, no cursor control and no terminal library, because it has
to be readable over ssh, in a Windows console and in a CI log — a
monitoring surface that itself fails to render is worse than none. A test
asserts every line is exactly the frame width at four different widths,
so a long value can never break the layout.

Two layout decisions came out of writing the tests rather than being
designed up front. An overlong value is clipped with an ellipsis — and
clipping is why the circuit-breaker line puts `MANUAL RESET REQUIRED`
*before* the trip reason: the first version appended it, and the one piece
of information telling an operator to do something was the piece that
disappeared. The free-text trip reason then moved to a continuation row of
its own, because it is the only unbounded field on the screen and it was
otherwise competing for space with the structured part of the same line.

**The regime panel shows the allocation tier and the descriptive label
side by side, never the label alone.** `RegimeLabel` ("calm", "crisis") is
assigned by ranking measured statistics and is reporting-only;
`AllocationRegime` is what actually drives exposure. A dashboard showing
only the label would imply the system acts on something it does not.

**Alerts: ten conditions, and why two of them look similar but are not.**
`MARKET_DATA_DISCONNECT` and `STALE_DATA` are separate alerts at separate
severities because they are separate problems with separate responses —
a feed that cannot be reached at all versus one that is merely behind.
`HealthChecker` already draws exactly that line (`UNHEALTHY` versus
`DEGRADED`), so the rules read it rather than re-deriving it. Likewise
`UNEXPECTED_POSITION` and `RECONCILIATION_MISMATCH` both come from the
reconciliation engine's mismatch list, but are split structurally on
`local_quantity == 0`: a broker position this system has no record of at
all means something outside the system traded the account, which is a
stronger signal than a disagreement about size.

**Some rules need two snapshots, and that is still pure.** A nonzero count
is not news — three orders were rejected this morning and are still
rejected this afternoon. What deserves an alert is the transition. So the
rejection rule fires on an *increase*, and the unexpected-cash rule
compares the change in cash against the change in observed fill cash flow.
The memory lives in the caller (`AlertManager` holds the previous
snapshot); the rules stay functions.

**Unexpected cash is a tolerance, not an equality.** Cash between two
snapshots should move by exactly the net cash flow of the fills observed
between them; anything else — a dividend, a fee sweep, a manual transfer,
a fill this system never saw — is by definition unexpected.
`FillTracker.cumulative_cash_flow` supplies the expected side, and it is
deliberately *gross of costs*: brokerage, taxes and slippage are priced at
fill time and are not visible in a `BrokerFill`, so the comparison is made
within a configurable fraction of equity that must exceed realistic cost
drag. Pretending it could be exact would make the alert fire on every
normal trading day.

**Rate limiting is not a nicety.** Most of these conditions persist until
someone acts on them: a disconnected broker stays disconnected through
every iteration of the monitoring loop, which at a 60-second poll is 60
identical alerts an hour. Without a cooldown the alerts that matter are
buried under the ones already known — precisely the failure mode alerting
exists to prevent. The cooldown is per alert *and subject*, so a mismatch
on one instrument never suppresses the alert about another, and a
suppressed alert is counted rather than discarded: the next one that gets
through reports how many it stands for. The interval is
`MonitoringConfig.alert_cooldown_seconds`, the one new configuration knob
this phase adds.

**An unconfigured channel fails closed.** `SUPPORTED_CHANNELS` is
`{"log"}` today, and `AlertManager` refuses to construct if
`alert_channels` names anything else. A channel an operator believes is
configured but that quietly delivers nothing is worse than no alerting at
all, so the gap is a startup error rather than a silent no-op. Alerts are
always logged regardless — that is the audit trail, not a channel.

**Wiring into the orchestrator is optional.** `Orchestrator` takes a
`snapshot_collector` and an `alert_manager`, both defaulting to `None`;
with neither it runs exactly as it did in Phase 11d, it just says nothing
about itself. When present, `publish_monitoring_snapshot()` runs at the
*end* of each monitoring-loop iteration, so what monitoring reports is the
state that iteration actually settled on rather than a state it was
passing through. It returns the snapshot so a caller can render it without
collecting a second, slightly different one.

`monitoring/dashboard.py` — the historical analytics views (regime
timeline, cost attribution, execution quality) — remains a stub. That is a
different artefact with a different audience: an operator watching a
running system needs current state in a terminal, which is what this phase
built first.

## End-to-end paper-trading validation (Phase 13)

Every prior phase proved its own layer in isolation. Phase 13 proves the
assembled whole: `validation/` wires the entire system in paper mode --
real `config/settings.yaml`, a real fitted and approved HMM, a real
`broker.adapters.paper_broker.PaperBroker` -- against a synthetic vendor
drop, and runs one coherent session through it: ingestion, features, the
HMM, ranking, portfolio construction, risk, execution, fills, accounting,
monitoring, shutdown, a crash, a restart, and reconciliation, with all
eight required failure injections triggered at the point in that
narrative where the real failure would actually occur.

**No live credentials, structurally.** `validation/harness.py` never
imports `broker.zerodha` and never reads `BROKER_API_KEY`/
`BROKER_API_SECRET`; the only broker it ever constructs is `PaperBroker`.
It does not call `broker.factory.build_broker` -- that factory hardcodes a
wall-clock time source, and this harness needs a controllable one for a
deterministic run -- so `settings.execution.mode == "paper"` is asserted
directly as a second, independent guard.

**The synthetic market exists to give every stage something to compute
over, not to say anything about returns.** `validation/synthetic_market.py`
writes vendor-shaped CSVs (bars, NIFTY 50, India VIX, an instrument master,
index membership) through a seeded random walk with alternating calm/
volatile blocks, so the regime engine has two genuinely different states
to separate. This repository's own production `config/settings.yaml` is
used unmodified -- including `hmm.training_window_days=756` and its
`candidate_states x random_seeds` grid -- so the synthetic market is sized
generously (1,100 sessions) to give that real config enough history to
fit against; the fit itself takes seconds, not minutes. Everything about
the *market* is fake; everything about the *system exercising it* is
real.

**Quotes are synthesized from the daily close, and this is stated as a
limitation, not hidden.** `LocalMarketDataProvider` is deliberately
historical-only (a backtest must never consult a live book), but
`PaperBroker` genuinely needs quotes -- to price a spread, check
staleness, reject a crossed market. `validation/paper_feed.py`'s
`ReplayQuoteFeed` closes that gap the only honest way available without a
live feed: the latest ingested close, stamped with the current simulated
clock, spread and depth as configured constants. This exercises every
path that depends on a quote existing, being fresh, and being consistent
with the order being priced -- the price guard, the staleness check,
partial fills against displayed depth -- honestly; it says nothing about
real spread magnitude, and the report's own "Limitations" section says so
explicitly rather than letting a reader assume otherwise.

**Invariants are checked after every stage, but a stage's own pass/fail
is judged separately from them.** `validation/invariants.py` checks six
of the seven properties the phase brief lists (`RESTART_IS_SAFE` is a
property of a *sequence* -- crash, correct refusal to trade, recovery --
not of one moment, so `validation.scenario` proves it directly instead).
Early in writing `validation/scenario.py`, every stage's `ok` folded in a
blanket "did every invariant just pass" check, which is wrong for a stage
whose entire purpose is to put the system into a state where an invariant
is *supposed* to read FAIL until a later stage resolves it -- a crash must
leave reconciliation failing until the operator-recovery stage runs, and
that is the crash-handling working, not a defect. `StageResult.ok` now
reflects only that stage's own context-aware assertion; per-stage
invariant snapshots stay attached for audit, and `SessionReport.final_invariants`
-- checked once, after the whole narrative including every recovery has
run -- is the authoritative "did the session end in a genuinely
consistent state" signal.

**A real bug, found by the "duplicate event" injection and fixed here.**
`orchestration.fill_tracker.FillTracker.poll` deduplicated incoming fills
against `_applied_fill_ids` with a membership filter computed once, up
front. A trade_id appearing *twice within the same `get_trades()`
response* -- exactly what a redelivered broker event looks like --
passed that filter both times, since neither copy was in the
already-applied set yet when the filter ran once for the whole batch;
both copies were then applied, double-counting the position. The fix
dedupes within the batch as well as against history
(`orchestration/fill_tracker.py`); `tests/unit/test_fill_tracker.py`'s
`test_a_fill_redelivered_twice_in_one_poll_is_applied_only_once` is the
regression test, and it is exactly the scenario `validation/scenario.py`'s
own `_failure_duplicate_event` stage exercises against the full system.
This is the validation phase doing its job: a bug no single layer's own
unit tests were positioned to find, because the layers on either side of
it (the broker's fill feed, the position tracker) were each individually
correct -- only assembling them and injecting exactly this failure
surfaced it.

**A second finding, documented rather than fixed.** The `application
crash` stage's recovery surfaces that `execution.execution_journal.ExecutionJournal`
is in-memory only (Phase 17's own stated scope -- durable persistence is
`storage/`'s unimplemented job). A crash therefore genuinely loses the
pre-crash orders' audit trail; `all orders traceable` holds for orders
created since the crash, not retroactively. This is an existing,
already-documented boundary, not a Phase 13 defect -- but an end-to-end
run is exactly what should say so in the generated report rather than
leave it implicit in a module docstring three phases back.

**Recovery in the `reconciliation` stage is the mechanism, not an
automated feature.** Neither a position mismatch nor an orphaned open
order is ever auto-resolved, on principle (Phase 17/18's own design): the
scenario plays the operator's role explicitly -- seed local state from
the broker's own reported truth for a missing position, cancel an
orphaned order (Phase 17's own prescribed resolution, since this system
has no signal/risk lineage for an order it never created and refuses to
adopt one blind) -- then calls `StartupSequence.acknowledge_and_recover`.
This demonstrates the tool Phases 17/18 provide for a human to use, not a
system that heals itself.

**The generated report is the phase's actual deliverable.**
`validation/report.py` renders a `SessionReport` as Markdown --
`docs/validation_report.md`, regenerated by `scripts/run_e2e_validation.py`
-- with the full narrative table, every final invariant, and a
"Limitations" section that states plainly what conclusions the run does
and does not support. `tests/unit/test_e2e_validation.py` runs the same
scenario as part of the ordinary test suite (the slowest test in the
repository, deliberately, since only the assembled whole can make an
end-to-end claim a per-layer unit test cannot).

## The live-trading safety gate (Phase 14)

Every prior phase built toward live trading being possible; this phase's
entire job is making sure it does not happen casually. Live order
submission is **still disabled** at the end of it -- that was explicit in
the brief, and nothing here changes `execution.mode`'s default or removes
any of the three confirmations `build_broker` already required.

**A fourth, independent confirmation, not a replacement for the other
three.** `broker.factory.build_broker` now requires
`enable_live_trading=True` *and* `preflight_confirmed=True` *and*
`execution.mode == "live"` *and* a passing `ComplianceGate` -- four
explicit flags/checks, none sufficient alone, mirroring the exact
"deliberate friction" pattern `enable_live_trading` already established
in Phase 15. `preflight_confirmed` is deliberately a plain boolean, not a
`PreflightReport` object or a file path `build_broker` reads and trusts:
passing a rich object would tempt a caller to construct one by hand
("just build a report that says PASS"), and reading a file would tempt a
caller to point at a stale one from last week. A boolean the caller must
have *just* obtained from an actual `run_preflight()` call carries the
same honesty requirement as `enable_live_trading` itself -- see
`build_broker`'s own docstring, and the dependency-rules note above on
why this stays a plain flag rather than `broker/` importing `live/`.

**The eighteen conditions split into two honestly different kinds, and
`docs/PRE_LIVE_CHECKLIST.md` says so.** Conditions 1-8 and 13-15 are
*evidence*: this repository's own test suite, actually run as a
subprocess every single time `python -m app.cli preflight` is invoked,
never a cached or remembered result. Conditions 9-12 and 16 are partly
*attestation*: `ComplianceConfig.broker_authorization_confirmed` is a
human's confirmation with the broker, not something a script can verify
against a broker's own dashboard; `docs/PRE_LIVE_CHECKLIST.md`'s own
"What PASS proves, and what it does not" section says this explicitly,
so a PASS is never mistaken for more certainty than it actually carries.

**Fail-closed applies to the checker itself, not just to what it
checks.** `--skip-test-suites` exists for fast local iteration on the
other ten conditions, but a run with it set reports those eight (nine,
once condition 8's internal adapter-test check is counted) as FAIL, not
skipped-and-therefore-ignored -- `run_preflight`'s own docstring is
explicit that such a run can never report an overall PASS. An
unverifiable prerequisite for live trading is not a satisfied one; a gate
that could be quietly bypassed by a fast-iteration flag would not be a
gate.

**Condition 16 (database backups) reads FAIL today, and that is the
correct answer, not a bug to route around.** `storage/database.py`
remains the unimplemented Phase 12 stub -- there is no real database, so
there is no backup/restore procedure to have tested. The check inspects
the module for that stub pattern and for a backup script under `scripts/`
rather than assuming either exists; fabricating a passing check here
would defeat the entire phase's purpose.

**The kill switch is a new, small, well-scoped addition, not a new
concept bolted onto existing code from the outside.**
`risk.circuit_breaker.CircuitBreaker.force_halt` mirrors `manual_reset`'s
own exact shape (operator, reason, both required and non-empty; persists;
logs at `CRITICAL`) but halts unconditionally, bypassing `evaluate`'s
threshold logic entirely -- the one thing `evaluate` itself can never do
is halt on a portfolio that looks fine, which is exactly what an
operator-invoked emergency stop needs to be able to do. Every trading
decision already respects a `HALTED` breaker (`RiskManager.evaluate`
rejects every proposed position the instant it is halted;
`Orchestrator`'s own daily cycle and monitoring loop both treat a halted
breaker as a hard stop), so engaging the kill switch needed no new
enforcement anywhere else -- only the halt primitive itself was missing.
`live.kill_switch.KillSwitch` gives that one call its own discoverable
name, matching how `live.preflight.PreflightCheck.KILL_SWITCH` refers to
it in the generated report.

**`app/cli.py` is the first genuine operational entry point this
repository has**, distinct from `main.py` (Phase 19's Phase-1-era
scaffold, still just configuration/logging) and from `scripts/` (one-off,
non-interactive tooling). `python -m app.cli preflight`'s exit code is
the primary interface, not its printed text -- `0` only when every
condition passed, `1` otherwise -- so it can gate a deploy step
mechanically, the same way `scripts/validate_config.py` and
`scripts/run_e2e_validation.py` already do for their own narrower checks.

## Production deployment and fail-closed hardening (Phase 15)

Phase 15 adds the things that turn a repository into a deployment, and --
more importantly -- revises one contract that only shows itself to be
wrong once a supervisor is involved.

### Fail-closed is an enumerable set, not a log message

`orchestration/fail_closed.py` names the six conditions under which this
system refuses to trade:

| `FailClosedReason` | Consequence |
|---|---|
| `UNKNOWN_BROKER_STATE` | do not place more orders |
| `STALE_MARKET_DATA` | do not trade |
| `RISK_ENGINE_FAILURE` | do not trade |
| `DATABASE_FAILURE` | do not trade |
| `CONFIGURATION_FAILURE` | do not trade |
| `MARKET_CALENDAR_UNCERTAINTY` | do not trade |

**Why this revises Phase 11d.** Three of these six previously raised out
of `Orchestrator.run_daily_cycle` uncaught. In a development shell that
reads as fail-closed: the process dies, so it certainly places no orders.
Under a production supervisor with `restart: unless-stopped` it reads
very differently — the process crash-loops, nothing is persisted about
*why*, the health check cannot answer because there is no process left to
answer it, and the operator sees a restart counter instead of a reason.

So `run_daily_cycle` now catches all six, records which one tripped on the
returned `DailyCycleReport`, persists `HALTED`, logs CRITICAL, and stays
alive to be asked about it. `OrchestratorError` is now reserved for
wiring/programming errors: **no operational failure raises out of
`run_daily_cycle`.** `run_forever`'s monitoring loop is wrapped the same
way, since the loop is what keeps a halted system observable.

Making the set an enum rather than prose buys three things: it is
testable as a whole (`tests/unit/test_fail_closed.py` asserts every
member is reachable, and a set-equality guard fails if a seventh is added
without a test), it is reportable on a dashboard, and `None` on a report
means "no fail-closed condition applied" rather than "it traded".

### What is actually deployed, and what is not

`app/service.py` is the long-running process the container supervises. It
discharges every startup obligation this repository currently can —
configuration loads and validates, logging is configured including the
durable audit trail, live mode is refused, the state store is proven
writable and parseable — and then heartbeats so an external check can
tell a wedged process from a working one.

**It does not run a trading day**, because that needs a composition root
(data provider, populated calendar, approved model artifact, constructed
broker) that does not exist until this deployment is provisioned. The
service logs that at WARNING on every start rather than presenting an
idle process as a trading one, and `docs/DEPLOYMENT.md` §1 says so first.
When the wiring lands, the loop becomes `Orchestrator.run_forever` and
nothing about the deployment changes.

### The health check's one counter-intuitive decision

`app/health.py` reports a **HALTED system as healthy**. Docker restarts
an unhealthy container, so this decision is precisely what gets
restarted. A halt means the fail-closed machinery worked; restarting
would discard the process that knows why, re-run startup into the same
condition, and halt again — reintroducing exactly the crash loop the
enum above exists to prevent. Unhealthy is reserved for a process that
has stopped making progress: no state file, an unparseable one, or a
heartbeat older than five minutes.

The check runs as a separate short-lived process reading the one file on
disk. A health check that asked the service's own in-memory objects
whether it was healthy would answer "yes" right up until the service
stopped being able to answer at all.

### Deployment layering

`deploy/` holds no Python and imports nothing. The refusals it encodes
are duplicated deliberately rather than shared: `deploy/entrypoint.sh`
refuses live mode in shell before any trading code is imported, and
`app/service.py` refuses it again in Python where the refusal has tests.
The shell check is the cruder one and exists because a container is the
thing most likely to be handed a stray `EXECUTION_MODE=live` by a
copy-pasted deploy command.

Neither is a *gate* in the Phase 14 sense. Deployment is not one of the
four gates in `broker/factory.py`, and that is the point: a container
image is copied between hosts, promoted between environments, and
restarted by a supervisor, and none of those events is a human deciding
to risk real money.

### Persistence, rotation and the audit trail

`monitoring/logger.py` now writes to two destinations, because they are
rotated by two different parties: stdout, rotated by Docker's `json-file`
driver (the process does not own that file), and an optional
`RotatingFileHandler` on a mounted volume, rotated by this process (Docker
does not know about that file). Both carry identical records; a separate
"audit" severity was rejected because the records an incident turns on
are ordinary INFO ones, and a filter deciding in advance which of those
matter will be wrong during the one incident that matters. An audit log
that cannot be opened is a startup failure, not a warning.

One real bug surfaced while testing this: `SystemStateStore.verify_accessible`
could raise a raw `OSError` from its own `mkdir`, past the typed boundary
that callers correctly catch `SystemStateStoreError` on. Fixed at the
source — a store whose failures do not all arrive as the store's own error
type has a hole in its fail-closed contract.

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
