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
- `broker/adapters/paper_broker.py` and `execution/position_tracker.py` import from
  `backtest/` (`backtest.costs.CostModel`, `backtest.engine.market_liquidity_stats`) on
  purpose (Phase 10-11a): a paper fill must be priced through the identical Indian
  cost/slippage model a backtest fill uses, or the two are not comparable. This is the one
  place a later layer intentionally depends on an earlier one across the broker/backtest
  package boundary; it does not go the other way -- `backtest/` never imports `broker/` or
  `execution/`.
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
| 11a | Position tracking (`execution/position_tracker.py`) | **Done** |
| 11b | Reconciliation, live operational controls (`execution/reconciliation.py`) | Stubbed |
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
(`PositionTracker.apply_corporate_action`/`force_exit`) and reconciliation against a broker's
own state (`execution/reconciliation.py`) remain Phase 11b — this phase's `PositionTracker`
is the source of truth for a single, unreconciled run, not yet cross-checked against
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
