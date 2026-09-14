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
