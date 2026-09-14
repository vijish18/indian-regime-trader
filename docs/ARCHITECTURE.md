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
| 2 | Market calendar, point-in-time instrument master | Stubbed |
| 3 | Data ingestion, corporate actions, data quality | Stubbed |
| 4 | Causal feature engineering + scaling, no-look-ahead tests | Stubbed |
| 5 | HMM engine, model registry | Stubbed |
| 6 | Regime policy, stock selector | Stubbed |
| 7 | Portfolio constructor, position sizer, risk manager | Stubbed |
| 8 | Backtest engine, Indian cost/slippage model, performance metrics | Stubbed |
| 9 | Walk-forward validation, stress testing | Stubbed |
| 10 | Broker interface, paper adapter, order manager | Stubbed |
| 11 | Position tracking, reconciliation, live operational controls | Stubbed |
| 12 | Alerts, health checks, dashboard, production go-live gate | Stubbed |

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
