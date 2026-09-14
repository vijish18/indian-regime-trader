# Indian Market Regime Trading System

An India-focused, long-only systematic equity trading system for NSE-listed cash
equities. Market exposure is gated by a Hidden Markov Model (HMM) that classifies
broad-market volatility regime from NIFTY 50 and India VIX; stock selection,
portfolio construction, risk management, and execution are kept as separate,
independently testable layers. See [docs/SPECIFICATION.md](docs/SPECIFICATION.md)
for the full engineering specification and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
for how it maps onto this repository.

**V1 scope:** cash equities only, long only, no leverage, no shorting, no
derivatives, no intraday strategy, daily data/decisions, paper trading required
before any live capital.

## Project status

**Phases 1-11 complete: configuration, the broker-independent data layer,
point-in-time universe construction, causal feature engineering, the HMM
regime engine, regime-aware portfolio allocation, stock selection, portfolio
construction, independent risk management, the Indian transaction-cost
model, and realistic walk-forward backtesting.**

- **Phase 1** — repository structure, type-safe/validated configuration,
  structured logging, environment handling, unit-test framework.
- **Phases 2-3** — market calendar, point-in-time instrument master, corporate
  actions, point-in-time index membership, local CSV/Parquet storage,
  ingestion, and data-quality validation. Historical data only: there is no
  broker or vendor connection, and `LocalMarketDataProvider.get_quote` raises
  by design so a backtest cannot consult a live book.
- **Universe engine** (`universe/universe.py`) — joins index membership,
  instrument status, and (optionally) corporate actions into a point-in-time
  eligible universe. `UniverseProvider.get_universe(as_of)` returns only
  constituents that were actually members, tradable, and not already subject
  to a completed merger/demerger/delisting on that exact date — the primary
  survivorship-bias control described in `docs/SPECIFICATION.md` section 2.1.
  Every exclusion is recorded with a reason rather than silently dropped.
  Liquidity/trend/momentum filtering is a separate concern
  (`universe/stock_selector.py`, see below) — see
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the scope boundary and
  documented data limitations.
- **Feature engineering** (`core/features/feature_engineering.py`) — a
  deliberately small, fully-documented set of 9 causal market-regime features
  from NIFTY 50 and India VIX (returns, realized vol, vol ratio, VIX level/
  change, trend, drawdown, normalized ATR, volume stress). Every rolling
  computation is a trailing, never-centered window with `min_periods` equal
  to its declared lookback, so warm-up is NaN rather than a guess, and
  appending future data never changes an already-computed value (tested
  directly, not just asserted). `FeaturePipeline.audit()` produces a full
  provenance trail (`FeatureSnapshot`: name, timestamp, value, source
  observations, lookback) for every value. This is the HMM's *input*
  pipeline only — the HMM itself (`core/regime/hmm_engine.py`) and the
  train/freeze/apply feature scaler used for walk-forward fitting
  (`core/features/feature_scaler.py`) remain Phase 5.

- **Regime engine** (`core/regime/`) — a Gaussian HMM that classifies market
  risk state, fitted by Baum-Welch over every (candidate state count × seed)
  pair and selected by BIC, with candidates rejected for non-convergence,
  degenerate states or near-singular covariance before selection. Live regime
  calls use a dedicated forward filter computing `P(state_t | obs_1..t)` —
  never smoothing, never Viterbi, both of which would let tomorrow's data
  change today's answer. States are described by *measured* statistics
  (annualized expected volatility, expected return, downside volatility,
  occupancy, expected duration, persistence); labels like "crisis" are
  assigned by ranking those measurements and are reporting-only, never a
  decision input. Models persist as versioned JSON artifacts with an explicit
  approval gate.
- **Regime-aware allocation** (`core/regime/allocation.py`) — turns a
  `RegimeState` history into an `AllocationTarget` (a gross-exposure band and
  a point target within it), never a stock pick: `RegimeAllocationEngine` has
  no access to any security's price, score, or candidacy. Classification uses
  only `expected_volatility` and `confidence` — never `RegimeLabel` — mapped
  through two configured volatility thresholds into LOW_RISK / NORMAL_RISK /
  HIGH_RISK, plus UNCERTAIN when the latest confidence is below
  `hmm.min_confidence`, when no tier has ever been confirmed yet, or when the
  confirmed tier has changed too often within the flicker window (all three
  reasons are pure functions over plain arrays, independently tested). A new
  tier only takes effect after `hmm.confirmation_bars` consecutive agreeing
  observations (or one at extreme confidence); until then the previous
  confirmed tier is held. Within a confirmed tier, the exposure target scales
  continuously with confidence across the tier's configured band — never a
  single fixed percentage, and never able to imply leverage (`AllocationTarget`
  re-validates `0 <= min <= max <= 1`). `core/regime/regime_policy.py` is a
  pure band lookup; `core/regime/baseline_policy.py`'s
  `RollingVolatilityBaseline` is the required non-HMM comparison strategy —
  classifying trailing realized volatility alone, with the same
  annualization convention and the same configured bands, into the identical
  `AllocationTarget` shape, so a later walk-forward run can swap one for the
  other and let docs/SPECIFICATION.md section 10.3's requirement ("HMM beats
  the simple baseline after costs") actually be checked rather than assumed.
- **Stock selection** (`universe/stock_selector.py`,
  `universe/factor_calculator.py`) — decides *which* stocks receive the risk
  budget the regime layer has already set; `StockSelector` has no access to
  the current regime, exposure target, or any risk state, and places no
  orders. Not a machine-learning model: six transparent factors (medium-term
  momentum, trend persistence, relative strength vs. NIFTY 50, volatility,
  drawdown, liquidity) combined by configured, non-negative weights into one
  composite score. Every factor is computed from adjusted price history
  ending exactly at the decision date (`price_basis=ADJUSTED`, per
  `data.interfaces.MarketDataProvider`'s own point-in-time contract), and
  standardized *cross-sectionally* — against the other candidates on that
  date, never against its own history over time, which is a different
  computation entirely
  (`core.features.feature_engineering.rolling_standardize`). Instruments
  with insufficient price history or below-threshold liquidity are excluded
  with a recorded reason before ranking, never silently dropped or included
  by default; excluding one candidate never perturbs another's score, since
  standardization only ever runs over the surviving set (tested directly).
  No fundamentals/quality factor — `data/` has no fundamentals source, and
  the gap is documented rather than faked.
- **Portfolio construction** (`portfolio/portfolio_constructor.py`) — combines
  the regime's risk budget (`AllocationTarget`), stock rankings (`StockScore`),
  and every configured position limit into one `TargetPortfolio`, long-only,
  cash-equity-only, no leverage. `construct()` runs a fixed waterfall of pure
  reductions — candidate selection, risk-adjusted raw weights, a correlation
  penalty between highly-correlated pairs, normalization and scaling to the
  regime's exact target exposure, then single-name, liquidity, and sector cap
  enforcement, and finally a minimum-weight floor — each step only ever
  shrinks a weight, so the pipeline converges in one pass. Weight trimmed by a
  cap becomes cash; it is never redistributed to another name, which keeps
  every cap violation degrading toward less risk, never more. `TargetPortfolio`
  is used structurally for both the target and the current portfolio, and
  `required_trades()` is a pure diff between the two into BUY/SELL/EXIT/HOLD
  actions — no order is sized or sent from this module. A defense-in-depth
  check re-validates the finished portfolio against every configured limit
  before returning it.
- **Independent risk management** (`risk/risk_manager.py`,
  `risk/circuit_breaker.py`, `risk/portfolio_risk_state.py`) — the layer
  with absolute veto power over every proposed `TargetPortfolio`; it never
  imports `core.regime`, so no regime label can talk it out of a check.
  `RiskManager` approves or rejects each proposed position outright (it
  never resizes one) against gross exposure, single-name and sector
  concentration, correlation concentration, position count, liquidity/ADV
  participation, stale data, abnormal spread, daily turnover, and the V1
  long-only/no-leverage/no-borrowing invariants — a position with no
  matching risk data is rejected outright, fail closed. A `CircuitBreaker`
  tracks account health independently: NORMAL / REDUCED_RISK / HALTED,
  driven by daily, rolling-window, and peak-to-trough drawdown, plus system
  and broker-connectivity health. REDUCED_RISK tightens exposure caps and
  forbids new positions (fail closed to "reject" when it can't tell what's
  new); HALTED rejects every proposed order with no exceptions, persists to
  disk so it survives an application restart, and clears only through an
  explicit, separately logged `manual_reset()` — never automatically. Every
  circuit-breaker transition and every risk rejection is logged with a
  structured reason.
- **Indian transaction-cost model** (`backtest/costs.py`,
  `backtest/cost_schedule.py`) — models brokerage, STT, exchange
  transaction charges, SEBI turnover fee, GST, stamp duty, DP charges, and
  slippage/spread/impact *separately*, never as one generic commission
  number. Rates are versioned data (`config/cost_schedules.yaml`), never a
  Python constant: each dated `CostSchedule` is selected by
  `CostScheduleRepository.schedule_as_of(trade_date)`, the same
  point-in-time pattern corporate actions and the HMM model registry use,
  so a rate change is added as a new dated entry rather than edited in
  place, and a trade date before the earliest known schedule fails closed
  (`MissingCostScheduleError`) instead of guessing. `TradeCost` is the
  fully deterministic brokerage + statutory breakdown for one trade leg;
  `ExecutionCostEstimate` adds the *estimated* slippage on top (the section
  9.1 research model: `max(min_bps, 0.5 * spread_bps + impact_bps(...))`).
  Every cost line item is tagged `DETERMINISTIC` / `BROKER_DEPENDENT` /
  `EXCHANGE_DEPENDENT` / `ESTIMATED` via `TradeCost.CATEGORY`, so which
  costs are which is a queryable class attribute, not just documentation.
  Every charge rounds to the nearest paisa (half-up, not Python's
  banker's-rounding default) before being summed, so a displayed total
  always exactly matches the sum of its displayed line items. STT applies
  on both legs for delivery equity, stamp duty on the buy leg only, and DP
  charges on the sell leg only — modeled per leg, never averaged.
- **Walk-forward backtesting** (`backtest/engine.py`, `backtest/walk_forward.py`,
  `backtest/performance.py`) — a true out-of-sample fit → freeze → OOS →
  advance → retrain loop, never fitting model parameters on future test
  data. Each fold's HMM and feature scaler are fit on a rolling (not
  expanding) training window only, frozen, and evaluated strictly on the
  following test window. `BacktestEngine` replays every session's decision
  — stock selection, portfolio construction, risk veto, cost pricing — from
  information available at that session's close only, and executes the
  resulting orders at the **next** session's opening price, never the
  signal's own close or open. Five strategies (buy-and-hold, the
  rolling-volatility baseline, a moving-average trend baseline, the HMM,
  and a shuffled-regime control) run over the identical fold sequence and
  identical downstream pipeline, isolating whether the HMM's regime timing
  — not just its existence — adds value after costs. Every fold's equity
  curve, positions, orders, fills, costs, regime, confidence, risk
  decisions, and turnover are recorded. Look-ahead and leakage are tested
  directly via truncation invariance: two environments built from the same
  random seed but differing amounts of data must produce bit-identical
  decisions for every date both of them cover.

**No position sizing or execution logic is implemented yet** —
`risk/position_sizer.py` (converting an approved target weight into a
final, risk-bounded order quantity) and everything past it remain typed
stubs that define the interfaces for later phases. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Before running ingestion, populate `config/nse_holidays.csv` from NSE's
published holiday list — it ships empty and the calendar fails closed rather
than guessing whether the exchange was open (see
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)).

## Repository layout

```
config/       Typed, schema-validated configuration (settings.yaml + loader)
core/
  regime/     Market-regime detection (HMM engine, regime policy, model registry)
  features/   Feature engineering + causal feature scaling
data/         Broker-independent market data: interfaces, models, calendar,
              instrument master, corporate actions, membership, storage,
              ingestion, validation
universe/     Point-in-time universe construction + stock selection
portfolio/    Portfolio construction (target weights) + position sizing
risk/         Independent risk management with veto authority
execution/    Order management, position tracking, reconciliation
broker/       Broker-neutral interface + adapters (paper adapter first)
backtest/     Walk-forward backtesting, cost/slippage model, stress testing
monitoring/   Structured logging, alerts, health checks, dashboard
storage/      Persistence layer (table schemas, DB session management)
scripts/      Operational / one-off scripts
tests/        Unit and integration tests
docs/         Specification, architecture, development guide
```

## Getting started

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\activate.bat on cmd.exe
pip install -e ".[dev]"

cp .env.example .env            # fill in local/paper values; never commit .env

pytest                          # run the unit test suite (no network, no broker)
mypy .                          # type-check
ruff check .                    # lint
```

## Configuration

All parameters that affect trading behavior — regime thresholds, exposure bands,
risk limits, cost assumptions, execution guards — live in
[config/settings.yaml](config/settings.yaml) and are validated against
[config/settings.schema.yaml](config/settings.schema.yaml) and the typed models in
`config/models.py` at load time. Nothing that belongs in configuration should be
hardcoded in module logic. Secrets and per-deployment values (broker keys, database
URL) live in `.env`, never in `settings.yaml`.

## Documentation

- [docs/SPECIFICATION.md](docs/SPECIFICATION.md) — the source engineering specification.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how the specification's layers map to this repo, module boundaries, and the phase build plan.
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — local setup, coding standards, test conventions.
