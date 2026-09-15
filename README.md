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

**Phases 1-15 complete: configuration, the broker-independent data layer,
point-in-time universe construction, causal feature engineering, the HMM
regime engine, regime-aware portfolio allocation, stock selection, portfolio
construction, independent risk management, the Indian transaction-cost
model, realistic walk-forward backtesting, performance analytics, stress
testing, a paper-trading engine, and a broker abstraction with a real
Zerodha Kite Connect adapter (live trading disabled by default).**

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
- **Performance analytics** (`backtest/comparison.py`, `backtest/robustness.py`,
  `backtest/report.py`) — turns each strategy's `PerformanceReport` into
  the comparison the walk-forward run exists to answer, built around one
  rule: nothing here ever emits a `success` verdict. `PerformanceReport`
  itself now also reports `recovery_duration_days` (time from the *worst*
  drawdown's trough back to its prior peak), `average_holding_period_days`
  (reconstructed from closed round trips in the trade log), and
  `pct_invested`/`pct_cash` (needs the new `cash_history` on
  `BacktestResult` — not derivable from the equity curve alone), plus
  `by_confidence()` alongside the existing `by_regime()` breakdown — both
  now aligned *positionally* with the equity curve rather than by date
  label, after finding that regime/confidence history (signal-dated) and
  the equity curve (execution-dated, one session later) never actually
  share date labels to reindex against. `compare_to_baseline`/`compare_all`
  compute a signed delta per metric between the HMM and each of the four
  required baselines (simple volatility classifier, buy-and-hold,
  trend-only, randomized control) and attach programmatically-generated
  caveats to every comparison — never empty: a standing reminder to check
  robustness, plus conditional ones for a low trade count, a high Sharpe
  alongside a large drawdown, an outperformance that comes with *worse*
  risk, an infinite profit factor, or cost drag eating a large share of
  gross P&L. `RobustnessSuite` runs a batch of caller-supplied variant
  closures across all seven required dimensions (parameter perturbation,
  training window, rebalance threshold, transaction cost, slippage,
  universe size, market period — the rebalance-threshold control,
  `BacktestEngine.min_rebalance_weight_delta`, is new this phase, and a
  real V1 option in its own right, not only a robustness knob) and reports
  each key metric's dispersion across them, with an explicit, opt-in
  `is_stable()` check rather than an automatic pass/fail. `backtest/report.py`
  writes all of this to CSV (one row per strategy or variant) and to
  Markdown/HTML (every comparison's caveats printed directly beneath its
  numbers; the robustness section never silently dropped when supplied).
- **Stress testing** (`backtest/stress_test.py`) — 20 Indian equity-market
  failure scenarios (market/data shocks, execution/infrastructure
  failures, model/decision failures), built around one requirement: risk
  controls must limit damage even if the HMM is wrong. `full_exposure_targets()`
  feeds a deliberately-not-the-HMM "always fully invested" signal through a
  real market shock, so the crash and regime-misclassification scenarios
  prove the circuit breaker — which watches realized P&L, not the regime
  label — halts or reduces risk regardless of what the exposure signal
  claims. `ShockedMarketDataProvider` applies a deterministic price/volume/
  availability/index shock on top of a real `MarketDataProvider`;
  `StressTestContext` rebuilds `StockSelector` and `PortfolioConstructor`
  fresh against the shocked feed via factory callables, since both
  otherwise capture their own market-data reference independent of the
  engine's. An infrastructure failure the engine itself catches
  (`BacktestEngineError`, e.g. missing mark-to-market data) is reported as
  a fail-closed pass, not propagated as a test failure — refusing to
  proceed on bad data is the system working as designed. No live broker or
  database exists yet, so execution-layer scenarios (partial fill, order
  rejection, broker outage, application restart, database failure) each
  test the specific mechanism that already exists for that failure mode,
  documented as such rather than simulating infrastructure that isn't
  built. 8 of the 20 scenarios run as Monte Carlo sweeps (100+ trials,
  deterministic per-trial seeding) over randomized shock magnitudes rather
  than one hand-picked case.
- **Paper-trading engine** (`broker/base.py`, `broker/adapters/paper_broker.py`,
  `execution/order_manager.py`, `execution/position_tracker.py`) — a full
  `Broker` implementation strategy code cannot distinguish from the live
  Zerodha adapter below, since both sit behind the identical interface.
  `OrderManager` owns a ten-state order lifecycle (`CREATED` through
  `FILLED`/`CANCELLED`/`REJECTED`/`EXPIRED`, plus `UNKNOWN` for a lost or
  ambiguous broker response) and enforces idempotency by a caller-supplied
  key — a resubmitted identical trade request is never turned into a
  second order; `PaperBroker` separately deduplicates by `client_order_id`
  itself, the same guarantee a real broker would enforce. An `UNKNOWN`
  order is resolved only by querying the broker's own truth
  (`Broker.get_order`, added to the interface this phase specifically for
  that), never by blind retry. `PaperBroker` prices every fill through the
  identical `backtest.costs.CostModel` a backtest fill uses (plus a
  genuine improvement: real bid/ask spread from a live `Quote`, not the
  backtest's assumed constant), matches against the quote's own displayed
  depth so an order larger than one match's share of the book partially
  fills instead of assuming unlimited liquidity, and rejects orders a real
  broker would (unsupported order type, stale or crossed quote, price
  outside the configured guard band, a sell beyond the held quantity, a
  buy beyond available cash) as defense in depth on top of whatever
  `RiskManager` already approved upstream. `PositionTracker` is the one
  portfolio-state shape (weighted-average cost, realized and unrealized
  P&L) both this paper broker and a live adapter produce.
  Connects to nothing real: no live broker, no live market-data feed.
- **Broker abstraction and a real Zerodha Kite Connect v3 adapter**
  (`broker/base.py`, `broker/errors.py`, `broker/factory.py`,
  `broker/zerodha/`) — the `Broker` interface grew `BrokerCapabilities`
  (query what an adapter supports before calling it), `BrokerFill` and
  `get_trades()` (trade/fill history, distinct from an order), and
  `subscribe_market_data()` (streaming where available — both
  `PaperBroker` and `KiteBroker` implement it, `PaperBroker` always
  raising `BrokerCapabilityError` since no paper feed exists).
  `broker/zerodha/kite_broker.py` is built strictly from Zerodha's
  published Kite Connect v3 documentation, fetched and verified while
  writing it — no invented endpoints, fields, auth methods, or
  WebSocket behavior; every genuine gap in the docs (no client-side
  order-ID lookup, no dedicated server-time endpoint, an
  incompletely-documented index-instrument WebSocket packet layout) is
  documented at the point it matters rather than guessed. It bridges
  Kite's broker-assigned order IDs to this system's client-generated
  ones with an in-memory map (not persisted — closing that gap is
  Phase 11b's reconciliation work), and implements the verified
  WebSocket subscribe/unsubscribe control messages and binary
  full/quote/ltp tick decoding for equities, driven by the caller one
  frame at a time rather than a background thread (no WebSocket
  transport ships by default — connecting a real socket is deliberately
  left unwired). **Live trading is gated twice, independently**:
  `broker/factory.py` only constructs a live-capable broker when
  `execution.mode == "live"` *and* the caller passes
  `enable_live_trading=True` explicitly; `KiteBroker` itself defaults
  that flag to `False` and re-checks it before every order-placing call.
  Credentials come from the environment (`BROKER_API_KEY`/
  `BROKER_API_SECRET`), never from `settings.yaml`. Every test runs
  against a scripted, in-memory fake HTTP transport — nothing here
  makes a real network call, and default mode remains paper.

**Position sizing, reconciliation, and live operational controls are not
implemented yet** — `risk/position_sizer.py` (converting an approved
target weight into a final, risk-bounded order quantity),
`execution/reconciliation.py`, and everything past them remain typed
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
backtest/     Walk-forward backtesting, cost/slippage model, performance analytics, stress testing
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
