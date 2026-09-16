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

**Phases 1-23 complete: configuration, the broker-independent data layer,
point-in-time universe construction, causal feature engineering, the HMM
regime engine, regime-aware portfolio allocation, stock selection, portfolio
construction, independent risk management, the Indian transaction-cost
model, realistic walk-forward backtesting, performance analytics, stress
testing, a paper-trading engine, a broker abstraction with a real Zerodha
Kite Connect adapter, India API/algo operational controls,
production-grade order management, restart recovery/broker
reconciliation, the orchestration layer that runs a full trading day
end to end, a terminal dashboard with rate-limited alerting, an
end-to-end paper-trading validation harness proving the whole system
against a synthetic market with full failure injection, and a live-trading
safety gate that keeps live order submission disabled until a formal
18-condition pre-live checklist genuinely passes, and a Docker production
deployment whose trading process fails closed on all six named unsafe
conditions — live trading is still disabled by default throughout, and
nothing in this codebase has ever placed a real order.**

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
  `execution/reconciliation.py`'s job, run at every startup), and implements the verified
  WebSocket subscribe/unsubscribe control messages and binary
  full/quote/ltp tick decoding for equities, driven by the caller one
  frame at a time rather than a background thread (no WebSocket
  transport ships by default — connecting a real socket is deliberately
  left unwired). **Live trading is gated three times, independently**
  (the third added by the compliance layer below): `broker/factory.py`
  only constructs a live-capable broker when `execution.mode == "live"`
  *and* the caller passes `enable_live_trading=True` explicitly;
  `KiteBroker` itself defaults that flag to `False` and re-checks it
  before every order-placing call. Credentials come from the environment
  (`BROKER_API_KEY`/`BROKER_API_SECRET`), never from `settings.yaml`.
  Every test runs against a scripted, in-memory fake HTTP transport —
  nothing here makes a real network call, and default mode remains paper.
- **India API/algo operational controls** (`config.models.ComplianceConfig`,
  `broker/compliance.py`, `docs/COMPLIANCE.md`) — before this phase's own
  code was written, the applicable NSE circular and SEBI retail-algo
  framework were researched and fetched directly (not recalled or
  assumed); `docs/COMPLIANCE.md` is that research record, including
  everything that could **not** be verified and why. `ComplianceGate` is
  the third independent live-trading gate: its constructor refuses to
  exist — `ComplianceError`, a hard failure — if
  `broker_authorization_confirmed` is not `True`, or `static_ip_primary`/
  `algo_identifier` are still `settings.yaml`'s default placeholders
  (`"0.0.0.0"`/`"UNSET"`), before any network call is even possible. Per
  order, it enforces an allow-list of order types/validities (`MARKET`
  and `IOC` excluded, per NSE's algo rules), a session-age check against
  `Broker.health_check().login_time` ("expired authentication" is a hard
  failure, not a retry), and a client-side sliding-window order-per-second
  throttle — and **never substitutes a prohibited value for an allowed
  one**: every check either returns the order unchanged (with the
  required algo-identifier tag applied) or raises, with no code path that
  silently swaps in a permitted type and proceeds.
  `ComplianceGuardedBroker` applies all of this uniformly by wrapping any
  `Broker`, including reimplementing `close_position` so a position close
  is tagged and gated exactly like any other order rather than bypassing
  the gate through the inner adapter's own internal call.
- **Production-grade order management** (`execution/order_manager.py`'s
  `ExecutionStateMachine`, `execution/order_reconciler.py`'s
  `OrderReconciler`/`RetryPolicy`, `execution/execution_journal.py`'s
  `ExecutionJournal`) — the order-lifecycle transition rules are now a
  standalone, independently-testable class rather than a table embedded
  in `OrderManager`, and every `signal_id`/`risk_decision_id` an order is
  created with is required, not optional: `create()` cannot produce an
  order with no traceable identity chain. `OrderManager.journal` records
  every create/transition automatically, so `signal_id -> risk_decision_id
  -> client_order_id -> broker_order_id -> fills` is always
  reconstructable (`ExecutionJournal.trace()`) without a caller having
  remembered to log anything. `OrderReconciler` adds stale-order
  detection, submission-timeout handling, and a `reconcile_after_reconnect`
  sweep (resolve every `UNKNOWN`, refresh every stale order, surface
  orphaned broker orders — never resubmitting anything) on top of the
  existing single-order `UNKNOWN` resolution; `RetryPolicy` only ever
  retries read-only broker queries, never `place_order`. Reconciling a
  broker-reported state this system's own step-by-step model would never
  have produced on its own (an order it last saw resting `OPEN`, cancelled
  through another channel while disconnected) is exactly why every
  non-terminal state can now reach any terminal state directly — a real
  finding from writing this phase's failure-injection tests, not a
  convenience. The CRITICAL scenario — a broker accepts an order but the
  response is lost before this system sees it — is proven end to end
  against a real `PaperBroker`: the broker is called exactly once, the
  local record reads `UNKNOWN` before reconciliation and the true,
  broker-confirmed state after, and the order is never duplicated by
  either the original attempt or the reconciliation that follows it.
- **Restart recovery and broker reconciliation**
  (`execution/system_state.py`'s `SystemStateStore`,
  `execution/reconciliation.py`'s `ReconciliationEngine`,
  `execution/startup.py`'s `StartupSequence`) — a real (re)start now runs
  a 13-step sequence, in order, before this system is ever allowed to
  place an order: load configuration, verify the application/schema
  version, verify the state store, verify broker connectivity, retrieve
  broker positions/open orders/fills (deduplicated by `trade_id`, so a
  redelivered broker event is never double-counted), compare broker
  state against local state, resolve what is safely resolvable (an
  `UNKNOWN` order — never a position-quantity mismatch), rebuild
  portfolio state, verify an approved regime model exists, verify the
  circuit breaker is not `HALTED`, and only then permit strategy
  execution. **A genuine discrepancy is never auto-resolved.** Any
  position mismatch — a broker position with no local record, a local
  position the broker no longer reports, or a plain quantity mismatch —
  moves the system to `RECONCILIATION_REQUIRED` and blocks execution;
  the only way out is `StartupSequence.acknowledge_and_recover(operator,
  reason)`, an explicit, logged, human-invoked re-run, mirroring
  `CircuitBreaker.manual_reset`'s own "never automatic" pattern rather
  than inventing a second one. `SystemStateStore` persists system state,
  the current model/strategy version, a portfolio snapshot, and the last
  processed market-data timestamp/broker event across restarts, using the
  identical JSON-file persistence shape `CircuitBreaker` already
  established, and stands in for "the database" this phase, since
  `storage/database.py` remains an unimplemented stub — documented as a
  stated scope decision, not hidden. All eight recovery scenarios named
  in this phase's brief (clean restart, crash during order submission,
  crash after fill, database restart, broker disconnect, duplicate
  broker event, missing local record, unknown local order) are tested;
  "crash during order submission" is additionally proven against a real
  `PaperBroker`, reusing Phase 10d's own CRITICAL-scenario broker double.

- **Application lifecycle and daily workflow** (`orchestration/`,
  `monitoring/health.py`, `risk/risk_state_builder.py`) — the layer that
  finally runs a *day*: a state-driven lifecycle
  (`STARTING → HEALTH_CHECK → RECONCILING → READY → RUNNING`, with
  `DEGRADED`, `HALTED` and `SHUTTING_DOWN` as exits) around a twenty-step
  workflow — load and validate configuration, verify the market calendar,
  data availability and broker connectivity, reconcile the portfolio, load
  and validate the approved model, compute the regime, rank stocks,
  construct the target portfolio, run the risk engine, calculate required
  trades, submit permitted orders, track fills, update the portfolio and
  risk rules, persist state, monitor health, and reconcile periodically.
  **It computes nothing itself**: every decision is delegated to the module
  that already owns it, and a test enforces structurally that no module in
  `orchestration/` imports a numerical library. Steps 1-6 delegate to Phase
  18's `StartupSequence` rather than re-implementing the same fail-closed
  checks a second time. `DEGRADED` means "this can clear on its own" (stale
  data, a mid-session reconciliation break — the next clean loop iteration
  returns to `RUNNING`); `HALTED` means "only an operator can clear this" (a
  tripped circuit breaker, a disconnected broker, an unusable model).
  `SIGINT`/`SIGTERM` are handled gracefully — the current iteration
  finishes, final state is persisted, the previous signal handlers are
  restored, and **positions are never closed merely because the process is
  shutting down** unless `close_positions_on_shutdown` was explicitly
  configured. Integration tests run the real pipeline (real selector,
  constructor, risk manager, circuit breaker, fitted/approved model, order
  manager and a real `PaperBroker`) against a synthetic multi-year market,
  asserting which lifecycle state each blocking condition lands in and that
  nothing downstream of it ran.

- **Monitoring: terminal dashboard and alerts** (`monitoring/snapshot.py`,
  `monitoring/terminal_dashboard.py`, `monitoring/alerts.py`) — a
  `SnapshotCollector` gathers the whole system's state **once** into a
  `MonitoringSnapshot`; both consumers read only that, so what an operator
  sees on screen and what triggers an alert can never be two different
  readings of the same moment. `render_dashboard` is a pure function from
  snapshot to text — plain ASCII, fixed 80 columns, no colour or terminal
  library, so it renders over ssh, in a Windows console and in a CI log —
  showing SYSTEM (status, uptime, trading session, data-feed health,
  broker connectivity, model version), PORTFOLIO (equity, cash, exposure,
  daily P&L, drawdown, positions), REGIME (allocation tier *and* the
  reporting-only HMM label, probability, confidence, persistence, India
  VIX, NIFTY 50), EXECUTION (submitted, fills, rejected, open, pending
  reconciliation) and RISK (risk mode, circuit breakers, concentration,
  turnover). `evaluate_alerts` is likewise pure, covering ten conditions:
  broker disconnect, market-data disconnect, stale data, order rejection,
  unknown order state, reconciliation mismatch, risk halt, excessive
  drawdown, unexpected position, and unexpected cash balance. **Alerts are
  rate-limited per condition and subject** — a disconnected broker stays
  disconnected through every loop iteration, and without a cooldown the
  alerts that matter drown in the ones already known; suppressed alerts
  are counted, not discarded, so the next one delivered reports how many
  it stands for. A channel named in config that this system cannot
  actually deliver to is refused at construction rather than silently
  dropping alerts. Wiring into the orchestrator is optional.
- **End-to-end paper-trading validation** (`validation/`) — proves the
  assembled system, not just each layer in isolation: this repository's
  own unmodified `config/settings.yaml`, a real fitted and approved HMM,
  and a real `PaperBroker` run one coherent session — ingestion,
  features, the HMM, ranking, portfolio construction, risk, execution,
  fills, accounting, monitoring, shutdown, a crash, a restart, and
  reconciliation — against a synthetic vendor drop generated and ingested
  through the real pipeline. All eight required failure injections (lost
  WebSocket, delayed market data, broker API timeout, rejected order,
  partial fill, duplicate event, application crash, database restart) are
  triggered at the point in that narrative where the real failure would
  occur, and every invariant (no duplicate positions, no negative cash,
  no leverage, all orders traceable, reconciliation succeeds, halted
  state persists, restart is safe) is checked throughout and once more at
  the very end against live state. **No live credentials, structurally**
  — the only broker ever constructed is `PaperBroker`; `BROKER_API_KEY`/
  `BROKER_API_SECRET` are never read. `scripts/run_e2e_validation.py`
  writes the result to `docs/validation_report.md`; the same scenario
  also runs inside the ordinary test suite
  (`tests/unit/test_e2e_validation.py`, the slowest test in the
  repository, deliberately). This phase's own "duplicate event"
  injection found and fixed a real bug: `FillTracker.poll` could
  double-count a fill redelivered twice within one `get_trades()`
  response — its membership filter was computed once, up front, so a
  trade_id not yet in the applied set let both copies through.
- **Live-trading safety gate** (`live/`, `app/cli.py`,
  `docs/PRE_LIVE_CHECKLIST.md`) — **live order submission is still
  disabled.** `broker.factory.build_broker` now requires a *fourth*
  independent confirmation, `preflight_confirmed=True`, on top of
  `execution.mode == "live"`, `enable_live_trading=True`, and a passing
  `ComplianceGate` — a plain boolean the caller must have just obtained
  from `python -m app.cli preflight`, the same trust model
  `enable_live_trading` already uses (a remembered or hardcoded `True`
  defeats the point). That command runs all eighteen pre-live conditions
  — unit/integration/look-ahead tests, walk-forward and stress-test
  completion, paper-trading evidence, broker reconciliation, API/
  compliance/static-IP/order-type/risk-limit configuration, a tested kill
  switch, restart recovery, monitoring, database backups, and two secrets
  scans — prints PASS/FAIL with a concrete reason for each, writes
  `docs/preflight_report.md`, and exits non-zero on any failure so it can
  gate a deploy step. Checks 1-8 and 13-15 are genuine evidence (this
  repository's own test suite, run as subprocesses, every single
  invocation — never a cached result); a run with fewer conditions
  checked (`--skip-test-suites`) can never report an overall PASS. A new
  kill switch (`risk.circuit_breaker.CircuitBreaker.force_halt`,
  `live.kill_switch.KillSwitch`) halts trading immediately regardless of
  current risk state, mirroring `manual_reset`'s own established pattern
  in the opposite direction. Run against this repository's own
  (deliberately unconfigured) development settings, the checklist
  correctly reports FAIL — compliance and static-IP configuration are
  still placeholders, and `storage/database.py` is still an unimplemented
  stub with no backup procedure to have tested — which is the checklist
  working exactly as intended, not a defect to route around.
- **Production deployment and fail-closed hardening** (`Dockerfile`,
  `deploy/`, `orchestration/fail_closed.py`, `app/service.py`,
  `app/health.py`, `docs/DEPLOYMENT.md`, `docs/OPERATIONS.md`,
  `docs/INCIDENT_RESPONSE.md`) — a Docker stack (non-root container on a
  read-only root filesystem with all capabilities dropped, Postgres on an
  internal network with a least-privilege role, separate volumes for
  state/logs/data, `restart: unless-stopped`, resource limits, health
  checks, two-layer log rotation, a verified backup script) plus host
  templates for a default-deny firewall, key-only SSH and an HTTPS
  reverse proxy. **The trading process fails closed on six named
  conditions** (`FailClosedReason`): unknown broker state, stale market
  data, risk-engine failure, database failure, configuration failure and
  market-calendar uncertainty. This deliberately revises Phase 19's
  contract — three of those used to raise out of `run_daily_cycle`, which
  reads as fail-closed in a shell but becomes a crash loop under a
  supervisor, burying the reason and leaving no process to answer a
  health check. They now halt, name themselves on the report, and stay
  alive to be asked. For the same reason `app/health.py` reports a
  **HALTED system as healthy**: restarting a correctly-halted process is
  exactly the crash loop being avoided. **Nothing here can turn live
  trading on** — deployment is not one of `broker/factory.py`'s four
  gates, and both the shell entrypoint and the Python service refuse to
  start in live mode (a distinct exit code, so a pipeline can page on it).
  The deployed service does not yet run a trading day, because no
  composition root is wired; it says so at WARNING on every start rather
  than presenting an idle process as a trading one. Smoke tests run at
  both levels: `tests/unit/test_deployment.py` for the logic and
  `tests/integration/test_docker_smoke.py`, which builds the real image
  and runs the real container — and which caught a real bug no unit test
  could have, since PEP 475 makes `time.sleep` resume after a signal, the
  60-second heartbeat wait outlived Docker's 30-second grace period, and
  the trading process was being SIGKILLed without ever running its
  shutdown path.

**Position sizing is not implemented yet** — `risk/position_sizer.py`
(reconciling the weight-based and stop-distance sizing formulas into one
canonical order quantity) remains a typed stub that defines the interface
for a later phase; the orchestration layer sizes orders with the same
simple weight-based shortcut the backtest engine documents for its own
fills. The historical operational analytics views (`monitoring/dashboard.py`
— regime timeline, cost attribution, execution quality) are also still a
stub; that is a different artefact with a different audience from the live
terminal dashboard above. See
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
execution/    Order management, position tracking, reconciliation, restart recovery
broker/       Broker-neutral interface + adapters (paper adapter first)
backtest/     Walk-forward backtesting, cost/slippage model, performance analytics, stress testing
orchestration/ Application lifecycle + the daily workflow that sequences every layer above
monitoring/   Structured logging, alerts, health checks, dashboard
validation/   End-to-end paper-trading validation harness (no live credentials)
live/         Live-trading safety gate: pre-live checklist, kill switch (live still disabled)
app/          Operational CLI, the long-running service, the health check
storage/      Persistence layer (table schemas, DB session management)
deploy/       Production deployment: compose stack, entrypoint, backup, host
              templates (firewall, SSH, HTTPS proxy), least-privilege DB init
Dockerfile    Production image (non-root, read-only root filesystem)
scripts/      Operational / one-off scripts
tests/        Unit and integration tests
docs/         Specification, architecture, deployment, operations, incident response
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
- [docs/MARKET_CALENDAR.md](docs/MARKET_CALENDAR.md) — where the NSE holiday data comes from, the Muhurat and mid-year-drift decisions, and what is still missing.
