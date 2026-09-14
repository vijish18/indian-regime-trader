# Indian Market Regime-Based Automated Trading System

**Codex Engineering + Strategy Specification**

A production-oriented blueprint for an automated long-only Indian equity system using a
volatility-regime Hidden Markov Model (HMM), market-regime features, stock selection,
disciplined risk controls, realistic backtesting, broker API execution, and operational
recovery.

> **Design verdict.** The core idea is valid: use an HMM to classify market conditions and
> use the classification to control exposure rather than asking the HMM to predict the next
> price. The original concept becomes materially stronger when the HMM is market-level,
> volatility-focused, leakage-safe, and separated from security selection and execution.
>
> **Scope:** NSE-listed cash equities in the first production release. Default mode is
> daily-bar / next-session execution. Intraday trading, derivatives, leverage and shorting
> are explicitly deferred until separately validated.

*Engineering specification — India — 2026*

> This document is a faithful transcription of the source specification PDF
> (`Indian_Market_Regime_Trading_System_Codex_Specification.pdf`), reformatted as
> Markdown for version control. See [ARCHITECTURE.md](ARCHITECTURE.md) for how this
> repository implements it, including a small number of internal ambiguities the
> architecture review resolved (documented there, not silently edited into this file).

---

## 1. Strategy Logic Review

The system should be built around four separate decisions. Combining them into one model
makes debugging, validation and risk control harder.

| Layer | Question | Primary output |
|---|---|---|
| Market regime | How risky is the broad Indian market right now? | Filtered HMM regime + probability |
| Security selection | Which stocks are worth owning in that regime? | Ranked candidate list |
| Portfolio construction | How much capital should each candidate receive? | Target weights |
| Execution | How do we trade those targets safely? | Orders + fills + reconciliation |

### 1.1 What is correct in the HMM concept

- Use the HMM primarily as a state classifier, not as a directional forecaster.
- Let volatility and market stress determine risk appetite; do not equate a high-return HMM
  state with a bullish trading rule.
- Use filtered inference in live and backtest loops. Do not use full-sequence Viterbi
  inference to make historical decisions.
- Use walk-forward validation so each test period is evaluated using only information that
  would have been available at that time.
- Keep the hard risk manager independent of the HMM so a bad regime classification cannot
  bypass portfolio limits.

### 1.2 Logic changes required before implementation

| Issue | Why it matters | New rule |
|---|---|---|
| Daily HMM vs intraday loop mismatch | A model trained on daily observations should not silently drive 5-minute decisions. | V1 is daily HMM + next-session execution. Build intraday as a separate mode later. |
| HMM trained independently per stock | Individual stock regimes are noisier and less useful as a common risk signal. | Use NIFTY 50 / broad-market features for the HMM; use stock-level logic only for selection and sizing. |
| Too many directional features for a volatility classifier | RSI/ROC/SMA slope can cause the HMM to learn direction instead of risk state. | Prioritize realized vol, downside vol, ATR, gaps, range, India VIX, volume stress and breadth. |
| 252-day training conflicts with the stated 504-day minimum | The model can be undertrained relative to the specification. | Use 756 trading days minimum for V1; evaluate 504/756/1008-day windows in research. |
| Fixed leverage based on regime | Indian broker margin rules and product constraints differ from the assumed US setup. | V1 cash-only, no borrowing, no leverage. |
| Single fixed 0.05% slippage | Actual impact varies by liquidity, spread, price, turnover and order size. | Model spread + market impact + broker/exchange charges; calibrate by symbol liquidity bucket. |
| Stops as the main protection | Gap risk can jump through a stop. | Stops are a last-resort control; portfolio and gap-risk sizing remain primary protections. |
| Static stock universe | Creates survivorship bias and ignores new/deleted constituents. | Maintain point-in-time universe snapshots with delistings, suspensions and corporate actions. |

> **Important.** The HMM should answer "how much market risk should we take?" Stock
> selection should remain independent, so we can measure whether the regime layer adds
> genuine incremental value.

---

## 2. V1 Objective and Trading Universe

The first production system should intentionally be narrow. Narrow systems are easier to
validate and safer to automate.

| Parameter | V1 default |
|---|---|
| Primary exchange | NSE |
| Instrument type | Cash equities |
| Position direction | Long only |
| Holding period | Multiple days to weeks |
| Signal frequency | Once per trading day |
| Regime model source | NIFTY 50 market data + India VIX |
| Stock universe | Point-in-time liquid NSE equity universe, configurable |
| Execution mode | Next-session limit execution with price guard |
| Leverage | 1.0x maximum |
| Paper trading | Mandatory before live |
| Live capital | Separate small-risk deployment bucket |

### 2.1 Universe construction

- Prefer liquid names with sufficient traded value and stable order-book quality.
- Use point-in-time membership. Do not backtest today's NIFTY 50 members all the way into
  the past.
- Exclude securities with persistent illiquidity, trading suspensions or unreliable
  historical data.
- Store the exact universe snapshot used on every rebalance date.
- Keep an explicit exclusion list for instruments with corporate-action or data-quality
  anomalies.

---

## 3. System Architecture

```
Market Data Layer (NSE bars, quotes, VIX, breadth, instrument master)
        |
Feature + Data Quality Layer
        |
HMM Regime Engine (filtered state probabilities)
        |
        +--------------------+
        |                    |
Security Selection    Portfolio / Risk
(stock ranking)        (target weights)
        |                    |
        +--------------------+
                 |
          Execution Engine (orders / fills)
                 |
          Broker API / NSE
```

Cross-cutting: PostgreSQL, audit log, alerts, dashboard, reconciliation, health checks, kill
switch, state snapshots.

### 3.1 Recommended repository

```
indian-regime-trader/
  config/
    settings.yaml
    instruments.yaml
    credentials.example.env
  data/
    ingestion.py
    market_data.py
    instrument_master.py
    corporate_actions.py
    universe.py
    feature_engineering.py
    data_quality.py
  models/
    hmm_engine.py
    model_registry.py
    feature_scaler.py
  strategy/
    regime_policy.py
    stock_selector.py
    portfolio_constructor.py
    signal_generator.py
  risk/
    risk_manager.py
    circuit_breaker.py
    exposure.py
    order_validator.py
  execution/
    broker_interface.py
    broker_adapter.py
    order_manager.py
    position_tracker.py
    reconciliation.py
  backtest/
    engine.py
    costs.py
    performance.py
    walk_forward.py
    stress_test.py
  monitoring/
    logger.py
    alerts.py
    health.py
    dashboard.py
  storage/
    models.py
    migrations/
  tests/
  scripts/
  main.py
  pyproject.toml
  README.md
```

> See [ARCHITECTURE.md](ARCHITECTURE.md) for the actual repository layout used, which
> renames/splits a few of the above (e.g. `models/` + `strategy/regime_policy.py` become
> `core/regime/` and `core/features/`) and explains why.

---

## 4. Indian Market Data Layer

| Dataset | Required fields / use |
|---|---|
| NIFTY 50 | OHLCV, adjusted/continuous series for research, point-in-time index data |
| India VIX | Daily close and optional intraday series; market stress feature |
| Equity bars | OHLCV per symbol at daily frequency for V1 |
| Quotes | Bid/ask for live execution and spread checks |
| Instrument master | Trading symbol, exchange, segment, token, ISIN, tick size, lot size, price bands, status |
| Corporate actions | Splits, bonuses, dividends, rights, mergers/demergers and effective dates |
| Trading calendar | Exchange holidays, special sessions and Muhurat trading where applicable |
| Universe history | Point-in-time inclusion/exclusion records |
| Optional breadth | Advance/decline, % above moving average, market breadth stress |

### 4.1 Data-quality rules

- Reject duplicated timestamps, impossible OHLC relationships and negative/zero prices
  where invalid.
- Detect missing sessions, stale prices and suspicious volume spikes.
- Distinguish a genuine zero-volume session from missing data.
- Keep raw and normalized datasets separately. Never overwrite raw vendor data.
- Version every corporate-action adjustment and instrument-master snapshot.

---

## 5. Feature Engineering for the HMM

The HMM should learn market risk states from features that are mostly orthogonal to the
separate stock-selection model.

| Feature | Suggested construction | Role |
|---|---|---|
| 1-day log return | ln(Ct/Ct-1) | Shock / momentum context |
| 5-day return | ln(Ct/Ct-5) | Short-term market stress |
| 20-day return | ln(Ct/Ct-20) | Medium-term drift |
| 20-day realized volatility | std of daily log returns × sqrt(252) | Core volatility state |
| 5/20 vol ratio | 5-day realized vol ÷ 20-day realized vol | Volatility acceleration |
| Downside volatility | std of negative returns over rolling window | Crash sensitivity |
| ATR / price | ATR(14) ÷ close | Range and gap environment |
| Overnight gap | open_t ÷ close_t-1 − 1 | Open-to-open shock risk |
| Range expansion | true range relative to rolling median | Stress regime |
| India VIX level | normalized / percentile transform | Forward-looking implied volatility proxy |
| India VIX change | 1-day / 5-day change | Stress acceleration |
| Volume stress | log volume z-score or turnover z-score | Participation / stress |
| Breadth stress | advance/decline or % constituents above MA | Market participation |

Feature scaling must be strictly causal. For live inference, the transformation applied to
time t must be computable using data available no later than t. For walk-forward OOS
testing, scaler parameters should be fit on the training window and frozen during the OOS
window.

---

## 6. HMM Regime Engine

Use a Gaussian HMM as the initial implementation. The model is a state classifier. It is
not granted authority to predict returns or override the risk engine.

| Setting | V1 default |
|---|---|
| Candidate states | 3, 4, 5 |
| Research extension | 6 states only if justified by validation |
| Covariance | Full, with diagonal fallback for numerical stability |
| Training window | 756 trading days minimum |
| Retraining | Quarterly by default, plus drift-triggered retraining |
| Inference | Forward filtering only |
| Minimum confidence | 0.60 by default, configurable |
| State confirmation | 2 consecutive observations unless confidence is extreme |
| State flicker window | 20 sessions |
| Persisted artifacts | model, scaler, feature list, training range, BIC/AIC, seed, metadata |

### 6.1 Do not rely on human-readable state labels

State IDs are arbitrary. After each retraining, map them by learned characteristics, not by
numeric ID. Use expected volatility, downside volatility, transition behavior and
historical stress statistics to classify them as Calm, Normal, Elevated or Crisis-like for
reporting only.

### 6.2 Filtered inference pseudocode

```
alpha_t(i) = emission_i(x_t) * sum_j( alpha_(t-1)(j) * transition(j,i) )
normalize(alpha_t)
current_state = argmax(alpha_t)
confidence = max(alpha_t)
```

The key correctness property is that appending future observations must not alter the
regime used for a prior decision when the model parameters are held constant and inference
is performed with the forward filter.

### 6.3 HMM validation gates

- Compare model stability across multiple random seeds.
- Reject models that repeatedly converge to degenerate states or near-singular covariance
  matrices.
- Measure regime occupancy: a state that almost never occurs is usually not useful
  operationally.
- Measure average duration and transition probabilities; excessive one-day switching should
  be penalized.
- Run a label-permutation test; strategy performance should not depend on arbitrary state
  numbering.
- Compare HMM policy against simpler volatility filters. The HMM must add value beyond a
  rolling-volatility baseline.

---

## 7. Regime Policy and Stock Selection

Separate market-risk policy from stock selection. The HMM changes portfolio gross exposure;
a stock-ranking model decides what to own inside that risk budget.

| Regime | Target gross exposure | Allowed behavior |
|---|---|---|
| Calm | 85–100% | Normal stock-selection rules; widest risk budget |
| Normal | 70–90% | Normal selection; tighter concentration controls |
| Elevated | 40–65% | Only highest-quality/liquid names; no leverage |
| Crisis-like | 0–30% | Defensive holdings or cash; new positions heavily restricted |

### 7.1 Security selection — V1 simple baseline

The stock selector should start intentionally simple so the incremental contribution of the
regime model can be measured.

- Liquidity filter: minimum rolling traded-value threshold.
- Trend filter: price above a medium/long-term moving-average condition OR positive trend
  score.
- Momentum score: 3- and 6-month risk-adjusted momentum, excluding the most recent short
  window where useful.
- Quality/earnings factors may be added only when clean, point-in-time fundamental data is
  available.
- Cap the number of holdings; start with 5–10 positions rather than dozens.
- Use volatility-adjusted position sizes so a highly volatile stock does not automatically
  dominate portfolio risk.

### 7.2 Position sizing

```
risk_budget_i = portfolio_equity * allowed_portfolio_risk
raw_weight_i  = selection_score_i / volatility_i
apply_single_name_cap
apply_sector_cap
apply_liquidity_cap
scale all weights so sum(weights) <= regime_gross_exposure
```

Do not use the fixed "95% per stock" allocation from the old design. The portfolio should
distribute the regime budget across several names, with explicit single-name and sector
caps.

---

## 8. Risk Management — Independent Veto Layer

> **Non-negotiable.** The risk engine has absolute veto power. A valid strategy signal can
> still be rejected because of exposure, liquidity, drawdown, stale data, instrument
> status, pending orders, broker state, or compliance constraints.

| Control | V1 default / rule |
|---|---|
| Max gross exposure | 100% |
| Max single stock | 15% |
| Max sector | 30% |
| Max concurrent positions | 10 |
| Max portfolio risk per new position | 0.50% of equity |
| Daily loss warning | 1.5% |
| Daily loss reduce | 2.0% |
| Daily loss halt | 3.0% |
| Weekly loss reduce | 4.0% |
| Weekly loss halt | 6.0% |
| Peak-to-trough hard halt | 10% |
| Leverage | Never > 1.0x in V1 |
| Stale market data | No new orders |
| Broker reconciliation failure | No new orders |
| Spread too wide | No new orders |
| Missing stop / protection state | No new order |

### 8.1 Risk sizing formula

For a position with entry E and stop S, the theoretical size is:

```
quantity = floor((equity * max_risk_per_position) / abs(E - S))
```

Then cap the quantity by single-name weight, sector exposure, liquidity participation,
available cash and broker/exchange quantity restrictions. For overnight gaps, stress the
stop distance rather than assuming the stop will fill at S.

---

## 9. Indian Cost and Slippage Model

Backtests must reflect Indian costs, not a zero-commission US assumption. At minimum model
the applicable brokerage plan, STT/CTT as relevant, exchange transaction charges, SEBI
turnover fee, GST, stamp duty, and slippage/market impact. Rates should be
configuration-driven and versioned because they can change.

| Cost component | Implementation |
|---|---|
| Brokerage | Broker-specific schedule from configuration |
| STT | Instrument and transaction-side specific |
| Exchange charges | NSE/segment specific |
| SEBI turnover fee | Configuration with effective date |
| GST | Applied to eligible service/charge components |
| Stamp duty | State/account relevant, configuration-driven |
| Slippage | Spread + market impact model |
| Impact | Liquidity bucket + order participation cap |

### 9.1 Slippage model

Replace the single fixed 0.05% assumption with at least two modes: a deterministic research
model and an empirical live-calibration model.

```
research_slippage = max(min_bps, 0.5 * spread_bps + impact_bps(order_value, ADV, volatility))
```

Log expected versus realized execution price on every order so the slippage model can be
recalibrated from live paper-trading data.

---

## 10. Walk-Forward Backtesting and Validation

Backtesting is a primary product feature, not a helper script. The goal is to establish
whether the HMM produces incremental value after costs and realistic execution.

| Component | V1 rule |
|---|---|
| Training window | 756 sessions |
| Validation/test window | 126 sessions |
| Roll step | 63 or 126 sessions |
| Model selection | BIC on training only |
| Feature scaling | Fit on training; freeze in OOS |
| Regime inference | Forward filter only |
| Signal timing | Decision after session t; execution on session t+1 |
| Costs | All configured Indian costs |
| Execution | Opening-window limit/guarded model; sensitivity analysis |
| Universe | Point-in-time |
| Corporate actions | Point-in-time adjusted and audited |

### 10.1 Required benchmarks

- NIFTY 50 buy-and-hold / appropriate benchmark return series.
- Simple 200-day moving-average risk filter.
- Rolling-volatility threshold strategy without HMM.
- Equal-weight / volatility-weight stock portfolio without HMM.
- A shuffled-regime control to test whether the HMM ordering itself matters.

### 10.2 Performance metrics

| Category | Metrics |
|---|---|
| Return | CAGR, annual return, total return |
| Risk | Max drawdown, drawdown duration, volatility, downside deviation |
| Risk-adjusted | Sharpe, Sortino, Calmar |
| Trading | Turnover, trade count, average holding period, win/loss, profit factor |
| Costs | Gross P&L, costs, slippage, net P&L |
| Regime | Time in state, returns by state, drawdown by state, turnover by state |
| Robustness | Seed sensitivity, parameter sensitivity, walk-forward dispersion |

### 10.3 Reject overfit strategies

- Do not select parameters only because they maximize CAGR.
- Prefer stable performance across neighboring parameters and windows.
- Treat a large improvement appearing only in one market period as suspect.
- Require the HMM strategy to outperform its simpler volatility baseline after costs on
  multiple OOS windows.

---

## 11. Stress Testing and Failure Simulation

- Gap-down shocks of 3%, 5%, 8%, 12% and larger where historically plausible.
- Circuit-limit / price-band scenarios where orders cannot be filled at the expected stop.
- Bid-ask widening by 2x, 5x and 10x normal spread.
- No-trade periods and temporary market halts.
- Broker API outage during an open position.
- WebSocket disconnect with delayed or missing fills.
- Duplicate order response / request timeout after the broker accepted the order.
- Partial fill followed by restart.
- Incorrect or stale instrument metadata.
- Corporate-action event between signal and execution.
- Wrong HMM regime classification for an entire stress period.
- Bad feature values / missing India VIX / missing breadth data.

The system should fail closed for risk-critical uncertainty: it may stop new entries,
reduce risk, or require manual intervention rather than guessing.

---

## 12. Indian Broker API and Execution Layer

Use a broker-neutral interface and implement one production adapter first. Candidate
brokers can be evaluated later without rewriting strategy code.

```python
class Broker:
    get_account()
    get_positions()
    get_open_orders()
    get_quotes(symbols)
    place_order(order)
    modify_order(order_id, changes)
    cancel_order(order_id)
    close_position(symbol)
    close_all_positions()
    health_check()
```

### 12.1 Order state machine

```
CREATED -> VALIDATED -> SUBMITTED -> ACKNOWLEDGED -> PARTIAL/FILLED
                                                    -> REJECTED
SUBMITTED/ACKNOWLEDGED -> CANCEL_REQUESTED -> CANCELLED
UNKNOWN state -> RECONCILIATION REQUIRED
```

### 12.2 Indian execution rules

- Do not hard-code a generic "market-order fallback". Order types and algo-order
  restrictions must match the broker and exchange framework in force.
- Use unique client-side trade IDs linking signal → risk decision → order → fill →
  position.
- Apply price collars / limit-price guards so a stale signal cannot cross an uncontrolled
  price range.
- Reject orders when the instrument is not tradable, the quote is stale, spread is
  excessive, or instrument metadata is inconsistent.
- Respect tick size, freeze/quantity restrictions, price bands and broker/exchange order
  constraints from the instrument master.
- Handle partial fills as first-class events rather than treating "submitted" as "position
  created".

---

## 13. India Retail Algo / Operational Compliance

This is a system requirement, not a documentation afterthought. The current SEBI
retail-algo framework and exchange implementation standards apply to API-based retail algo
trading. The production build must be aligned with the chosen broker's supported route and
current exchange onboarding requirements.

| Area | Requirement for system design |
|---|---|
| API route | Use the broker's supported retail API/algo route; confirm whether the client is treated as a tech-savvy client/direct API or another supported model. |
| Static IP | Support a fixed outbound IP where required by the applicable retail API route. |
| Algo identification | Preserve and submit required algo identification/tagging exactly as the broker/exchange integration specifies. |
| Order types | Do not assume market/IOC orders are permitted for algo flow; enforce the broker/exchange supported set. |
| Traceability | Every decision and order must have an auditable ID and timestamp. |
| Audit logs | Persist immutable strategy, risk, order and execution events with timestamps. |
| Broker onboarding | Do not switch to live until the broker confirms the account/API/algo setup is eligible for the intended use. |
| Rules drift | Add a compliance configuration version and a release checklist so a regulatory/broker change can block deployment. |

> **Current-state note.** NSE's published retail-algo FAQ states that client static IP is
> required for a Tech Savvy Investor using API, that API-originated orders are treated as
> algo orders under the framework, and that market orders are not permitted for algo
> orders. The system therefore treats compliance metadata and order-type constraints as
> executable configuration rather than informal notes. Verify the broker-specific
> implementation before production.

---

## 14. India Market Session and Calendar Handling

- Use the exchange calendar as the source of truth; never assume every weekday is a trading
  day.
- Implement regular session, pre-open, and special-session handling according to the
  exchange segment.
- Use Asia/Kolkata internally for strategy timestamps, while storing UTC timestamps for
  unambiguous event ordering.
- Treat Muhurat trading and other special sessions as explicit session types.
- Never open or close positions solely because wall-clock time changed; depend on the
  session state supplied by the market-calendar service.

---

## 15. Reliability, Reconciliation and Recovery

### 15.1 Startup sequence

1. Load versioned configuration and validate it.
2. Load exchange calendar and current trading session.
3. Connect to broker and run health check.
4. Fetch actual positions, cash, buying power and open orders.
5. Reconcile broker state against local state; quarantine inconsistencies.
6. Validate instrument master freshness.
7. Validate market-data freshness.
8. Load the current approved HMM model and feature scaler.
9. Load prior portfolio state snapshot.
10. Only then enable new signal generation.

### 15.2 Fail-closed conditions

- Broker API unhealthy.
- Market data stale or missing.
- Position reconciliation mismatch.
- Unknown open order.
- Instrument master older than configured freshness limit.
- Risk state cannot be reconstructed.
- Compliance configuration incomplete.
- System clock out of tolerance.

---

## 16. Monitoring and Alerts

| Monitor | Alert threshold / behavior |
|---|---|
| Broker connectivity | Immediate alert on disconnect; no new entries |
| Market data | Immediate alert on stale feed |
| Reconciliation | Immediate halt on mismatch |
| Daily P&L | Warning / reduce / halt thresholds |
| Drawdown | Peak DD progression and hard halt |
| Execution quality | Expected vs realized slippage |
| Order rejection | Per-order reason + rolling rejection count |
| HMM stability | Flicker rate, confidence, state duration |
| Model freshness | Alert when retraining is due |
| System heartbeat | Alert when worker stops emitting heartbeat |

---

## 17. Core Database Objects

| Table | Key fields |
|---|---|
| instruments | instrument_id, symbol, ISIN, exchange, segment, tick_size, lot_size, price bands, effective_from/to |
| universe_snapshots | as_of, symbol, inclusion_reason, exclusion_reason |
| bars | instrument_id, timestamp, OHLCV, data_source, adjustment_version |
| features | timestamp, feature_set_version, feature values |
| hmm_models | model_id, train_range, feature_version, state_count, BIC, seed, artifact_path |
| regime_history | timestamp, state_id, state_label, probability vector, confidence |
| signals | signal_id, timestamp, symbol, target_weight, reason, regime_id |
| risk_decisions | signal_id, approved, modified_weight, rejection_reason |
| orders | trade_id, broker_order_id, symbol, side, qty, order_type, limit_price, status |
| fills | order_id, fill_qty, fill_price, timestamp, fees |
| positions | symbol, qty, avg_price, current_price, unrealized_pnl, target_weight |
| portfolio_snapshots | timestamp, equity, cash, gross_exposure, drawdown |
| audit_events | timestamp, component, event_type, payload_hash, severity |
| system_state | snapshot version, last processed market timestamp, health state |

---

## 18. Codex Build Plan

| Phase | Deliverable |
|---|---|
| Phase 1 — Repository + configuration | Create the repository, type-safe configuration, logging, environment handling, unit-test framework and CI. No live broker. |
| Phase 2 — Market calendar + instrument master | Implement India session calendar, point-in-time instruments, tick size, lot size, price bands and tradability checks. |
| Phase 3 — Data ingestion + quality | Implement NIFTY 50, India VIX and equity data ingestion, validation, raw/normalized storage and reproducible data snapshots. |
| Phase 4 — Causal features | Implement volatility-focused market features and frozen train/OOS scaling. Add data-leakage tests. |
| Phase 5 — HMM engine | Implement Gaussian HMM, BIC selection for 3/4/5 states, numerical safeguards, model registry and forward filtering. |
| Phase 6 — Baseline strategy | Implement simple volatility regime exposure policy and a separate stock selector. No broker integration. |
| Phase 7 — Portfolio/risk engine | Implement position sizing, sector/single-name limits, drawdown breakers, liquidity limits and fail-closed rules. |
| Phase 8 — Realistic backtester | Implement point-in-time universe, next-session execution timing, Indian costs, slippage and walk-forward evaluation. |
| Phase 9 — Stress testing | Implement gap, spread, outage, partial-fill, stale-data, corporate-action and wrong-regime simulations. |
| Phase 10 — Broker adapter | Implement one broker behind the Broker interface, paper trading first. Add reconciliation and order state machine. |
| Phase 11 — Live operational controls | Static IP/API configuration, compliance metadata, secure secrets, kill switch, monitoring, alerts, deployment and restart recovery. |
| Phase 12 — Production gate | Require all tests, backtest reports, paper-trading evidence, reconciliation tests and broker/algo onboarding checks to pass before live capital is enabled. |

---

## 19. Testing Requirements

- Unit tests for every feature formula and every risk constraint.
- Property tests for order quantity rounding, tick-size handling and target-weight
  conservation.
- No-look-ahead tests for features, scaler, HMM training and inference.
- Replay tests where the same historical event stream must produce identical decisions for
  a fixed seed/version.
- Order state-machine tests for duplicate responses, timeout, rejection, partial fill and
  recovery.
- Reconciliation tests where local state intentionally disagrees with broker state.
- Calendar tests around holidays, weekends and special sessions.
- Corporate-action adjustment tests.
- Cost-model tests with known fee schedules and effective dates.
- Kill-switch tests that verify absolutely no new order can be created after hard halt.
- Paper-trading soak tests before live deployment.

---

## 20. Production Go-Live Gate

| Gate | Required result |
|---|---|
| Data quality | Pass |
| No-look-ahead tests | Pass |
| Walk-forward performance | Stable across multiple OOS periods |
| HMM vs simple baseline | Demonstrates incremental value after costs |
| Stress testing | No uncontrolled failure mode |
| Risk engine | Independent veto proven |
| Broker paper trading | Successful end-to-end |
| Reconciliation | Clean under restart and partial-fill scenarios |
| Execution quality | Within configured slippage budget |
| Compliance/API onboarding | Broker confirms intended use is supported |
| Operational monitoring | Alerts verified |
| Kill switch | Verified in live-like environment |
| Capital limit | Initial deployment cap configured |
| Rollback | Known procedure documented |

---

## 21. Suggested V1 Configuration

```yaml
market:
  exchange: NSE
  timezone: Asia/Kolkata
  bar_timeframe: 1D
  execution_delay_sessions: 1

hmm:
  states: [3, 4, 5]
  training_days: 756
  retrain_sessions: 63
  min_confidence: 0.60
  confirmation_bars: 2
  flicker_window: 20
  max_covariance_condition: configurable

regime_policy:
  calm_gross_exposure: 0.90
  normal_gross_exposure: 0.80
  elevated_gross_exposure: 0.55
  crisis_gross_exposure: 0.20

risk:
  max_single_name: 0.15
  max_sector: 0.30
  max_risk_per_position: 0.005
  daily_reduce: 0.02
  daily_halt: 0.03
  weekly_reduce: 0.04
  weekly_halt: 0.06
  max_peak_drawdown: 0.10
  max_gross_exposure: 1.00
  max_leverage: 1.00

execution:
  mode: paper
  order_price_guard_bps: configurable
  stale_quote_seconds: configurable
  max_participation_adv: configurable
  no_market_order_fallback: true

backtest:
  test_days: 126
  step_days: 63
  include_indian_costs: true
  include_slippage: true
  point_in_time_universe: true
```

> Note: the `regime_policy` values above (0.90 / 0.80 / 0.55 / 0.20) are single-point
> defaults that fall within the exposure *bands* given in section 7 (85–100% / 70–90% /
> 40–65% / 0–30%) — the two are consistent, not contradictory. See
> [config/settings.yaml](../config/settings.yaml), which models each regime as an explicit
> `min_gross_exposure`/`max_gross_exposure` band rather than a single point value.

---

## 22. Questions the System Must Be Able to Answer

- How much of the portfolio was exposed when the HMM classified the market as elevated
  risk?
- Did the HMM outperform a simple volatility threshold?
- How much return came from stock selection versus regime timing?
- How much P&L was lost to brokerage, statutory charges and slippage?
- What happened if the stop gapped through by 3x the expected distance?
- Can the system recover safely after an API timeout or server restart?
- Can we reconstruct every order from signal through fill from the audit log?
- What happens if the model is wrong for an entire crisis period?
- Does the strategy still work with nearby parameter values?
- Does the system refuse to trade when market data, broker state or compliance metadata is
  uncertain?

---

## 23. Implementation Guidance for Codex

Do not ask Codex to generate the whole system in one prompt. Give Codex one phase at a
time. After each phase, run tests, inspect the diff, run a replay/backtest where relevant,
and freeze the interface before moving on.

For every Codex phase require:

1. Type hints on public interfaces.
2. Unit tests for normal + failure paths.
3. Deterministic fixtures.
4. Structured logging.
5. No hidden globals.
6. Config-driven thresholds.
7. Explicit timestamps/timezones.
8. Backward-compatible database migrations.
9. No network calls in unit tests unless explicitly marked integration tests.
10. A README section explaining how to run the phase locally.

---

## 24. Regulatory and Market References

The engineering specification should be checked against the latest applicable broker and
exchange documentation before each production deployment. The following references were
used for the current India-specific design:

- SEBI — "Safer participation of retail investors in Algorithmic trading", Circular
  SEBI/HO/MIRSD/MIRSD-PoD/P/CIR/2025/0000013, 4 Feb 2025.
- SEBI — Extension of timeline / implementation update for the retail algo framework,
  including applicability from 1 Apr 2026.
- NSE — Retail Algo FAQ, 3 Nov 2025: client static IP for Tech Savvy Investor API, API
  order treatment/tagging, and market/IOC restrictions.
- NSE — Market Timings & Holidays: current equity and derivatives session structure and
  special-session handling.
- NSE — SEBI Turnover Fees, STT and Other Levies: current published rates and
  applicability.
- NSE — Adjustments in case of Corporate Actions: splits, bonuses, rights,
  mergers/demergers, dividends and derivative adjustments.

> **Final recommendation.** Start with NIFTY 50 + India VIX as the market-regime layer, a
> simple liquid-stock selector, cash-only exposure, daily decisions, and strong
> risk/reconciliation controls. Do not introduce leverage, derivatives, intraday HMM
> signals or complex ML stock selection until the baseline proves incremental value out of
> sample.

*Engineering specification — India — 2026*
