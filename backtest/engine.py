"""Single-strategy backtest engine: replays signal -> risk -> execution
decisions session by session, with strictly next-session execution timing
and full Indian costs/slippage. See docs/SPECIFICATION.md section 10.

``walk_forward.py`` is responsible for the fit/freeze/roll loop across
folds and for producing each fold's ``exposure_targets`` (a market-timing
signal already computed from only in-window information -- see that
module for how the HMM is fit on a training window and frozen before this
engine ever sees a test-window date). This engine takes that signal as a
given, already-causal input and is deliberately agnostic to *how* it was
produced: the same loop drives the HMM strategy, every non-HMM baseline,
and the randomized control, which is exactly what makes the comparison
between them meaningful (docs/SPECIFICATION.md section 10.1).

## The twelve-step per-session pipeline

For every session ``T`` in the requested date range, in order:

1. Build the point-in-time eligible universe and rank it
   (``StockSelector.select(T)`` -- causal by construction, Phase 6b).
2. (implicit in 1) compute factors from adjusted price history ending at T.
3-4. Not repeated here -- the regime/HMM steps already happened once, up
   front, when ``exposure_targets`` was built (see module docstring above).
5. ``RegimeAllocationEngine``/baseline's answer for T is read from
   ``exposure_targets[T]``.
6. Stock selection's ranked candidates feed ``PortfolioConstructor``.
7. ``PortfolioConstructor.construct(...)`` combines the risk budget,
   rankings, and every configured limit into one proposed
   ``TargetPortfolio``.
8. ``RiskManager.evaluate(...)`` approves or vetoes each proposed position;
   :func:`_apply_risk_decisions` folds that into the portfolio actually
   acted on.
9. The weight deltas between the currently-held portfolio and the
   risk-approved target become hypothetical orders (:class:`OrderRecord`) --
   still no price, no quantity, no broker call.
10. Orders are sized (a simple weight-based ``floor(notional / fill_price)``,
    not the full stop-distance reconciliation ``risk/position_sizer.py``
    will own) and filled at the *next* session's execution price -- see
    "Execution timing" below.
11. ``backtest.costs.CostModel`` prices every fill's full Indian cost
    breakdown, deducted from cash immediately.
12. Cash, holdings, and every recorded series (equity, positions, orders,
    fills, regime, confidence, risk decisions, turnover) are updated.

## Execution timing: no same-bar execution on the signal's own close

The signal for session T is built from information available at T's close
(T's own closing bar is the most recent price stock selection and
portfolio construction ever see). The resulting orders execute at session
``T + 1``'s **opening** price -- never at T's close, and never at T's own
open. This is the "signal at close of day T, execution at day T+1" rule
docs/SPECIFICATION.md section 10 requires, made concrete as one specific,
named fill assumption (``next-session open``) rather than left implicit. A
full guarded-limit-order microstructure model (partial fills, price-guard
rejection) is ``execution/`` and ``broker/``'s job, not this backtest
engine's -- this V1 assumes every hypothetical order fills completely at
the next session's open.

## Rebalance threshold

``min_rebalance_weight_delta`` (default ``0.0``, i.e. every proposed change
is traded, matching this engine's original behavior) reverts any position
whose weight would move by less than the threshold back to its current
weight instead of trading it -- a deliberately small, local control for
"is a modest drift worth its transaction cost", used both as a real V1
option and as one of the robustness-diagnostic dimensions
(``backtest/robustness.py``, Phase 12).

## What's not here

Real bid/ask spread data does not exist in this dataset -- ``assumed_spread_bps``
is a single configured constant standing in for it (a documented V1
simplification, not a fabricated market fact). Position sizing here is a
simple weight-to-quantity conversion, not ``risk/position_sizer.py``'s
stop-distance risk-based reconciliation (Phase 7c, still stubbed).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import cast

import pandas as pd

from backtest import checkpoint
from backtest.costs import CostModel, ExecutionCostEstimate, TradeSide
from config.models import RiskConfig
from core.regime.allocation import AllocationRegime, AllocationTarget
from data.delisting import DelistingNotice
from data.errors import DataNotAvailableError
from data.interfaces import CorporateActionProvider, MarketDataProvider, TradingCalendar
from data.models import CorporateActionType, DailyBar, PriceBasis
from portfolio.portfolio_constructor import (
    PortfolioConstructor,
    RequiredTrade,
    TargetPortfolio,
    TargetPosition,
    TradeAction,
    empty_portfolio,
    required_trades,
)
from risk.circuit_breaker import CircuitBreaker
from risk.portfolio_risk_state import PortfolioRiskState, PositionRisk
from risk.risk_manager import RiskDecision, RiskManager
from risk.stop_loss import StopBreach, StopLossPolicy
from risk.stop_loss import evaluate as evaluate_stop
from storage.atomic import atomic_write
from universe.stock_selector import StockSelector

_TRADING_DAYS_PER_YEAR = 252


def market_liquidity_stats(
    market_data: MarketDataProvider,
    instrument_id: str,
    as_of: dt.date,
    lookback_days: int = 20,
) -> tuple[float, float]:
    """``(avg_daily_value_inr, annualized_volatility)`` from raw
    traded-value/return history over the ``lookback_days`` sessions ending
    at (and including) ``as_of``.

    A module-level function, not a method, so both this engine's historical
    replay and ``broker.adapters.paper_broker.PaperBroker``'s live/paper
    simulation price slippage from the identical liquidity/volatility
    estimate -- ``backtest.costs.CostModel``'s square-root impact model
    is meaningless without a consistent definition of "how liquid is this
    instrument" feeding it from both callers.
    """
    start = as_of - dt.timedelta(days=lookback_days * 3)
    try:
        bars = market_data.get_equity_bars(
            instrument_id, start, as_of, price_basis=PriceBasis.ADJUSTED
        )
    except DataNotAvailableError:
        return 0.0, 0.0
    bars = bars[-lookback_days:]
    if len(bars) < 2:
        return 0.0, 0.0
    values = [float(bar.close) * bar.volume for bar in bars]
    avg_daily_value = sum(values) / len(values)
    closes = pd.Series([float(bar.close) for bar in bars])
    returns = closes.pct_change().dropna()
    volatility = (
        float(returns.std(ddof=0) * math.sqrt(_TRADING_DAYS_PER_YEAR)) if not returns.empty else 0.0
    )
    return avg_daily_value, volatility


class BacktestEngineError(RuntimeError):
    """A backtest run could not proceed -- invalid inputs or a data gap
    fail-closed rather than silently skipping a session."""


@dataclass(frozen=True, slots=True)
class OrderRecord:
    """One hypothetical order: a weight delta the risk-approved target
    implies, still with no quantity, no price, no broker call.
    """

    signal_date: dt.date
    execution_date: dt.date
    instrument_id: str
    action: TradeAction
    current_weight: float
    target_weight: float
    delta_weight: float


@dataclass(frozen=True, slots=True)
class FillRecord:
    """One order actually filled: quantity, price, and its full cost
    breakdown."""

    order: OrderRecord
    side: TradeSide
    quantity: int
    fill_price: float
    execution_cost: ExecutionCostEstimate


@dataclass(frozen=True, slots=True)
class DailyRiskDecisions:
    as_of: dt.date
    decisions: tuple[RiskDecision, ...]


@dataclass(frozen=True, slots=True)
class StopExit:
    """One position taken out by a stop rather than by the ranking.

    Kept separately from ``fills`` (which it also appears in) because the
    question "how much did the stops actually do" cannot be answered from a
    trade log: a stop exit and a rebalance exit are the same SELL row there.
    """

    execution_date: dt.date
    breach: StopBreach
    quantity: int
    cost_basis: float
    """What the position cost to acquire, buy-side charges included."""

    net_proceeds: float
    """What the sale actually credited, sell-side charges deducted."""

    @property
    def realized_pnl(self) -> float:
        return self.net_proceeds - self.cost_basis


@dataclass(frozen=True, slots=True)
class ShareAdjustment:
    """One split or bonus restating a held position's share count.

    The share count and the price move inversely and by exactly the same
    factor, so the position's *value* is unchanged across the event. That is
    the whole content of a split or a bonus: it is not a return.
    """

    instrument_id: str
    ex_date: dt.date
    action_type: str
    quantity_before: int
    quantity_after: int
    price_factor: float
    """What prices before ``ex_date`` must be multiplied by to compare with
    prices after it. Shares are multiplied by its reciprocal."""


@dataclass(frozen=True, slots=True)
class BacktestResult:
    strategy_name: str
    equity_curve: pd.Series
    """Total-return equity, indexed by *execution* date (see the module
    docstring's execution-timing rule) -- ``performance.PerformanceCalculator``'s
    input."""

    trade_log: pd.DataFrame
    """One row per fill: ``signal_date, execution_date, instrument_id,
    side, quantity, fill_price, gross_value, cost, net_value`` --
    ``performance.PerformanceCalculator``'s other input."""

    regime_history: pd.Series
    """Indexed by *signal* date -- the ``AllocationRegime`` value active
    when that session's decision was made."""

    confidence_history: pd.Series
    """Indexed by signal date -- ``AllocationTarget.confidence`` each
    session."""

    turnover_history: pd.Series
    """Indexed by signal date -- that session's proposed turnover
    (sum of ``|delta_weight|`` across every instrument), before any risk
    veto or execution."""

    positions_history: dict[dt.date, dict[str, int]]
    """Execution date -> ``{instrument_id: quantity}`` held after that
    session's fills."""

    cash_history: pd.Series
    """Indexed by execution date -- the cash balance after that session's
    fills and dividend credits. Paired with ``equity_curve`` (same index),
    this is what lets ``performance.PerformanceCalculator`` report percent
    invested / percent cash without re-deriving a position's market value
    from price history a second time."""

    orders: tuple[OrderRecord, ...]
    fills: tuple[FillRecord, ...]
    risk_decisions: tuple[DailyRiskDecisions, ...]

    stop_exits: tuple[StopExit, ...] = ()
    """Every position a stop took out, in order. Empty when the engine ran
    with no stop policy -- which is not the same as a policy that never
    fired, so a reader can tell the two apart."""

    stop_checks_skipped: dict[str, int] = field(default_factory=dict)
    """instrument -> sessions where a holding could not be stop-checked
    because it had no bar that day (suspension, halt, missing data). Recorded
    rather than silently passed over: an unprotected session is a fact about
    the run, and a stop that cannot see a price must not invent a breach."""

    share_adjustments: tuple[ShareAdjustment, ...] = ()
    """Every split and bonus applied to a held position, in order. Empty when
    the engine ran without a corporate-action provider."""

    share_adjustments_skipped: dict[str, int] = field(default_factory=dict)
    """instrument -> corporate actions on a held position whose share factor
    could not be derived (a rights issue carries no implicit one). Recorded,
    never guessed: the alternative is a position whose share count is quietly
    wrong for the rest of the run."""


class BacktestEngine:
    """Runs one deterministic, single-strategy backtest over a fixed
    sequence of signal dates and an already-computed exposure-target
    series.
    """

    def __init__(
        self,
        calendar: TradingCalendar,
        market_data: MarketDataProvider,
        stock_selector: StockSelector,
        portfolio_constructor: PortfolioConstructor,
        risk_config: RiskConfig,
        cost_model: CostModel,
        circuit_breaker_state_dir: Path,
        corporate_actions: CorporateActionProvider | None = None,
        assumed_spread_bps: float = 10.0,
        correlation_lookback_days: int = 60,
        min_correlation_observations: int = 20,
        rolling_drawdown_window_days: int = 5,
        max_fill_search_days: int = 5,
        min_rebalance_weight_delta: float = 0.0,
        stale_mark_lookback_days: int = 400,
        stopped_trading_sessions: int = 5,
        stale_mark_warn_days: int = 30,
        stop_loss_policy: StopLossPolicy | None = None,
        checkpoint_dir: Path | None = None,
        run_identity: str = "",
        checkpoint_sessions: int = 1,
        liquidate_at_end: bool = False,
        delisting_notices: tuple[DelistingNotice, ...] = (),
    ) -> None:
        self.calendar = calendar
        self.checkpoint_dir = checkpoint_dir
        self.run_identity = run_identity
        self.checkpoint_sessions = checkpoint_sessions
        self.liquidate_at_end = liquidate_at_end
        self.delisting_notices = delisting_notices
        if checkpoint_sessions < 1:
            raise ValueError("checkpoint_sessions must be positive")
        self.market_data = market_data
        self.stock_selector = stock_selector
        self.portfolio_constructor = portfolio_constructor
        self.risk_config = risk_config
        self.cost_model = cost_model
        self.circuit_breaker_state_dir = circuit_breaker_state_dir
        self.corporate_actions = corporate_actions
        self.assumed_spread_bps = assumed_spread_bps
        self.correlation_lookback_days = correlation_lookback_days
        self.min_correlation_observations = min_correlation_observations
        self.rolling_drawdown_window_days = rolling_drawdown_window_days
        # Kept for caller compatibility; execution no longer searches future bars.
        self.max_fill_search_days = max_fill_search_days
        self.stale_mark_lookback_days = stale_mark_lookback_days
        if stopped_trading_sessions < 1:
            raise BacktestEngineError("stopped_trading_sessions must be >= 1")
        self.stopped_trading_sessions = stopped_trading_sessions
        """Consecutive sessions without a bar after which a held stock is
        treated as having stopped trading and is exited at its last real
        trade. Point-in-time: it is known on the day, with no look-ahead."""
        """How far back :meth:`_last_close` will look for a traded price when
        the ordinary window finds none. Covers a suspension, which in Indian
        cash equity routinely runs to months. Valuation only."""

        self.stale_mark_warn_days = stale_mark_warn_days
        self.unadjustable_marks: dict[str, int] = {}
        """instrument -> how many sessions were marked at an unadjusted last
        traded price because a corporate action between the bar and the mark
        date has no derivable factor. Empty for a run that never valued a
        holding through such an event."""

        self.stale_marks: dict[str, int] = {}
        """instrument -> worst mark age in days, for marks older than
        ``stale_mark_warn_days``. Empty for a run that never valued a
        suspended holding, which is the normal case."""

        self.stop_loss_policy = stop_loss_policy
        """Per-position stops (risk/stop_loss.py), or ``None`` to run without
        any -- which is what every backtest did before they existed, and is
        kept available so a run can measure what the stops are worth by
        comparing against it."""

        self.min_rebalance_weight_delta = min_rebalance_weight_delta
        """A position whose weight would move by less than this is left
        untouched (reverted to its current weight, or never opened) rather
        than traded -- a rebalance-threshold control that trims needless
        turnover/cost from tiny drifts. ``0.0`` (default) rebalances every
        proposed change, however small, matching Phase 11's original
        behavior."""

    def run(
        self,
        strategy_name: str,
        exposure_targets: dict[dt.date, AllocationTarget],
        signal_dates: list[dt.date],
        initial_equity: float,
    ) -> BacktestResult:
        if not signal_dates:
            raise BacktestEngineError("signal_dates must not be empty")
        if list(signal_dates) != sorted(signal_dates):
            raise BacktestEngineError("signal_dates must be ascending")
        if len(signal_dates) != len(set(signal_dates)):
            # A duplicate signal date would otherwise sort into itself and
            # slip past the ascending check above, then replay the same
            # session's decision twice -- the same failure mode as a
            # duplicated broker order response. Rejected here, not just in
            # the duplicate-order stress scenario, because this is a real
            # input-validation gap, not only a test fixture.
            raise BacktestEngineError("signal_dates must not contain duplicates")
        if initial_equity <= 0:
            raise BacktestEngineError(f"initial_equity must be > 0, got {initial_equity}")
        missing = [day for day in signal_dates if day not in exposure_targets]
        if missing:
            raise BacktestEngineError(f"exposure_targets is missing entries for: {missing}")

        state_path = self.circuit_breaker_state_dir / f"{strategy_name}.json"
        identity = hashlib.sha256(
            json.dumps(
                {
                    "run": self.run_identity,
                    "strategy": strategy_name,
                    "dates": [day.isoformat() for day in signal_dates],
                    "equity": initial_equity,
                    "targets": checkpoint.encode(exposure_targets),
                    "liquidate": self.liquidate_at_end,
                    "delisting_notices": checkpoint.encode(self.delisting_notices),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        checkpoint_path = (
            self.checkpoint_dir / f"{strategy_name}.{signal_dates[0]}.session.json"
            if self.checkpoint_dir
            else None
        )
        restored = checkpoint.load(checkpoint_path, identity) if checkpoint_path else None
        if state_path.exists():
            state_path.unlink()
        if restored and restored["breaker"] is not None:
            atomic_write(state_path, restored["breaker"])
        circuit_breaker = CircuitBreaker(self.risk_config, state_path)
        risk_manager = RiskManager(self.risk_config, circuit_breaker)

        cash = initial_equity
        holdings: dict[str, int] = {}
        avg_price: dict[str, float] = {}
        cost_basis: dict[str, float] = {}
        equity_history: list[float] = []
        equity_points: dict[dt.date, float] = {}
        cash_points: dict[dt.date, float] = {}
        regime_points: dict[dt.date, str] = {}
        confidence_points: dict[dt.date, float] = {}
        turnover_points: dict[dt.date, float] = {}
        positions_history: dict[dt.date, dict[str, int]] = {}
        orders: list[OrderRecord] = []
        fills: list[FillRecord] = []
        trade_log_rows: list[dict[str, object]] = []
        risk_decision_records: list[DailyRiskDecisions] = []
        stop_exits: list[StopExit] = []
        stop_checks_skipped: dict[str, int] = {}
        share_adjustments: list[ShareAdjustment] = []
        share_adjustments_skipped: dict[str, int] = {}

        current_target = empty_portfolio(signal_dates[0], AllocationRegime.NORMAL_RISK)

        completed_sessions = 0
        if restored:
            cash = restored["cash"]
            holdings = restored["holdings"]
            avg_price = restored["avg_price"]
            cost_basis = restored["cost_basis"]
            equity_history = restored["equity_history"]
            equity_points = restored["equity_points"]
            cash_points = restored["cash_points"]
            regime_points = restored["regime_points"]
            confidence_points = restored["confidence_points"]
            turnover_points = restored["turnover_points"]
            positions_history = restored["positions_history"]
            orders = restored["orders"]
            fills = restored["fills"]
            trade_log_rows = restored["trade_log_rows"]
            risk_decision_records = restored["risk_decision_records"]
            stop_exits = restored["stop_exits"]
            stop_checks_skipped = restored["stop_checks_skipped"]
            share_adjustments = restored["share_adjustments"]
            share_adjustments_skipped = restored["share_adjustments_skipped"]
            current_target = restored["current_target"]
            completed_sessions = restored["completed_sessions"]
            self.stale_marks = restored["stale_marks"]
            self.unadjustable_marks = restored["unadjustable_marks"]
            if not 0 <= completed_sessions <= len(signal_dates):
                raise BacktestEngineError("Invalid checkpoint session count")
            print(
                f"resumed {strategy_name} {signal_dates[0]}: "
                f"{completed_sessions}/{len(signal_dates)} sessions",
                flush=True,
            )

        for session_index, signal_date in enumerate(signal_dates):
            if session_index < completed_sessions:
                continue
            exposure_target = exposure_targets[signal_date]
            regime_points[signal_date] = exposure_target.regime.value
            confidence_points[signal_date] = exposure_target.confidence

            candidates = self.stock_selector.select(signal_date)
            equity_at_signal = self._mark_to_market(cash, holdings, signal_date)
            equity_history.append(equity_at_signal)

            proposed = self.portfolio_constructor.construct(
                candidates,
                exposure_target,
                signal_date,
                equity_at_signal,
                current_portfolio=current_target,
            )

            risk_state = self._build_risk_state(
                proposed, equity_at_signal, signal_date, equity_history
            )
            decisions = risk_manager.evaluate(proposed, risk_state, current=current_target)
            risk_decision_records.append(DailyRiskDecisions(signal_date, tuple(decisions)))

            executed_target = _apply_risk_decisions(proposed, decisions, current_target)
            if self.min_rebalance_weight_delta > 0:
                executed_target = _apply_rebalance_threshold(
                    executed_target, current_target, self.min_rebalance_weight_delta
                )

            notices = {n.instrument_id: n for n in self.delisting_notices if n.active(signal_date)}
            if notices:
                positions = tuple(
                    p for p in executed_target.positions if p.instrument_id not in notices
                )
                gross = sum(p.target_weight for p in positions)
                executed_target = TargetPortfolio(
                    as_of=executed_target.as_of,
                    positions=positions,
                    cash_weight=round(1 - gross, 12),
                    regime=executed_target.regime,
                    gross_exposure=round(gross, 12),
                )
            trades = list(required_trades(executed_target, current_target))
            # Retry against actual holdings even if a previous unfilled exit
            # already removed the name from the target portfolio.
            traded_ids = {t.instrument_id for t in trades if t.action is TradeAction.EXIT}
            for iid in sorted(notices):
                if holdings.get(iid, 0) > 0 and iid not in traded_ids:
                    weight = current_target.weight_for(iid) if current_target else 0.0
                    trades.append(RequiredTrade(iid, weight, 0.0, -weight, TradeAction.EXIT))
            turnover_points[signal_date] = sum(abs(trade.delta_weight) for trade in trades)

            execution_date = self.calendar.next_trading_day(signal_date)

            # Entitlement belongs to holders entering the ex-date, not to
            # buyers at its open. Snapshot before both share restatements and
            # rebalance fills; an ex-date seller keeps the entitlement.
            # Cash timing remains the explicit ex-date approximation until
            # payment-date entitlement accounting is supplied and verified.
            dividend_credit = self.dividend_cash_credit(holdings, execution_date)

            # Before this session's fills. A split or bonus is effective at
            # the open of its ex-date, so the rebalance below must transact
            # against the restated share count -- not the count from
            # yesterday, at today's already-restated price.
            share_adjustments.extend(
                self.apply_share_adjustments(
                    holdings,
                    avg_price,
                    cost_basis,
                    execution_date,
                    share_adjustments_skipped,
                )
            )

            for trade in trades:
                if trade.action is TradeAction.HOLD:
                    continue
                order = OrderRecord(
                    signal_date=signal_date,
                    execution_date=execution_date,
                    instrument_id=trade.instrument_id,
                    action=trade.action,
                    current_weight=trade.current_weight,
                    target_weight=trade.target_weight,
                    delta_weight=trade.delta_weight,
                )
                orders.append(order)

                fill = self._execute(order, cash, holdings, equity_at_signal)
                if fill is None:
                    continue
                fills.append(fill)
                cash = self._apply_fill(fill, cash, holdings, avg_price, cost_basis)
                row = _trade_log_row(fill)
                if trade.instrument_id in notices and fill.side is TradeSide.SELL:
                    notice = notices[trade.instrument_id]
                    row.update(
                        exit_reason="announced_delisting_exit",
                        event_known_on=notice.known_on.isoformat(),
                        event_source_url=notice.source_url,
                    )
                trade_log_rows.append(row)

            cash += dividend_credit

            # A held stock that has stopped trading -- a merger, a scheme, a
            # delisting -- can never be sold at a new price, and holding it
            # to the fold end used to abort the whole run. After
            # stopped_trading_sessions sessions without a bar it is exited at
            # its last real trade. After the fills, so the day's own order for
            # it (which found no bar) cannot collide with the exit.
            cash, executed_target = self._exit_stopped_trading(
                signal_date=signal_date,
                execution_date=execution_date,
                cash=cash,
                holdings=holdings,
                avg_price=avg_price,
                cost_basis=cost_basis,
                current_target=executed_target,
                orders=orders,
                fills=fills,
                trade_log_rows=trade_log_rows,
                force=False,
            )

            # Stops resolve within the execution session, after the
            # rebalance and before the close is marked. They are resting
            # orders: the decision to protect every holding at -3% was made
            # before the session opened, so a fill inside it is not
            # look-ahead -- only *when* it filled is being resolved, the same
            # argument _next_open makes for a rebalance fill.
            cash, executed_target = self._apply_stops(
                signal_date=signal_date,
                execution_date=execution_date,
                cash=cash,
                holdings=holdings,
                avg_price=avg_price,
                cost_basis=cost_basis,
                current_target=executed_target,
                orders=orders,
                fills=fills,
                trade_log_rows=trade_log_rows,
                stop_exits=stop_exits,
                skipped=stop_checks_skipped,
            )

            # A scheduled fold-end close is known before this session. Record
            # actual sales and costs rather than silently turning holdings into cash.
            if self.liquidate_at_end and signal_date == signal_dates[-1]:
                for instrument_id in sorted(holdings):
                    quantity = holdings[instrument_id]
                    if quantity <= 0:
                        continue
                    bar = self._session_bar(instrument_id, execution_date)
                    if bar is None:
                        # Stopped trading inside the last few sessions of the
                        # fold, before the in-session pass would have caught
                        # it. Same rule: its last real trade.
                        cash, executed_target = self._exit_stopped_trading(
                            signal_date=signal_date,
                            execution_date=execution_date,
                            cash=cash,
                            holdings=holdings,
                            avg_price=avg_price,
                            cost_basis=cost_basis,
                            current_target=executed_target,
                            orders=orders,
                            fills=fills,
                            trade_log_rows=trade_log_rows,
                            force=True,
                            only=instrument_id,
                        )
                        continue
                    price = float(bar.close)
                    adv, vol = self._market_stats(instrument_id, signal_date)
                    cost = self.cost_model.estimate_execution_cost(
                        instrument_id,
                        TradeSide.SELL,
                        quantity,
                        price,
                        execution_date,
                        spread_bps=self.assumed_spread_bps,
                        avg_daily_value=adv,
                        volatility=vol,
                    )
                    weight = executed_target.weight_for(instrument_id)
                    order = OrderRecord(
                        signal_date,
                        execution_date,
                        instrument_id,
                        TradeAction.EXIT,
                        weight,
                        0.0,
                        -weight,
                    )
                    fill = FillRecord(order, TradeSide.SELL, quantity, price, cost)
                    orders.append(order)
                    fills.append(fill)
                    cash = self._apply_fill(fill, cash, holdings, avg_price, cost_basis)
                    row = _trade_log_row(fill)
                    row["exit_reason"] = "fold_end_liquidation"
                    trade_log_rows.append(row)
                executed_target = empty_portfolio(signal_date, exposure_target.regime)

            equity_at_execution = self._mark_to_market(cash, holdings, execution_date)
            equity_points[execution_date] = equity_at_execution
            positions_history[execution_date] = dict(holdings)
            cash_points[execution_date] = cash

            current_target = executed_target

            if checkpoint_path and (
                (session_index + 1) % self.checkpoint_sessions == 0
                or session_index + 1 == len(signal_dates)
            ):
                checkpoint.save(
                    checkpoint_path,
                    identity,
                    {
                        "completed_sessions": session_index + 1,
                        "cash": cash,
                        "holdings": holdings,
                        "avg_price": avg_price,
                        "cost_basis": cost_basis,
                        "equity_history": equity_history,
                        "equity_points": equity_points,
                        "cash_points": cash_points,
                        "regime_points": regime_points,
                        "confidence_points": confidence_points,
                        "turnover_points": turnover_points,
                        "positions_history": positions_history,
                        "orders": orders,
                        "fills": fills,
                        "trade_log_rows": trade_log_rows,
                        "risk_decision_records": risk_decision_records,
                        "stop_exits": stop_exits,
                        "stop_checks_skipped": stop_checks_skipped,
                        "share_adjustments": share_adjustments,
                        "share_adjustments_skipped": share_adjustments_skipped,
                        "current_target": current_target,
                        "breaker": state_path.read_text(encoding="utf-8")
                        if state_path.exists()
                        else None,
                        "stale_marks": self.stale_marks,
                        "unadjustable_marks": self.unadjustable_marks,
                    },
                )

        equity_curve = pd.Series(equity_points, dtype=float).sort_index()
        trade_log = (
            pd.DataFrame(trade_log_rows)
            if trade_log_rows
            else pd.DataFrame(
                columns=[
                    "signal_date",
                    "execution_date",
                    "instrument_id",
                    "side",
                    "quantity",
                    "fill_price",
                    "gross_value",
                    "cost",
                    "net_value",
                ]
            )
        )

        return BacktestResult(
            strategy_name=strategy_name,
            equity_curve=equity_curve,
            trade_log=trade_log,
            regime_history=pd.Series(regime_points).sort_index(),
            confidence_history=pd.Series(confidence_points, dtype=float).sort_index(),
            turnover_history=pd.Series(turnover_points, dtype=float).sort_index(),
            positions_history=positions_history,
            cash_history=pd.Series(cash_points, dtype=float).sort_index(),
            orders=tuple(orders),
            fills=tuple(fills),
            risk_decisions=tuple(risk_decision_records),
            stop_exits=tuple(stop_exits),
            stop_checks_skipped=dict(stop_checks_skipped),
            share_adjustments=tuple(share_adjustments),
            share_adjustments_skipped=dict(share_adjustments_skipped),
        )

    # -- stops --------------------------------------------------------------

    def _session_bar(self, instrument_id: str, session_date: dt.date) -> DailyBar | None:
        """That instrument's own bar for exactly ``session_date``, RAW.

        RAW, not ADJUSTED, for the same reason :meth:`_next_open` is: a stop
        level is compared against, and fills at, prices a trader saw on the
        day. An adjusted bar restates the session into the window's terms and
        would move the level.

        ``None`` when the instrument did not trade that session. The caller
        records that rather than treating it as "no breach", because an
        unprotected session is a fact about the run.
        """
        try:
            bars = self.market_data.get_equity_bars(
                instrument_id, session_date, session_date, price_basis=PriceBasis.RAW
            )
        except DataNotAvailableError:
            return None
        if not bars:
            return None
        bar = bars[0]
        return bar if bar.session_date == session_date else None

    def _apply_stops(
        self,
        *,
        signal_date: dt.date,
        execution_date: dt.date,
        cash: float,
        holdings: dict[str, int],
        avg_price: dict[str, float],
        cost_basis: dict[str, float],
        current_target: TargetPortfolio,
        orders: list[OrderRecord],
        fills: list[FillRecord],
        trade_log_rows: list[dict[str, object]],
        stop_exits: list[StopExit],
        skipped: dict[str, int],
    ) -> tuple[float, TargetPortfolio]:
        """Sell every holding whose stop fired during ``execution_date``.

        Returns the cash balance after the exits and the target portfolio
        with the exited names removed. Dropping them matters: leave a sold
        name in ``current_target`` and the next session diffs against a
        position that no longer exists, proposing a sell of nothing and, worse,
        reporting a weight the book does not hold.

        A stopped-out name is not blocked from being bought back. If the
        selector still ranks it tomorrow it is re-entered at tomorrow's open,
        with a fresh entry price and therefore a fresh stop. That is a
        deliberate choice, not an oversight -- a cooling-off rule is a
        separate strategy decision, and inventing one here would quietly
        change what the backtest measures.
        """
        policy = self.stop_loss_policy
        if policy is None or not policy.enabled or not holdings:
            return cash, current_target

        for instrument_id in sorted(holdings):
            quantity = holdings[instrument_id]
            entry_price = avg_price.get(instrument_id, 0.0)
            basis = cost_basis.get(instrument_id, 0.0)
            if quantity <= 0 or entry_price <= 0 or basis <= 0:
                continue

            bar = self._session_bar(instrument_id, execution_date)
            if bar is None:
                skipped[instrument_id] = skipped.get(instrument_id, 0) + 1
                continue

            avg_daily_value, volatility = self._market_stats(instrument_id, signal_date)

            def net_sale_value(
                price: float,
                _id: str = instrument_id,
                _quantity: int = quantity,
                _adv: float = avg_daily_value,
                _vol: float = volatility,
            ) -> float:
                return self.cost_model.estimate_execution_cost(
                    _id,
                    TradeSide.SELL,
                    _quantity,
                    price,
                    execution_date,
                    spread_bps=self.assumed_spread_bps,
                    avg_daily_value=_adv,
                    volatility=_vol,
                ).net_value

            breach = evaluate_stop(
                instrument_id,
                entry_price=entry_price,
                cost_basis=basis,
                session_high=float(bar.high),
                session_low=float(bar.low),
                session_open=float(bar.open),
                reference_price=float(bar.close),
                net_sale_value=net_sale_value,
                policy=policy,
            )
            if breach is None:
                continue

            current_weight = current_target.weight_for(instrument_id)
            order = OrderRecord(
                signal_date=signal_date,
                execution_date=execution_date,
                instrument_id=instrument_id,
                action=TradeAction.EXIT,
                current_weight=current_weight,
                target_weight=0.0,
                delta_weight=-current_weight,
            )
            execution_cost = self.cost_model.estimate_execution_cost(
                instrument_id,
                TradeSide.SELL,
                quantity,
                breach.fill_price,
                execution_date,
                spread_bps=self.assumed_spread_bps,
                avg_daily_value=avg_daily_value,
                volatility=volatility,
            )
            fill = FillRecord(
                order=order,
                side=TradeSide.SELL,
                quantity=quantity,
                fill_price=breach.fill_price,
                execution_cost=execution_cost,
            )
            orders.append(order)
            fills.append(fill)
            stop_exits.append(
                StopExit(
                    execution_date=execution_date,
                    breach=breach,
                    quantity=quantity,
                    cost_basis=basis,
                    net_proceeds=execution_cost.net_value,
                )
            )
            cash = self._apply_fill(fill, cash, holdings, avg_price, cost_basis)
            row = _trade_log_row(fill)
            # Tagged like fold-end and delisting exits. Untagged, a stop is
            # indistinguishable from a rebalance sell and nobody can count
            # how often the rule actually fired.
            row["exit_reason"] = breach.reason.value
            trade_log_rows.append(row)
            current_target = _without_position(current_target, instrument_id)

        return cash, current_target

    # -- pricing / market facts --------------------------------------------

    def _last_raw_bar(self, instrument_id: str, as_of: dt.date) -> DailyBar | None:
        """The last bar that actually printed on or before ``as_of``, RAW.

        RAW because it is used as an exit price: the price a trade really
        happened at, not one restated into later terms.
        """
        start = as_of - dt.timedelta(days=self.stale_mark_lookback_days)
        try:
            bars = self.market_data.get_equity_bars(
                instrument_id, start, as_of, price_basis=PriceBasis.RAW
            )
        except DataNotAvailableError:
            return None
        return bars[-1] if bars else None

    def _exit_stopped_trading(
        self,
        *,
        signal_date: dt.date,
        execution_date: dt.date,
        cash: float,
        holdings: dict[str, int],
        avg_price: dict[str, float],
        cost_basis: dict[str, float],
        current_target: TargetPortfolio,
        orders: list[OrderRecord],
        fills: list[FillRecord],
        trade_log_rows: list[dict[str, object]],
        force: bool,
        only: str | None = None,
    ) -> tuple[float, TargetPortfolio]:
        """Exit held stocks that have stopped trading, at their last real trade.

        A merger, a scheme of arrangement or a delisting ends a stock's
        trading. JSLHISAR last traded on 2023-03-08 before merging into JSL;
        every strategy still held it at the fold end on 2023-06-05, with no
        bar to sell into, and all ten runs aborted. 142 names in the liquid
        universe stop trading inside the backtest window.

        The policy, chosen deliberately as an approximation: sell at the last
        price that actually traded, tagged ``stopped_trading_last_close`` so
        every instance can be audited. For a merger that is close to what a
        holder had -- acquirer shares worth about that much at the time. For
        an insolvency it is generous, because holders there typically
        received nearly nothing; those names rarely stay liquid enough to be
        held, since the ranking drops them long before.

        A stock counts as stopped after ``stopped_trading_sessions``
        consecutive sessions with no bar, which is knowable on the day. A
        short suspension that ends sooner resumes normally. ``force`` skips
        that test at a fold end, where the position must be closed today.
        """
        ids = [only] if only is not None else sorted(holdings)
        for instrument_id in ids:
            quantity = holdings.get(instrument_id, 0)
            if quantity <= 0:
                continue
            if self._session_bar(instrument_id, execution_date) is not None:
                continue
            last = self._last_raw_bar(instrument_id, execution_date)
            if last is None:
                if force:
                    raise BacktestEngineError(
                        f"Cannot liquidate {instrument_id} at fold end {execution_date}: "
                        "it has never traded within the lookback; no price exists"
                    )
                continue
            silent = len(self.calendar.trading_days_between(last.session_date, execution_date)) - 1
            if not force and silent < self.stopped_trading_sessions:
                continue

            price = float(last.close)
            adv, vol = self._market_stats(instrument_id, signal_date)
            cost = self.cost_model.estimate_execution_cost(
                instrument_id,
                TradeSide.SELL,
                quantity,
                price,
                execution_date,
                spread_bps=self.assumed_spread_bps,
                avg_daily_value=adv,
                volatility=vol,
            )
            weight = current_target.weight_for(instrument_id) if current_target else 0.0
            order = OrderRecord(
                signal_date, execution_date, instrument_id, TradeAction.EXIT, weight, 0.0, -weight
            )
            fill = FillRecord(order, TradeSide.SELL, quantity, price, cost)
            orders.append(order)
            fills.append(fill)
            cash = self._apply_fill(fill, cash, holdings, avg_price, cost_basis)
            row = _trade_log_row(fill)
            row["exit_reason"] = "stopped_trading_last_close"
            row["last_trade_date"] = last.session_date.isoformat()
            trade_log_rows.append(row)
            current_target = _without_position(current_target, instrument_id)
        return cash, current_target

    def _last_close(self, instrument_id: str, as_of: dt.date, lookback_days: int = 15) -> float:
        """The most recent traded close, for marking a holding to market.

        Valuation only -- execution prices come from :meth:`_next_open`. That
        distinction is what makes the fallback below correct rather than
        convenient: a stale price here misstates reported equity, but it can
        never be mistaken for a price something traded at.

        **Why this searches twice.** An Indian equity can stop trading for
        months while still being held: suspension, a surveillance move, or
        repeated circuit-limit days with no crossing trades. ADANITRANS last
        traded 2021-06-08 and next traded 2021-09-13, and the 15-day window
        missed 2021-06-24 by a single day -- which killed four 32-fold runs
        eleven folds in.

        Refusing to value a held position is the wrong answer to that. The
        position exists, the portfolio has to be worth something, and the
        convention every fund and regulator uses for a suspended holding is
        its last traded price until a fair-value adjustment. So the narrow
        window stays as the fast path for the overwhelmingly common case, and
        a wide one catches suspensions. Only a genuine absence of any price
        ever still raises.

        Marks older than ``stale_mark_warn_days`` are counted in
        ``stale_marks`` so a run that leaned on this is auditable afterwards
        rather than silently equivalent to one that did not.
        """
        for window in (lookback_days, self.stale_mark_lookback_days):
            if window < lookback_days:
                continue
            start = as_of - dt.timedelta(days=window)
            try:
                bars = self.market_data.get_equity_bars(
                    instrument_id, start, as_of, price_basis=PriceBasis.ADJUSTED
                )
            except DataNotAvailableError:
                continue
            except ValueError:
                # An underivable corporate action sits between a bar in this
                # window and as_of -- a demerger or similar whose price factor
                # is not computable from the action terms. NSE:VEDL demerged
                # on 2026-04-30 and killed four 33-fold runs on the final
                # fold, after ten hours each.
                #
                # The position is held and has to be worth something, so it is
                # marked at its last TRADED price, unadjusted. That is the
                # most defensible number available: the adjustment genuinely
                # cannot be computed, and refusing to value a holding is worse
                # than valuing it on the terms the market last actually
                # printed. Counted, never silent -- a run that leaned on this
                # is not equivalent to one that did not.
                self.unadjustable_marks[instrument_id] = (
                    self.unadjustable_marks.get(instrument_id, 0) + 1
                )
                try:
                    bars = self.market_data.get_equity_bars(
                        instrument_id, start, as_of, price_basis=PriceBasis.RAW
                    )
                except DataNotAvailableError:
                    continue
            if not bars:
                continue
            age = (as_of - bars[-1].session_date).days
            if age > self.stale_mark_warn_days:
                self.stale_marks[instrument_id] = max(self.stale_marks.get(instrument_id, 0), age)
            return float(bars[-1].close)

        raise BacktestEngineError(f"no price data for {instrument_id} on or before {as_of}")

    def _next_open(self, instrument_id: str, execution_date: dt.date) -> float | None:
        """Use only this session's observed open; never book a future price today.

        An unfilled order can be reconsidered on a later signal. Searching
        ahead without moving the fill's accounting date creates look-ahead.
        """
        bar = self._session_bar(instrument_id, execution_date)
        return float(bar.open) if bar is not None else None

    def _market_stats(
        self, instrument_id: str, as_of: dt.date, lookback_days: int = 20
    ) -> tuple[float, float]:
        """``(avg_daily_value_inr, annualized_volatility)`` from raw
        traded-value/return history, independent of stock selection's own
        factor computation -- needed for instruments being exited that may
        no longer be in the current candidate ranking.
        """
        return market_liquidity_stats(self.market_data, instrument_id, as_of, lookback_days)

    def _max_pairwise_correlation(
        self, instrument_ids: list[str], as_of: dt.date
    ) -> tuple[float, tuple[str, str]] | None:
        if len(instrument_ids) < 2:
            return None
        start = as_of - dt.timedelta(days=self.correlation_lookback_days * 3)
        returns: dict[str, pd.Series] = {}
        for instrument_id in instrument_ids:
            try:
                bars = self.market_data.get_equity_bars(
                    instrument_id, start, as_of, price_basis=PriceBasis.ADJUSTED
                )
            except DataNotAvailableError:
                continue
            if len(bars) < 2:
                continue
            closes = pd.Series(
                [float(bar.close) for bar in bars], index=[bar.session_date for bar in bars]
            ).tail(self.correlation_lookback_days + 1)
            series = closes.pct_change().dropna()
            if not series.empty:
                returns[instrument_id] = series
        if len(returns) < 2:
            return None
        frame = pd.DataFrame(returns).dropna(how="any")
        if len(frame) < self.min_correlation_observations:
            return None
        correlation = frame.corr()
        best: tuple[float, tuple[str, str]] | None = None
        ids = sorted(returns)
        for i, first in enumerate(ids):
            for second in ids[i + 1 :]:
                value = cast(float, correlation.loc[first, second])
                if pd.isna(value):
                    continue
                if best is None or value > best[0]:
                    best = (value, (first, second))
        return best

    # -- risk state ---------------------------------------------------------

    def _build_risk_state(
        self,
        proposed: TargetPortfolio,
        equity: float,
        as_of: dt.date,
        equity_history: list[float],
    ) -> PortfolioRiskState:
        position_risks = []
        for position in proposed.positions:
            avg_daily_value, _volatility = self._market_stats(position.instrument_id, as_of)
            position_risks.append(
                PositionRisk(
                    instrument_id=position.instrument_id,
                    sector=position.sector,
                    avg_daily_value_inr=avg_daily_value,
                    quote_age_seconds=0.0,
                    spread_bps=self.assumed_spread_bps,
                )
            )

        correlation = self._max_pairwise_correlation(
            [position.instrument_id for position in proposed.positions], as_of
        )

        daily_pnl_pct = 0.0
        rolling_pnl_pct = 0.0
        if len(equity_history) >= 2:
            previous = equity_history[-2]
            daily_pnl_pct = max(0.0, 1.0 - equity_history[-1] / previous) if previous > 0 else 0.0
            window = equity_history[-self.rolling_drawdown_window_days :]
            window_peak = max(window)
            rolling_pnl_pct = (
                max(0.0, 1.0 - equity_history[-1] / window_peak) if window_peak > 0 else 0.0
            )
        all_time_peak = max(equity_history) if equity_history else equity
        peak_to_trough = max(0.0, 1.0 - equity / all_time_peak) if all_time_peak > 0 else 0.0

        return PortfolioRiskState(
            as_of=dt.datetime.combine(as_of, dt.time(15, 30), tzinfo=dt.UTC),
            equity=equity,
            positions=tuple(position_risks),
            daily_pnl_pct=daily_pnl_pct,
            rolling_pnl_pct=rolling_pnl_pct,
            peak_to_trough_drawdown_pct=peak_to_trough,
            daily_turnover_pct_so_far=0.0,
            max_pairwise_correlation=correlation[0] if correlation else None,
            correlated_pair=correlation[1] if correlation else None,
            system_healthy=True,
            system_detail=None,
            broker_connected=True,
            broker_detail=None,
        )

    # -- execution ------------------------------------------------------------

    def _execute(
        self,
        order: OrderRecord,
        cash: float,
        holdings: dict[str, int],
        equity_at_signal: float,
    ) -> FillRecord | None:
        bar = self._session_bar(order.instrument_id, order.execution_date)
        if bar is None:
            return None
        # Existing holdings can exit a trade-for-trade series. Do not create
        # new exposure there, even if yesterday's EQ signal requested a buy.
        if order.action is not TradeAction.EXIT and order.delta_weight > 0:
            if bar.trading_series is not None and bar.trading_series != "EQ":
                return None
        fill_price = float(bar.open)
        if fill_price is None or fill_price <= 0:
            return None

        held = holdings.get(order.instrument_id, 0)
        if order.action is TradeAction.EXIT:
            quantity = held
            side = TradeSide.SELL
        elif order.delta_weight > 0:
            side = TradeSide.BUY
            notional = order.delta_weight * equity_at_signal
            quantity = math.floor(notional / fill_price)
            quantity = min(quantity, math.floor(max(cash, 0.0) / fill_price))
        else:
            side = TradeSide.SELL
            notional = abs(order.delta_weight) * equity_at_signal
            quantity = math.floor(notional / fill_price)
            quantity = min(quantity, held)

        if quantity <= 0:
            return None

        avg_daily_value, volatility = self._market_stats(order.instrument_id, order.signal_date)
        execution_cost = self.cost_model.estimate_execution_cost(
            order.instrument_id,
            side,
            quantity,
            fill_price,
            order.execution_date,
            spread_bps=self.assumed_spread_bps,
            avg_daily_value=avg_daily_value,
            volatility=volatility,
        )
        return FillRecord(
            order=order,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            execution_cost=execution_cost,
        )

    @staticmethod
    def _apply_fill(
        fill: FillRecord,
        cash: float,
        holdings: dict[str, int],
        avg_price: dict[str, float] | None = None,
        cost_basis: dict[str, float] | None = None,
    ) -> float:
        """Post one fill to the ledger and return the new cash balance.

        ``avg_price`` and ``cost_basis`` are what the stop rules need and
        nothing else reads, so they are optional: a caller that only wants
        the cash/quantity arithmetic (``backtest/stress_test.py``) passes
        neither and behaves exactly as before.

        The two are deliberately different quantities. ``avg_price`` is the
        weighted average *price* paid per share -- what the 3% hard stop is
        measured from, because that is the number on the screen and the
        number a broker's stop order would reference. ``cost_basis`` is the
        total *cash* the position consumed, buy-side charges included -- what
        profitability is measured against, because a position is not in
        profit until it has earned back what it cost to open.
        """
        instrument_id = fill.order.instrument_id
        if fill.side is TradeSide.BUY:
            cash -= fill.execution_cost.net_value
            held = holdings.get(instrument_id, 0)
            new_quantity = held + fill.quantity
            holdings[instrument_id] = new_quantity
            if avg_price is not None:
                previous = avg_price.get(instrument_id, 0.0)
                avg_price[instrument_id] = (
                    held * previous + fill.quantity * fill.fill_price
                ) / new_quantity
            if cost_basis is not None:
                cost_basis[instrument_id] = (
                    cost_basis.get(instrument_id, 0.0) + fill.execution_cost.net_value
                )
        else:
            cash += fill.execution_cost.net_value
            held = holdings.get(instrument_id, 0)
            remaining = held - fill.quantity
            if remaining <= 0:
                holdings.pop(instrument_id, None)
                if avg_price is not None:
                    avg_price.pop(instrument_id, None)
                if cost_basis is not None:
                    cost_basis.pop(instrument_id, None)
            else:
                holdings[instrument_id] = remaining
                # A partial sale retires its share of the basis and leaves the
                # average price alone: selling half a position does not change
                # what the other half was bought at.
                if cost_basis is not None and held > 0:
                    cost_basis[instrument_id] = cost_basis.get(instrument_id, 0.0) * (
                        remaining / held
                    )
        return cash

    def _mark_to_market(self, cash: float, holdings: dict[str, int], as_of: dt.date) -> float:
        equity = cash
        for instrument_id, quantity in holdings.items():
            equity += quantity * self._last_close(instrument_id, as_of)
        return equity

    def apply_share_adjustments(
        self,
        holdings: dict[str, int],
        avg_price: dict[str, float],
        cost_basis: dict[str, float],
        execution_date: dt.date,
        skipped: dict[str, int],
    ) -> list[ShareAdjustment]:
        """Restate held share counts for splits and bonuses going ex today.

        Without this a bonus or a split is recorded as a catastrophic loss.
        IEX issued a 2:1 bonus with ex-date 2021-12-03: the price went from
        729.55 to 255.75 and every holder's share count tripled. An engine
        that adjusts neither sells the *old* share count at the *new* price
        and books a 67.8% loss on a position that was economically flat.
        Across one 33-fold run that fabricated 4.0 crore of losses over 50
        events -- an order of magnitude larger than the strategy's entire
        reported P&L, and the single largest error in the backtest.

        Three ledgers move together, and they do not move the same way:

        ``holdings``
            multiplied by the share factor. More shares.
        ``avg_price``
            multiplied by the *price* factor. Each share cost proportionally
            less, because the same money now buys more of them.
        ``cost_basis``
            **unchanged.** A split costs nothing and earns nothing; the cash
            that left the account to open the position did not move. This is
            why the two are kept as separate ledgers rather than one derived
            from the other.

        The share factor is the reciprocal of
        :meth:`CorporateAction.price_adjustment_factor`, deliberately rather
        than a second reading of ``ratio_new``/``ratio_old``. That method
        already encodes the difference between a split (10-for-2 restates
        price by 2/10) and a bonus (2-for-1 restates by 1/3), and honours an
        ``explicit_price_factor`` where one is given. Deriving shares from it
        means the two can never disagree, and a dividend -- factor 1 -- falls
        through as a no-op without being special-cased.

        Applied at the *open* of the ex-date, before the session's fills: the
        adjusted share count is what a rebalance on that day transacts
        against. Fills on the ex-date already price at the adjusted level.

        Fractional entitlements are floored. An exchange pays cash in lieu of
        a fraction; crediting nothing is a small understatement in the
        account's favour, which is the right direction for a backtest.
        """
        if self.corporate_actions is None:
            return []

        applied: list[ShareAdjustment] = []
        for instrument_id in sorted(holdings):
            quantity = holdings.get(instrument_id, 0)
            if quantity <= 0:
                continue
            for action in self.corporate_actions.actions_for(
                instrument_id, execution_date, execution_date
            ):
                if action.action_type is CorporateActionType.DIVIDEND:
                    continue
                if action.action_type not in (CorporateActionType.SPLIT, CorporateActionType.BONUS):
                    # A price factor never specifies replacement shares,
                    # rights subscription or cash consideration. Do not turn
                    # a demerger factor into extra shares of the parent.
                    skipped[instrument_id] = skipped.get(instrument_id, 0) + 1
                    continue
                try:
                    factor = action.price_adjustment_factor()
                except ValueError:
                    # A rights issue has no implicit factor -- subscribing is
                    # a funded decision this engine does not model. Counted
                    # so the run can say how many it passed over.
                    skipped[instrument_id] = skipped.get(instrument_id, 0) + 1
                    continue
                if factor == 1 or factor <= 0:
                    continue

                adjusted = int(Decimal(quantity) / factor)
                if adjusted == quantity:
                    continue
                holdings[instrument_id] = adjusted
                if instrument_id in avg_price:
                    avg_price[instrument_id] = avg_price[instrument_id] * float(factor)
                applied.append(
                    ShareAdjustment(
                        instrument_id=instrument_id,
                        ex_date=execution_date,
                        action_type=str(action.action_type.value),
                        quantity_before=quantity,
                        quantity_after=adjusted,
                        price_factor=float(factor),
                    )
                )
                quantity = adjusted
        return applied

    def dividend_cash_credit(self, holdings: dict[str, int], execution_date: dt.date) -> float:
        """Total cash to credit for dividends going ex on ``execution_date``
        for currently-held positions -- kept as a pure function of the
        ledger so it can be unit-tested without a full backtest run.

        Uses ``CorporateActionProvider.actions_for`` (the interface every
        implementation provides) and filters for dividends itself, rather
        than a concrete implementation's own convenience method -- this
        engine depends only on the ABC, never on
        ``InMemoryCorporateActionProvider`` specifically.
        """
        if self.corporate_actions is None:
            return 0.0
        total = 0.0
        for instrument_id, quantity in holdings.items():
            for action in self.corporate_actions.actions_for(
                instrument_id, execution_date, execution_date
            ):
                is_dividend = action.action_type is CorporateActionType.DIVIDEND
                if is_dividend and action.cash_amount is not None:
                    total += float(action.cash_amount) * quantity
        return total


def _apply_risk_decisions(
    proposed: TargetPortfolio,
    decisions: list[RiskDecision],
    current: TargetPortfolio | None,
) -> TargetPortfolio:
    """Fold risk-manager verdicts into the portfolio actually acted on.

    An approved position is executed at its proposed weight. A rejected
    position that is currently held keeps its *current* weight -- no new
    trade for it, since the veto blocks the proposed change, not the
    status quo. A rejected position that is not currently held is simply
    excluded -- it is never opened. If the resulting total would still
    exceed 1.0 (only possible via a rejection unrelated to sizing, e.g.
    ``MISSING_RISK_DATA``, reintroducing a larger stale weight), every
    position is scaled down proportionally so the no-leverage invariant
    always holds -- the same defense-in-depth principle
    ``portfolio/portfolio_constructor.py`` applies to its own output.
    """
    approved_ids = {decision.instrument_id for decision in decisions if decision.approved}
    current_by_id = {
        position.instrument_id: position for position in (current.positions if current else ())
    }

    final_positions: list[TargetPosition] = []
    for position in proposed.positions:
        if position.instrument_id in approved_ids:
            final_positions.append(position)
        elif position.instrument_id in current_by_id:
            final_positions.append(current_by_id[position.instrument_id])

    gross = sum(position.target_weight for position in final_positions)
    if gross > 1.0:
        scale = 1.0 / gross
        final_positions = [_scale_position(position, scale) for position in final_positions]
        gross = 1.0

    cash_weight = round(1.0 - gross, 12)
    return TargetPortfolio(
        as_of=proposed.as_of,
        positions=tuple(final_positions),
        cash_weight=cash_weight,
        regime=proposed.regime,
        gross_exposure=round(gross, 12),
    )


def _apply_rebalance_threshold(
    executed_target: TargetPortfolio,
    current: TargetPortfolio | None,
    threshold: float,
) -> TargetPortfolio:
    """Reverts any position whose weight would move by less than
    ``threshold`` back to its current weight (or drops it entirely if it
    was never held) -- a small drift is left alone rather than traded.

    Built directly from ``TargetPosition`` objects already present on
    ``executed_target``/``current`` (never from ``RequiredTrade``, which
    lacks the sector/rank/score/binding_constraint fields a
    ``TargetPosition`` needs) so a reverted position keeps its original
    provenance intact.
    """
    current_by_id = {
        position.instrument_id: position for position in (current.positions if current else ())
    }
    executed_ids = {position.instrument_id for position in executed_target.positions}

    final_positions: list[TargetPosition] = []
    for position in executed_target.positions:
        current_position = current_by_id.get(position.instrument_id)
        current_weight = current_position.target_weight if current_position else 0.0
        if abs(position.target_weight - current_weight) < threshold:
            if current_position is not None:
                final_positions.append(current_position)
            # else: a brand-new position smaller than the threshold is
            # simply never opened.
        else:
            final_positions.append(position)

    for instrument_id, current_position in current_by_id.items():
        if instrument_id in executed_ids:
            continue
        # A full exit is itself a weight delta equal to the current
        # weight; below the threshold, the exit is skipped and the
        # position stays put.
        if current_position.target_weight < threshold:
            final_positions.append(current_position)

    gross = sum(position.target_weight for position in final_positions)
    if gross > 1.0:
        # Reverting a position to its (larger) current weight while other
        # positions keep their (already within-budget) executed weight can
        # push the total slightly over 1.0 -- the same edge case
        # `_apply_risk_decisions` defends against, and the same fix:
        # scale everything down proportionally rather than let it through.
        scale = 1.0 / gross
        final_positions = [_scale_position(position, scale) for position in final_positions]
        gross = 1.0

    cash_weight = round(1.0 - gross, 12)
    return TargetPortfolio(
        as_of=executed_target.as_of,
        positions=tuple(final_positions),
        cash_weight=cash_weight,
        regime=executed_target.regime,
        gross_exposure=round(gross, 12),
    )


def _scale_position(position: TargetPosition, scale: float) -> TargetPosition:
    return TargetPosition(
        instrument_id=position.instrument_id,
        symbol=position.symbol,
        target_weight=position.target_weight * scale,
        sector=position.sector,
        rank=position.rank,
        score=position.score,
        binding_constraint=position.binding_constraint,
    )


def _without_position(portfolio: TargetPortfolio, instrument_id: str) -> TargetPortfolio:
    """``portfolio`` with one name removed and its weight returned to cash.

    A :class:`TargetPortfolio` must always sum to exactly 1.0, so a position
    cannot simply be dropped -- its weight has to go somewhere, and cash is
    the only honest place: a stop exit produces cash, not a larger position in
    whatever else was held.
    """
    weight = portfolio.weight_for(instrument_id)
    if weight == 0.0 and instrument_id not in portfolio.instrument_ids:
        return portfolio
    positions = tuple(
        position for position in portfolio.positions if position.instrument_id != instrument_id
    )
    return TargetPortfolio(
        as_of=portfolio.as_of,
        positions=positions,
        cash_weight=portfolio.cash_weight + weight,
        regime=portfolio.regime,
        gross_exposure=portfolio.gross_exposure - weight,
    )


def _trade_log_row(fill: FillRecord) -> dict[str, object]:
    return {
        "signal_date": fill.order.signal_date,
        "execution_date": fill.order.execution_date,
        "instrument_id": fill.order.instrument_id,
        "side": fill.side.value,
        "quantity": fill.quantity,
        "fill_price": fill.fill_price,
        "gross_value": fill.execution_cost.gross_value,
        "cost": fill.execution_cost.total_cost,
        "net_value": fill.execution_cost.net_value,
    }
