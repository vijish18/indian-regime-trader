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

## What's not here

Real bid/ask spread data does not exist in this dataset -- ``assumed_spread_bps``
is a single configured constant standing in for it (a documented V1
simplification, not a fabricated market fact). Position sizing here is a
simple weight-to-quantity conversion, not ``risk/position_sizer.py``'s
stop-distance risk-based reconciliation (Phase 7c, still stubbed).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from backtest.costs import CostModel, ExecutionCostEstimate, TradeSide
from config.models import RiskConfig
from core.regime.allocation import AllocationRegime, AllocationTarget
from data.errors import DataNotAvailableError
from data.interfaces import CorporateActionProvider, MarketDataProvider, TradingCalendar
from data.models import CorporateActionType, PriceBasis
from portfolio.portfolio_constructor import (
    PortfolioConstructor,
    TargetPortfolio,
    TargetPosition,
    TradeAction,
    empty_portfolio,
    required_trades,
)
from risk.circuit_breaker import CircuitBreaker
from risk.portfolio_risk_state import PortfolioRiskState, PositionRisk
from risk.risk_manager import RiskDecision, RiskManager
from universe.stock_selector import StockSelector

_TRADING_DAYS_PER_YEAR = 252


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

    orders: tuple[OrderRecord, ...]
    fills: tuple[FillRecord, ...]
    risk_decisions: tuple[DailyRiskDecisions, ...]


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
    ) -> None:
        self.calendar = calendar
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
        self.max_fill_search_days = max_fill_search_days

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
        if initial_equity <= 0:
            raise BacktestEngineError(f"initial_equity must be > 0, got {initial_equity}")
        missing = [day for day in signal_dates if day not in exposure_targets]
        if missing:
            raise BacktestEngineError(f"exposure_targets is missing entries for: {missing}")

        state_path = self.circuit_breaker_state_dir / f"{strategy_name}.json"
        if state_path.exists():
            state_path.unlink()
        circuit_breaker = CircuitBreaker(self.risk_config, state_path)
        risk_manager = RiskManager(self.risk_config, circuit_breaker)

        cash = initial_equity
        holdings: dict[str, int] = {}
        equity_history: list[float] = []
        equity_points: dict[dt.date, float] = {}
        regime_points: dict[dt.date, str] = {}
        confidence_points: dict[dt.date, float] = {}
        turnover_points: dict[dt.date, float] = {}
        positions_history: dict[dt.date, dict[str, int]] = {}
        orders: list[OrderRecord] = []
        fills: list[FillRecord] = []
        trade_log_rows: list[dict[str, object]] = []
        risk_decision_records: list[DailyRiskDecisions] = []

        current_target = empty_portfolio(signal_dates[0], AllocationRegime.NORMAL_RISK)

        for signal_date in signal_dates:
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

            trades = required_trades(executed_target, current_target)
            turnover_points[signal_date] = sum(abs(trade.delta_weight) for trade in trades)

            execution_date = self.calendar.next_trading_day(signal_date)
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
                cash = self._apply_fill(fill, cash, holdings)
                trade_log_rows.append(_trade_log_row(fill))

            cash += self.dividend_cash_credit(holdings, execution_date)

            equity_at_execution = self._mark_to_market(cash, holdings, execution_date)
            equity_points[execution_date] = equity_at_execution
            positions_history[execution_date] = dict(holdings)

            current_target = executed_target

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
            orders=tuple(orders),
            fills=tuple(fills),
            risk_decisions=tuple(risk_decision_records),
        )

    # -- pricing / market facts --------------------------------------------

    def _last_close(self, instrument_id: str, as_of: dt.date, lookback_days: int = 15) -> float:
        start = as_of - dt.timedelta(days=lookback_days)
        try:
            bars = self.market_data.get_equity_bars(
                instrument_id, start, as_of, price_basis=PriceBasis.ADJUSTED
            )
        except DataNotAvailableError as exc:
            raise BacktestEngineError(
                f"no price data for {instrument_id} on or before {as_of}"
            ) from exc
        if not bars:
            raise BacktestEngineError(f"no price data for {instrument_id} on or before {as_of}")
        return float(bars[-1].close)

    def _next_open(self, instrument_id: str, execution_date: dt.date) -> float | None:
        """The execution-date open, or the first available session's open
        within ``max_fill_search_days`` after it -- a "resting order"
        assumption for a session where the instrument didn't trade, not a
        look-ahead: the order's terms were already fixed before this price
        is read, only *when* it fills is being resolved.
        """
        end = execution_date + dt.timedelta(days=self.max_fill_search_days * 2)
        try:
            bars = self.market_data.get_equity_bars(
                instrument_id, execution_date, end, price_basis=PriceBasis.ADJUSTED
            )
        except DataNotAvailableError:
            return None
        if not bars:
            return None
        return float(bars[0].open)

    def _market_stats(
        self, instrument_id: str, as_of: dt.date, lookback_days: int = 20
    ) -> tuple[float, float]:
        """``(avg_daily_value_inr, annualized_volatility)`` from raw
        traded-value/return history, independent of stock selection's own
        factor computation -- needed for instruments being exited that may
        no longer be in the current candidate ranking.
        """
        start = as_of - dt.timedelta(days=lookback_days * 3)
        try:
            bars = self.market_data.get_equity_bars(
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
            float(returns.std(ddof=0) * math.sqrt(_TRADING_DAYS_PER_YEAR))
            if not returns.empty
            else 0.0
        )
        return avg_daily_value, volatility

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
        peak_to_trough = (
            max(0.0, 1.0 - equity / all_time_peak) if all_time_peak > 0 else 0.0
        )

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
        fill_price = self._next_open(order.instrument_id, order.execution_date)
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

        avg_daily_value, volatility = self._market_stats(
            order.instrument_id, order.signal_date
        )
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
    def _apply_fill(fill: FillRecord, cash: float, holdings: dict[str, int]) -> float:
        instrument_id = fill.order.instrument_id
        if fill.side is TradeSide.BUY:
            cash -= fill.execution_cost.net_value
            holdings[instrument_id] = holdings.get(instrument_id, 0) + fill.quantity
        else:
            cash += fill.execution_cost.net_value
            remaining = holdings.get(instrument_id, 0) - fill.quantity
            if remaining <= 0:
                holdings.pop(instrument_id, None)
            else:
                holdings[instrument_id] = remaining
        return cash

    def _mark_to_market(
        self, cash: float, holdings: dict[str, int], as_of: dt.date
    ) -> float:
        equity = cash
        for instrument_id, quantity in holdings.items():
            equity += quantity * self._last_close(instrument_id, as_of)
        return equity

    def dividend_cash_credit(
        self, holdings: dict[str, int], execution_date: dt.date
    ) -> float:
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
        final_positions = [
            _scale_position(position, scale) for position in final_positions
        ]
        gross = 1.0

    cash_weight = round(1.0 - gross, 12)
    return TargetPortfolio(
        as_of=proposed.as_of,
        positions=tuple(final_positions),
        cash_weight=cash_weight,
        regime=proposed.regime,
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
