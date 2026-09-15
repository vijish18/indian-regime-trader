"""Performance metrics computed on a backtest equity curve. See
docs/SPECIFICATION.md section 10.2.

The equity curve must be total-return (price appreciation + dividend
income), not price-return-only, to be comparable against a total-return
NIFTY 50 benchmark (docs/ARCHITECTURE.md) -- ``backtest/engine.py`` credits
dividends to cash as they go ex, so the curve it produces already satisfies
this.

## Definitions this module commits to

``gross_pnl``/``net_pnl``/``total_costs`` are computed at the *portfolio*
level, not by matching individual buy/sell lots: ``net_pnl`` is simply
``equity_curve.iloc[-1] - equity_curve.iloc[0]`` (the curve is already net
of every cost, since ``backtest/engine.py`` deducts costs from cash as they
occur), ``total_costs`` sums ``trade_log``'s own cost column, and
``gross_pnl = net_pnl + total_costs`` -- the exact inverse of
``backtest.costs.net_pnl``. ``cost_pct_of_turnover`` reuses
``backtest.costs.cost_pct_of_turnover`` directly rather than
reimplementing the ratio a second time. This sidesteps FIFO lot-matching
entirely, which correctly gives a portfolio-level gross/net/cost breakdown
but is *not* a per-trade P&L attribution.

``win_rate`` and ``profit_factor`` are consequently defined at the *daily*
level (fraction of days with a positive equity return; sum of positive
daily P&L over the absolute sum of negative daily P&L) rather than the
per-trade level that would require the FIFO lot-matching this module
deliberately avoids. This is a legitimate, commonly used alternative
definition, not an approximation of the per-trade one -- documented here so
a reader compares like with like.

``turnover`` is total traded notional (``trade_log``'s gross value column,
summed) as a multiple of starting equity -- not annualized, since a
walk-forward fold's own test-window length already determines the
comparison horizon.

## Phase 12 additions

``recovery_duration_days`` is a different question from
``drawdown_duration_days``: the latter is the *longest single streak* of
sessions spent below the running peak anywhere in the series;
``recovery_duration_days`` is specifically how long it took to climb back
out of the *worst* drawdown's own trough to its prior peak -- ``None`` if
the series ends still underwater from that trough.

``average_holding_period_days`` and ``pct_invested``/``pct_cash`` are
*not* computable from ``equity_curve``/``trade_log`` alone.
``pct_invested``/``pct_cash`` need ``cash_history`` (an optional third
argument to :meth:`PerformanceCalculator.compute`, sourced from
``backtest.engine.BacktestResult.cash_history``) -- the equity curve alone
conflates "cash" and "invested" into one number.
``average_holding_period_days`` is reconstructed from ``trade_log`` by
simulating a running per-instrument quantity across fills in
chronological order and timing each *closed* round trip (quantity
0 -> nonzero -> 0); a position still open at the end of the window is not
counted, the standard convention for this metric. Both degrade to
``float("nan")`` when the inputs needed to compute them are not supplied
or no round trip ever closed -- "not computable from what was given", not
a fabricated zero.

``gross_return``/``net_return`` are the percentage counterparts of
``gross_pnl``/``net_pnl`` (divided by starting equity); ``net_return`` is
definitionally identical to ``total_return``, included as its own field
only so a reader who came looking for "net return" by that name finds it
without needing to know it is called something else here.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import cast

import pandas as pd

from backtest.costs import cost_pct_of_turnover as _cost_pct_of_turnover

_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    cagr: float
    total_return: float
    max_drawdown: float
    drawdown_duration_days: int
    recovery_duration_days: int | None
    """Sessions from the worst drawdown's trough back to its prior peak;
    ``None`` if the series ends still underwater from that trough."""

    volatility: float
    downside_deviation: float
    sharpe: float
    sortino: float
    calmar: float
    turnover: float
    average_holding_period_days: float
    """Mean length of a *closed* round trip per instrument (open-to-flat).
    ``float("nan")`` if no round trip closed within the window."""

    pct_invested: float
    """Mean fraction of equity held in positions (not cash) across the
    window. ``float("nan")`` if ``cash_history`` was not supplied."""

    pct_cash: float
    """``1 - pct_invested``, reported directly rather than making a caller
    subtract. ``float("nan")`` under the same condition as ``pct_invested``."""

    trade_count: int
    win_rate: float
    profit_factor: float
    gross_pnl: float
    total_costs: float
    net_pnl: float
    gross_return: float
    """``gross_pnl`` as a fraction of starting equity."""

    net_return: float
    """Identical to ``total_return`` -- see this module's docstring."""

    cost_pct_of_turnover: float
    """``total_costs`` as a fraction of total traded notional --
    ``backtest.costs.cost_pct_of_turnover``, reused directly rather than
    reimplemented, so there is exactly one definition of this ratio."""


class PerformanceCalculator:
    def compute(
        self,
        equity_curve: pd.Series,
        trade_log: pd.DataFrame,
        cash_history: pd.Series | None = None,
    ) -> PerformanceReport:
        """``cash_history`` is optional (indexed identically to
        ``equity_curve``, e.g. ``backtest.engine.BacktestResult.cash_history``);
        without it, ``pct_invested``/``pct_cash`` report ``float("nan")``
        rather than a fabricated number.
        """
        if equity_curve.empty:
            raise ValueError("cannot compute performance from an empty equity curve")
        if equity_curve.isna().any():
            raise ValueError("equity_curve contains NaN")
        if (equity_curve <= 0).any():
            raise ValueError("equity_curve contains a non-positive value")
        if cash_history is not None and not cash_history.index.equals(equity_curve.index):
            raise ValueError("cash_history must share equity_curve's index exactly")

        equity = equity_curve.astype(float)
        daily_returns = equity.pct_change().dropna()

        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        n_sessions = len(equity)
        cagr = _cagr(equity.iloc[0], equity.iloc[-1], n_sessions)
        max_drawdown, drawdown_duration_days = _drawdown_stats(equity)
        recovery_duration_days = _recovery_duration(equity)
        volatility = _annualized_std(daily_returns)
        downside_deviation = _annualized_downside_deviation(daily_returns)
        sharpe = _ratio(daily_returns.mean(), daily_returns.std(ddof=0))
        downside_deviation_daily = (
            downside_deviation / math.sqrt(_TRADING_DAYS_PER_YEAR)
            if downside_deviation > 0
            else 0.0
        )
        sortino = _ratio(daily_returns.mean(), downside_deviation_daily)
        calmar = _calmar(cagr, max_drawdown)

        net_pnl = float(equity.iloc[-1] - equity.iloc[0])
        has_cost_column = "cost" in trade_log.columns and not trade_log.empty
        total_costs = float(trade_log["cost"].sum()) if has_cost_column else 0.0
        gross_pnl = net_pnl + total_costs
        has_turnover_column = "gross_value" in trade_log.columns and not trade_log.empty
        total_turnover_value = (
            float(trade_log["gross_value"].sum()) if has_turnover_column else 0.0
        )
        turnover = total_turnover_value / float(equity.iloc[0])
        cost_ratio = (
            _cost_pct_of_turnover(total_costs, total_turnover_value)
            if total_turnover_value > 0
            else 0.0
        )
        trade_count = int(len(trade_log))
        win_rate = float((daily_returns > 0).mean()) if len(daily_returns) > 0 else 0.0
        profit_factor = _profit_factor(daily_returns)
        average_holding_period_days = _average_holding_period_days(trade_log)
        pct_invested, pct_cash = _pct_invested_and_cash(equity, cash_history)
        gross_return = gross_pnl / float(equity.iloc[0])

        return PerformanceReport(
            cagr=cagr,
            total_return=total_return,
            max_drawdown=max_drawdown,
            drawdown_duration_days=drawdown_duration_days,
            recovery_duration_days=recovery_duration_days,
            volatility=volatility,
            downside_deviation=downside_deviation,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            turnover=turnover,
            average_holding_period_days=average_holding_period_days,
            pct_invested=pct_invested,
            pct_cash=pct_cash,
            trade_count=trade_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            gross_pnl=gross_pnl,
            total_costs=total_costs,
            net_pnl=net_pnl,
            gross_return=gross_return,
            net_return=total_return,
            cost_pct_of_turnover=cost_ratio,
        )

    def by_regime(self, equity_curve: pd.Series, regime_history: pd.Series) -> pd.DataFrame:
        """Time spent in regime, return, volatility, and drawdown broken
        out by the regime active on each session (docs/SPECIFICATION.md
        section 10.2, "time in state, returns by state...").

        ``regime_history`` and ``equity_curve`` are aligned *positionally*,
        not by date label: ``backtest.engine.BacktestResult.regime_history``
        is indexed by *signal* date and ``equity_curve`` by *execution*
        date (one trading day later, by this engine's own next-session
        execution rule), so the two never share date labels to reindex
        against in the first place. ``regime_history`` must have exactly
        one entry per ``equity_curve`` point, in the same chronological
        order; its last entry (the regime read on the final signal date)
        has no following return within this window to attribute and is
        dropped. "Drawdown during regime" treats that regime's own
        sessions (not necessarily contiguous) as one synthetic equity path
        built purely from its own returns, in chronological order -- the
        peak-to-trough decline the portfolio experienced specifically
        while that regime was in force, not the share of the *portfolio's*
        overall drawdown that happened to overlap with it.
        """
        if equity_curve.empty:
            raise ValueError("cannot compute regime-conditional performance from an empty curve")

        daily_returns = equity_curve.astype(float).pct_change().dropna()
        regimes = _align_positionally(regime_history, equity_curve, daily_returns, "regime_history")
        return _grouped_report(daily_returns, regimes)

    def by_confidence(
        self,
        equity_curve: pd.Series,
        confidence_history: pd.Series,
        low_threshold: float = 0.60,
        high_threshold: float = 0.85,
    ) -> pd.DataFrame:
        """The same breakdown as :meth:`by_regime`, but grouped by
        confidence tier (LOW / MEDIUM / HIGH) instead of regime label.

        A session's confidence is bucketed as LOW (< ``low_threshold``),
        HIGH (>= ``high_threshold``), or MEDIUM (between the two) --
        answering "does this strategy only perform well when it is
        confident" (or, just as informatively, whether it does *not*),
        which a single blended Sharpe ratio cannot show on its own.
        ``confidence_history`` is aligned positionally with
        ``equity_curve``, for the same reason documented on
        :meth:`by_regime`.
        """
        if equity_curve.empty:
            raise ValueError(
                "cannot compute confidence-conditional performance from an empty curve"
            )
        if not (0.0 <= low_threshold <= high_threshold <= 1.0):
            raise ValueError(
                f"thresholds must satisfy 0 <= low_threshold <= high_threshold <= 1, "
                f"got {low_threshold}, {high_threshold}"
            )

        daily_returns = equity_curve.astype(float).pct_change().dropna()
        confidence = _align_positionally(
            confidence_history, equity_curve, daily_returns, "confidence_history"
        )
        tiers = confidence.apply(
            lambda value: _confidence_tier(float(value), low_threshold, high_threshold)
        )
        return _grouped_report(daily_returns, tiers)


def _cagr(start_equity: float, end_equity: float, n_sessions: int) -> float:
    if n_sessions <= 1:
        return 0.0
    years = (n_sessions - 1) / _TRADING_DAYS_PER_YEAR
    if years <= 0 or start_equity <= 0:
        return 0.0
    return float((end_equity / start_equity) ** (1.0 / years) - 1.0)


def _drawdown_stats(equity: pd.Series) -> tuple[float, int]:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(-drawdown.min()) if len(drawdown) else 0.0

    longest_duration = 0
    current_duration = 0
    for value in drawdown:
        if value < 0:
            current_duration += 1
            longest_duration = max(longest_duration, current_duration)
        else:
            current_duration = 0
    return max_drawdown, longest_duration


def _recovery_duration(equity: pd.Series) -> int | None:
    """Sessions from the worst drawdown's trough back to its prior peak;
    ``None`` if the series ends still underwater from that trough. ``0``
    if the series never drew down at all (nothing to recover from)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    if len(drawdown) == 0 or drawdown.min() >= 0:
        return 0

    trough_position = int(drawdown.to_numpy().argmin())
    peak_level = running_max.iloc[trough_position]
    for offset, value in enumerate(equity.iloc[trough_position + 1 :], start=1):
        if value >= peak_level:
            return offset
    return None


def _align_positionally(
    signal_indexed: pd.Series, equity_curve: pd.Series, daily_returns: pd.Series, name: str
) -> pd.Series:
    """``signal_indexed`` (one entry per ``equity_curve`` point, same
    chronological order, but not sharing its date labels -- see
    :meth:`PerformanceCalculator.by_regime`) reduced to one entry per
    ``daily_returns`` point by dropping its last entry and adopting
    ``daily_returns``'s own index, so the two can be grouped together.
    """
    if len(signal_indexed) != len(equity_curve):
        raise ValueError(
            f"{name} has {len(signal_indexed)} entries but equity_curve has "
            f"{len(equity_curve)}; they must be aligned positionally, one {name} entry "
            "per equity_curve point in the same chronological order"
        )
    return pd.Series(signal_indexed.to_numpy()[:-1], index=daily_returns.index)


def _grouped_report(daily_returns: pd.Series, group_labels: pd.Series) -> pd.DataFrame:
    """Sessions, mean return, volatility, cumulative return, and
    "drawdown during group" for each distinct value in ``group_labels`` --
    shared by ``by_regime`` and ``by_confidence``, which differ only in
    what they group by.
    """
    frame = pd.DataFrame({"return": daily_returns, "group": group_labels})
    grouped = frame.groupby("group")

    report = grouped.agg(
        sessions=pd.NamedAgg(column="return", aggfunc="count"),
        mean_daily_return=pd.NamedAgg(column="return", aggfunc="mean"),
        volatility=pd.NamedAgg(
            column="return",
            aggfunc=lambda s: float(s.std(ddof=0) * math.sqrt(_TRADING_DAYS_PER_YEAR)),
        ),
        cumulative_return=pd.NamedAgg(
            column="return", aggfunc=lambda s: float((1.0 + s).prod() - 1.0)
        ),
        max_drawdown=pd.NamedAgg(column="return", aggfunc=_group_max_drawdown),
    )
    report["share_of_sessions"] = report["sessions"] / len(frame)
    return report


def _group_max_drawdown(returns: pd.Series) -> float:
    """Peak-to-trough decline of the synthetic equity path built purely
    from this group's own returns, in chronological order.

    Prepends an implicit baseline of ``1.0`` before compounding: without
    it, the group's very first return could not register as a drawdown at
    all (``cummax`` of a series is trivially equal to itself at its own
    first point), silently missing a drop that happened on the group's
    first session.
    """
    synthetic_equity = pd.concat([pd.Series([1.0]), (1.0 + returns).cumprod()])
    running_max = synthetic_equity.cummax()
    drawdown = synthetic_equity / running_max - 1.0
    return float(-drawdown.min()) if len(drawdown) else 0.0


def _confidence_tier(value: float, low_threshold: float, high_threshold: float) -> str:
    if value < low_threshold:
        return "low"
    if value >= high_threshold:
        return "high"
    return "medium"


def _average_holding_period_days(trade_log: pd.DataFrame) -> float:
    """Mean length of a *closed* round trip per instrument, simulated by
    walking ``trade_log`` in execution-date order and tracking each
    instrument's running quantity: a position still open at the end of
    the window is excluded, the standard convention for this metric.
    """
    required_columns = {"instrument_id", "execution_date", "side", "quantity"}
    if trade_log.empty or not required_columns.issubset(trade_log.columns):
        return float("nan")

    ordered = trade_log.sort_values("execution_date")
    open_quantity: dict[str, int] = {}
    open_date: dict[str, dt.date] = {}
    holding_periods_days: list[int] = []

    for _, series_row in ordered.iterrows():
        instrument_id = str(series_row["instrument_id"])
        quantity = int(series_row["quantity"])
        execution_date = cast(dt.date, series_row["execution_date"])
        signed_quantity = quantity if series_row["side"] == "buy" else -quantity
        previous_quantity = open_quantity.get(instrument_id, 0)
        new_quantity = previous_quantity + signed_quantity

        if previous_quantity == 0 and new_quantity != 0:
            open_date[instrument_id] = execution_date
        elif previous_quantity != 0 and new_quantity == 0:
            entry_date = open_date.pop(instrument_id, None)
            if entry_date is not None:
                holding_periods_days.append((execution_date - entry_date).days)

        open_quantity[instrument_id] = new_quantity

    if not holding_periods_days:
        return float("nan")
    return float(sum(holding_periods_days) / len(holding_periods_days))


def _pct_invested_and_cash(
    equity: pd.Series, cash_history: pd.Series | None
) -> tuple[float, float]:
    if cash_history is None:
        return float("nan"), float("nan")
    cash_fraction = cash_history.astype(float) / equity
    pct_cash = float(cash_fraction.mean())
    pct_invested = 1.0 - pct_cash
    return pct_invested, pct_cash


def _annualized_std(daily_returns: pd.Series) -> float:
    if len(daily_returns) == 0:
        return 0.0
    return float(daily_returns.std(ddof=0) * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _annualized_downside_deviation(daily_returns: pd.Series) -> float:
    if len(daily_returns) == 0:
        return 0.0
    downside = daily_returns.clip(upper=0.0)
    return float(math.sqrt((downside**2).mean()) * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _ratio(mean_daily: float, denominator_daily: float) -> float:
    if denominator_daily <= 0 or not math.isfinite(denominator_daily):
        return 0.0
    return float(mean_daily / denominator_daily * math.sqrt(_TRADING_DAYS_PER_YEAR))


def _calmar(cagr: float, max_drawdown: float) -> float:
    if max_drawdown == 0:
        return math.inf if cagr > 0 else 0.0
    return float(cagr / max_drawdown)


def _profit_factor(daily_returns: pd.Series) -> float:
    gains = daily_returns[daily_returns > 0].sum()
    losses = -daily_returns[daily_returns < 0].sum()
    if losses == 0:
        return math.inf if gains > 0 else 0.0
    return float(gains / losses)
