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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from backtest.costs import cost_pct_of_turnover as _cost_pct_of_turnover

_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    cagr: float
    total_return: float
    max_drawdown: float
    drawdown_duration_days: int
    volatility: float
    downside_deviation: float
    sharpe: float
    sortino: float
    calmar: float
    turnover: float
    trade_count: int
    win_rate: float
    profit_factor: float
    gross_pnl: float
    total_costs: float
    net_pnl: float
    cost_pct_of_turnover: float
    """``total_costs`` as a fraction of total traded notional --
    ``backtest.costs.cost_pct_of_turnover``, reused directly rather than
    reimplemented, so there is exactly one definition of this ratio."""


class PerformanceCalculator:
    def compute(self, equity_curve: pd.Series, trade_log: pd.DataFrame) -> PerformanceReport:
        if equity_curve.empty:
            raise ValueError("cannot compute performance from an empty equity curve")
        if equity_curve.isna().any():
            raise ValueError("equity_curve contains NaN")
        if (equity_curve <= 0).any():
            raise ValueError("equity_curve contains a non-positive value")

        equity = equity_curve.astype(float)
        daily_returns = equity.pct_change().dropna()

        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        n_sessions = len(equity)
        cagr = _cagr(equity.iloc[0], equity.iloc[-1], n_sessions)
        max_drawdown, drawdown_duration_days = _drawdown_stats(equity)
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

        return PerformanceReport(
            cagr=cagr,
            total_return=total_return,
            max_drawdown=max_drawdown,
            drawdown_duration_days=drawdown_duration_days,
            volatility=volatility,
            downside_deviation=downside_deviation,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            turnover=turnover,
            trade_count=trade_count,
            win_rate=win_rate,
            profit_factor=profit_factor,
            gross_pnl=gross_pnl,
            total_costs=total_costs,
            net_pnl=net_pnl,
            cost_pct_of_turnover=cost_ratio,
        )

    def by_regime(self, equity_curve: pd.Series, regime_history: pd.Series) -> pd.DataFrame:
        """Returns and volatility broken out by the regime active on each
        session (docs/SPECIFICATION.md section 10.2, "time in state, returns
        by state...").

        ``regime_history`` must share (a superset of) ``equity_curve``'s
        index; each session's return is attributed to whichever regime was
        reported for that session.
        """
        if equity_curve.empty:
            raise ValueError("cannot compute regime-conditional performance from an empty curve")

        daily_returns = equity_curve.astype(float).pct_change().dropna()
        regimes = regime_history.reindex(daily_returns.index)
        if regimes.isna().any():
            missing = regimes[regimes.isna()].index.tolist()
            raise ValueError(f"regime_history has no entry for session(s): {missing}")

        frame = pd.DataFrame({"return": daily_returns, "regime": regimes})
        grouped = frame.groupby("regime")

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
        )
        report["share_of_sessions"] = report["sessions"] / len(frame)
        return report


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
