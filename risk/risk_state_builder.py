"""Builds a ``risk.portfolio_risk_state.PortfolioRiskState`` for a live
proposed portfolio (Phase 19, feeding step 12: "run risk engine").

``backtest/engine.py`` already builds the equivalent state privately, inside
``BacktestEngine._build_risk_state``/``_market_stats``/``_max_pairwise_correlation``,
for its own historical replay. That code is deliberately left untouched
here (refactoring an already-completed, already-tested phase's internals
is out of this phase's scope) -- this module reimplements the same
approach for live use instead, reusing the one piece that was already a
public, module-level function: ``backtest.engine.market_liquidity_stats``
(``docs/ARCHITECTURE.md``'s dependency rules already document
``broker/``/``execution/`` importing from ``backtest/`` on exactly this
kind of shared-pricing-model precedent).

The one thing a live system has that a backtest does not need to
reconstruct: a stream of *future* equity points to look back over. Instead
of re-deriving that from scratch, ``EquityHistory`` is a small, explicit,
append-only curve the orchestrator maintains across a run (and can seed
from a persisted snapshot at startup).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from backtest.engine import market_liquidity_stats
from data.errors import DataNotAvailableError
from data.interfaces import MarketDataProvider
from data.models import PriceBasis
from portfolio.portfolio_constructor import TargetPortfolio
from risk.portfolio_risk_state import PortfolioRiskState, PositionRisk


@dataclass
class EquityHistory:
    """A trailing, append-only equity curve. Every percentage in
    ``PortfolioRiskState`` is a *positive fraction of loss* (``0.0`` if
    flat or up), matching that class's own documented convention.
    """

    values: list[float] = field(default_factory=list)

    def append(self, equity: float) -> None:
        self.values.append(equity)

    def daily_pnl_pct(self) -> float:
        if len(self.values) < 2:
            return 0.0
        previous = self.values[-2]
        if previous <= 0:
            return 0.0
        return max(0.0, 1.0 - self.values[-1] / previous)

    def rolling_pnl_pct(self, window_days: int) -> float:
        if not self.values:
            return 0.0
        window = self.values[-window_days:]
        peak = max(window)
        if peak <= 0:
            return 0.0
        return max(0.0, 1.0 - self.values[-1] / peak)

    def peak_to_trough_drawdown_pct(self) -> float:
        if not self.values:
            return 0.0
        peak = max(self.values)
        if peak <= 0:
            return 0.0
        return max(0.0, 1.0 - self.values[-1] / peak)


def max_pairwise_correlation(
    market_data: MarketDataProvider,
    instrument_ids: list[str],
    as_of: dt.date,
    *,
    lookback_days: int = 60,
    min_observations: int = 20,
) -> tuple[float, tuple[str, str]] | None:
    """The single most-correlated pair among ``instrument_ids``, from
    trailing adjusted-close returns -- ``None`` if there are fewer than two
    instruments or not enough overlapping history to compute one reliably.
    Mirrors ``BacktestEngine._max_pairwise_correlation``'s approach (same
    lookback shape, same minimum-observations guard) without importing its
    private method.
    """
    if len(instrument_ids) < 2:
        return None
    start = as_of - dt.timedelta(days=lookback_days * 3)
    returns: dict[str, pd.Series] = {}
    for instrument_id in instrument_ids:
        try:
            bars = market_data.get_equity_bars(
                instrument_id, start, as_of, price_basis=PriceBasis.ADJUSTED
            )
        except DataNotAvailableError:
            continue
        if len(bars) < 2:
            continue
        closes = pd.Series(
            [float(bar.close) for bar in bars], index=[bar.session_date for bar in bars]
        ).tail(lookback_days + 1)
        series = closes.pct_change().dropna()
        if not series.empty:
            returns[instrument_id] = series
    if len(returns) < 2:
        return None
    frame = pd.DataFrame(returns).dropna(how="any")
    if len(frame) < min_observations:
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


def build_risk_state(
    proposed: TargetPortfolio,
    equity: float,
    as_of: dt.datetime,
    market_data: MarketDataProvider,
    equity_history: EquityHistory,
    *,
    rolling_drawdown_window_days: int = 5,
    liquidity_lookback_days: int = 20,
    correlation_lookback_days: int = 60,
    min_correlation_observations: int = 20,
    assumed_spread_bps: float = 10.0,
    daily_turnover_pct_so_far: float = 0.0,
    system_healthy: bool = True,
    system_detail: str | None = None,
    broker_connected: bool = True,
    broker_detail: str | None = None,
) -> PortfolioRiskState:
    position_risks = []
    for position in proposed.positions:
        avg_daily_value, _volatility = market_liquidity_stats(
            market_data, position.instrument_id, as_of.date(), liquidity_lookback_days
        )
        position_risks.append(
            PositionRisk(
                instrument_id=position.instrument_id,
                sector=position.sector,
                avg_daily_value_inr=avg_daily_value,
                quote_age_seconds=0.0,
                spread_bps=assumed_spread_bps,
            )
        )

    correlation = max_pairwise_correlation(
        market_data,
        [position.instrument_id for position in proposed.positions],
        as_of.date(),
        lookback_days=correlation_lookback_days,
        min_observations=min_correlation_observations,
    )

    return PortfolioRiskState(
        as_of=as_of,
        equity=equity,
        positions=tuple(position_risks),
        daily_pnl_pct=equity_history.daily_pnl_pct(),
        rolling_pnl_pct=equity_history.rolling_pnl_pct(rolling_drawdown_window_days),
        peak_to_trough_drawdown_pct=equity_history.peak_to_trough_drawdown_pct(),
        daily_turnover_pct_so_far=daily_turnover_pct_so_far,
        max_pairwise_correlation=correlation[0] if correlation else None,
        correlated_pair=correlation[1] if correlation else None,
        system_healthy=system_healthy,
        system_detail=system_detail,
        broker_connected=broker_connected,
        broker_detail=broker_detail,
    )
