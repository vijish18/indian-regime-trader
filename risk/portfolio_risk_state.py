"""The one snapshot every risk check reads from: measured facts about the
book, the market, and the operating environment -- never a regime label, a
stock score, or anything else from an upstream layer.

``PortfolioRiskState`` is assembled by whatever orchestrates a trading cycle
(the backtest engine, later the live loop), not by ``risk/`` itself: this
module intentionally has no dependency on ``data.interfaces.MarketDataProvider``
or ``broker.base.Broker``. That keeps every risk check a pure function of
plain data, which is what makes ``risk/risk_manager.py`` and
``risk/circuit_breaker.py`` deterministic and property-testable without a
market data double or a broker double -- and keeps ``risk/`` from acquiring
a dependency on ``broker/`` or ``execution/``, which sit downstream of it
(docs/ARCHITECTURE.md, "Dependency rules").
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PositionRisk:
    """Per-instrument facts needed to risk-check one proposed position --
    not the position itself (that's ``portfolio.portfolio_constructor.TargetPosition``),
    just the market/liquidity context a risk check needs to evaluate it.
    """

    instrument_id: str
    sector: str
    avg_daily_value_inr: float
    """Trailing average daily traded value, for the liquidity/ADV-participation check."""

    quote_age_seconds: float
    """How stale the last quote for this instrument is, for the stale-data check."""

    spread_bps: float
    """Current bid/ask spread in basis points, for the abnormal-spread check."""

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must not be empty")
        if self.avg_daily_value_inr < 0:
            raise ValueError(
                f"avg_daily_value_inr must be >= 0, got {self.avg_daily_value_inr}"
            )
        if self.quote_age_seconds < 0:
            raise ValueError(f"quote_age_seconds must be >= 0, got {self.quote_age_seconds}")
        if self.spread_bps < 0:
            raise ValueError(f"spread_bps must be >= 0, got {self.spread_bps}")


@dataclass(frozen=True, slots=True)
class PortfolioRiskState:
    """Everything ``risk/`` needs to know about the world to evaluate a
    proposed trade, other than the proposal itself.
    """

    as_of: dt.datetime
    equity: float
    positions: tuple[PositionRisk, ...]
    """Market/liquidity facts for every instrument that might appear in a
    proposed target portfolio -- a lookup table, not a holding list."""

    daily_pnl_pct: float
    """Today's drawdown so far, as a positive fraction (0.02 = down 2%)."""

    rolling_pnl_pct: float
    """Drawdown over a trailing multi-session window the caller defines
    (e.g. 5 trading days) -- the "rolling drawdown" check, distinct from
    both same-day P&L and since-inception peak-to-trough."""

    peak_to_trough_drawdown_pct: float
    """Drawdown from the strategy's all-time equity peak to today, as a
    positive fraction."""

    daily_turnover_pct_so_far: float
    """Turnover (bought + sold, as a fraction of equity) already executed
    earlier today, before the trade currently being evaluated."""

    max_pairwise_correlation: float | None
    correlated_pair: tuple[str, str] | None
    """The single most-correlated pair currently in (or proposed for) the
    book, if any -- ``None`` for both, or a value for both, never a mix."""

    system_healthy: bool
    system_detail: str | None
    broker_connected: bool
    broker_detail: str | None

    def __post_init__(self) -> None:
        if self.equity <= 0:
            raise ValueError(f"equity must be > 0, got {self.equity}")
        ids = [position.instrument_id for position in self.positions]
        if len(ids) != len(set(ids)):
            raise ValueError("PortfolioRiskState.positions has a duplicate instrument_id")
        for pct_name, pct_value in (
            ("daily_pnl_pct", self.daily_pnl_pct),
            ("rolling_pnl_pct", self.rolling_pnl_pct),
            ("peak_to_trough_drawdown_pct", self.peak_to_trough_drawdown_pct),
            ("daily_turnover_pct_so_far", self.daily_turnover_pct_so_far),
        ):
            if pct_value < 0:
                raise ValueError(f"{pct_name} must be >= 0, got {pct_value}")
        if (self.max_pairwise_correlation is None) != (self.correlated_pair is None):
            raise ValueError(
                "max_pairwise_correlation and correlated_pair must both be set or both be None"
            )
        if self.max_pairwise_correlation is not None and not (
            -1.0 <= self.max_pairwise_correlation <= 1.0
        ):
            raise ValueError(
                f"max_pairwise_correlation must be in [-1, 1], got {self.max_pairwise_correlation}"
            )

    def position_risk(self, instrument_id: str) -> PositionRisk | None:
        for position in self.positions:
            if position.instrument_id == instrument_id:
                return position
        return None
