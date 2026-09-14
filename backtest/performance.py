"""Performance metrics computed on a backtest equity curve. See
docs/SPECIFICATION.md section 10.2.

The equity curve must be total-return (price appreciation + dividend
income), not price-return-only, to be comparable against a total-return
NIFTY 50 benchmark (docs/ARCHITECTURE.md).

Not implemented yet (Phase 8/9).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
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


class PerformanceCalculator:
    def compute(self, equity_curve: pd.Series, trade_log: pd.DataFrame) -> PerformanceReport:
        raise NotImplementedError("Phase 8/9: performance metrics are not implemented yet.")

    def by_regime(self, equity_curve: pd.Series, regime_history: pd.Series) -> pd.DataFrame:
        """Returns, drawdown, and turnover broken out by regime label
        (docs/SPECIFICATION.md section 10.2, "time in state, returns by
        state...").
        """
        raise NotImplementedError("Phase 8/9: regime-conditional metrics are not implemented yet.")
