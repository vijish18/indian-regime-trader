"""Operational dashboard views: regime-transition audit trail, exposure by
regime, execution quality, cost attribution, and drawdown progression. See
docs/SPECIFICATION.md section 16 and section 22
("Questions the System Must Be Able to Answer").

Not implemented yet (Phase 12).
"""

from __future__ import annotations

import pandas as pd


class Dashboard:
    def regime_timeline(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """State-probability time series alongside price and exposure, for
        debugging regime-driven decisions.
        """
        raise NotImplementedError("Phase 12: dashboard is not implemented yet.")

    def cost_attribution(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Live cost-as-%-of-gross-P&L trend, to catch cost-model drift
        versus the backtest assumptions in backtest/costs.py.
        """
        raise NotImplementedError("Phase 12: dashboard is not implemented yet.")

    def execution_quality(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Expected vs. realized execution price per order."""
        raise NotImplementedError("Phase 12: dashboard is not implemented yet.")
