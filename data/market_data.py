"""Read access to market data: NIFTY 50 and India VIX daily bars, per-symbol
equity OHLCV, and live quotes. See docs/SPECIFICATION.md section 4.

Not implemented yet (Phase 3).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Quote:
    instrument_id: str
    bid: float
    ask: float
    last_price: float
    as_of: dt.datetime


class MarketDataClient:
    """Read access to raw and normalized market data. Raw vendor data is
    never overwritten; normalized data is derived and versioned
    (docs/SPECIFICATION.md section 4.1).
    """

    def get_nifty50_bars(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Daily OHLCV for the NIFTY 50 index over ``[start, end]``."""
        raise NotImplementedError("Phase 3: market data access is not implemented yet.")

    def get_india_vix_bars(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Daily OHLCV for India VIX over ``[start, end]``."""
        raise NotImplementedError("Phase 3: market data access is not implemented yet.")

    def get_equity_bars(self, instrument_id: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Daily OHLCV for one equity instrument over ``[start, end]``."""
        raise NotImplementedError("Phase 3: market data access is not implemented yet.")

    def get_quote(self, instrument_id: str) -> Quote:
        """Latest live quote for one instrument, for execution-time spread
        and staleness checks.
        """
        raise NotImplementedError("Phase 3: market data access is not implemented yet.")

    def is_stale(self, instrument_id: str, max_age_minutes: int) -> bool:
        """True if the latest available quote/bar for ``instrument_id`` is
        older than ``max_age_minutes``.
        """
        raise NotImplementedError("Phase 3: market data access is not implemented yet.")
