"""V1 baseline stock selector: liquidity filter, trend filter, and
risk-adjusted momentum score. See docs/SPECIFICATION.md section 7.1.

Deliberately simple so the incremental contribution of the regime layer can
be measured against a stable selection baseline. Must not take the current
regime label as an input -- see universe/__init__.py.

Not implemented yet (Phase 6).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from config.models import SelectionConfig
from universe.universe import UniverseSnapshot


@dataclass(frozen=True)
class CandidateScore:
    instrument_id: str
    score: float
    passed_liquidity_filter: bool
    passed_trend_filter: bool


class StockSelector:
    """Ranks the point-in-time universe using liquidity, trend, and
    risk-adjusted momentum filters.
    """

    def __init__(self, config: SelectionConfig) -> None:
        self.config = config

    def liquidity_filter(self, universe: UniverseSnapshot, as_of: dt.date) -> set[str]:
        """Instrument IDs meeting the minimum rolling traded-value threshold."""
        raise NotImplementedError("Phase 6: stock selection is not implemented yet.")

    def trend_filter(self, instrument_id: str, prices: pd.Series, as_of: dt.date) -> bool:
        """True if price is above the medium/long-term moving average, or
        the trend score is positive.
        """
        raise NotImplementedError("Phase 6: stock selection is not implemented yet.")

    def momentum_score(self, instrument_id: str, prices: pd.Series, as_of: dt.date) -> float:
        """Risk-adjusted momentum over the configured lookback windows
        (``selection.momentum_lookback_months``), excluding the most recent
        short window.
        """
        raise NotImplementedError("Phase 6: stock selection is not implemented yet.")

    def rank_candidates(self, universe: UniverseSnapshot, as_of: dt.date) -> list[CandidateScore]:
        """Return candidates passing all filters, ranked by score, capped to
        ``selection.max_holdings``.
        """
        raise NotImplementedError("Phase 6: stock selection is not implemented yet.")
