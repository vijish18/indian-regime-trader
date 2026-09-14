"""Point-in-time instrument reference data: trading symbol, exchange,
segment, token, ISIN, tick size, lot size, price bands, tradability status.
See docs/SPECIFICATION.md section 4 and section 17 (``instruments`` table).

Not implemented yet (Phase 2).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    """One point-in-time instrument record, valid for
    ``[effective_from, effective_to)``.
    """

    instrument_id: str
    symbol: str
    isin: str
    exchange: str
    segment: str
    tick_size: float
    lot_size: int
    price_band_pct: float | None
    tradable: bool
    effective_from: dt.date
    effective_to: dt.date | None


class InstrumentMaster:
    """Read access to point-in-time instrument reference data.

    Callers must always pass an ``as_of`` date explicitly rather than
    implicitly assuming "current" data, so that both backtests and live
    trading go through the same point-in-time lookup path.
    """

    def get(self, instrument_id: str, as_of: dt.date) -> Instrument:
        """The instrument record valid on ``as_of``."""
        raise NotImplementedError("Phase 2: instrument master is not implemented yet.")

    def get_by_symbol(self, symbol: str, exchange: str, as_of: dt.date) -> Instrument:
        """The instrument record valid on ``as_of``, looked up by symbol."""
        raise NotImplementedError("Phase 2: instrument master is not implemented yet.")

    def is_fresh(self, max_age_days: int, reference_date: dt.date) -> bool:
        """True if the instrument master snapshot is not older than
        ``max_age_days`` relative to ``reference_date`` (used by the startup
        freshness check, docs/SPECIFICATION.md section 15.1 step 6).
        """
        raise NotImplementedError("Phase 2: instrument master is not implemented yet.")
