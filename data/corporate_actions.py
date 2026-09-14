"""Corporate action records and price-adjustment logic: splits, bonuses,
rights, mergers/demergers, dividends. See docs/SPECIFICATION.md section 4
and section 7 (corporate-action problems).

Adjustment must only ever use corporate actions with an effective/ex-date
<= the date being adjusted -- back-adjusting historical prices using an
action announced or confirmed later is a look-ahead leak.

Not implemented yet (Phase 3).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class CorporateActionType(StrEnum):
    SPLIT = "split"
    BONUS = "bonus"
    DIVIDEND = "dividend"
    RIGHTS = "rights"
    MERGER = "merger"
    DEMERGER = "demerger"
    DELISTING = "delisting"


@dataclass(frozen=True)
class CorporateAction:
    instrument_id: str
    action_type: CorporateActionType
    ex_date: dt.date
    ratio_or_amount: float
    adjustment_version: int
    successor_instrument_id: str | None = None  # for merger/demerger identity changes


class CorporateActionsProcessor:
    """Applies point-in-time corporate-action adjustments to price series
    and open positions.
    """

    def adjustment_factor(self, instrument_id: str, as_of: dt.date) -> float:
        """Cumulative price-adjustment factor for ``instrument_id`` as of
        ``as_of``, using only actions with ``ex_date <= as_of``.
        """
        raise NotImplementedError("Phase 3: corporate action adjustment is not implemented yet.")

    def adjust_series(self, raw_ohlcv: pd.DataFrame, instrument_id: str) -> pd.DataFrame:
        """Return a back-adjusted OHLCV series. The raw series in
        ``data_cache/raw`` is never overwritten (docs/SPECIFICATION.md
        section 4.1).
        """
        raise NotImplementedError("Phase 3: corporate action adjustment is not implemented yet.")

    def pending_identity_change(self, instrument_id: str, as_of: dt.date) -> CorporateAction | None:
        """A merger/demerger/delisting event affecting ``instrument_id`` on
        or after ``as_of``, if any -- used by execution/position_tracker.py
        to force-handle a held position whose instrument identity changes.
        """
        raise NotImplementedError("Phase 3: corporate action adjustment is not implemented yet.")
