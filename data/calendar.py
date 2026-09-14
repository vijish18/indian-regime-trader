"""NSE trading calendar: source of truth for whether a given date is a
trading session, and what kind of session it is. Never assume every weekday
is a trading day, and never open/close positions based on wall-clock time
alone (docs/SPECIFICATION.md section 14).

Not implemented yet (Phase 2).
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum


class SessionType(StrEnum):
    CLOSED = "closed"
    PRE_OPEN = "pre_open"
    REGULAR = "regular"
    SPECIAL = "special"  # e.g. Muhurat trading


class TradingCalendar:
    """Answers session-related questions for NSE equities.

    All strategy-facing timestamps are Asia/Kolkata; all persisted timestamps
    are UTC (docs/SPECIFICATION.md section 14).
    """

    def __init__(
        self,
        holiday_dates: frozenset[dt.date],
        special_sessions: dict[dt.date, SessionType],
    ) -> None:
        self.holiday_dates = holiday_dates
        self.special_sessions = special_sessions

    def is_trading_day(self, date: dt.date) -> bool:
        """True if NSE equities are open for regular or special trading on
        ``date``.
        """
        raise NotImplementedError("Phase 2: trading calendar is not implemented yet.")

    def session_type(self, at: dt.datetime) -> SessionType:
        """The session type in force at a specific Asia/Kolkata timestamp."""
        raise NotImplementedError("Phase 2: trading calendar is not implemented yet.")

    def next_trading_day(self, date: dt.date) -> dt.date:
        """The next date on which NSE equities trade, strictly after ``date``."""
        raise NotImplementedError("Phase 2: trading calendar is not implemented yet.")

    def previous_trading_day(self, date: dt.date) -> dt.date:
        """The most recent date before ``date`` on which NSE equities traded."""
        raise NotImplementedError("Phase 2: trading calendar is not implemented yet.")

    def trading_days_between(self, start: dt.date, end: dt.date) -> list[dt.date]:
        """All trading days in ``[start, end]``, inclusive."""
        raise NotImplementedError("Phase 2: trading calendar is not implemented yet.")
