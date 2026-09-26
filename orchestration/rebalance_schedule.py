"""Which sessions the bot trades on (``bot.rebalance`` in settings.yaml).

``daily`` trades every session. ``weekly`` trades on the first session of
each ISO week: Monday, or Tuesday when Monday is a holiday, and so on. The
walk-forward backtest rebalanced every fifth session of a fold; the first
session of the week is the same cadence in a form a person can plan a
login around.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol


class _Calendar(Protocol):
    def is_trading_day(self, day: dt.date) -> bool: ...

    def previous_trading_day(self, day: dt.date) -> dt.date: ...

    def next_trading_day(self, day: dt.date) -> dt.date: ...


def is_rebalance_session(mode: str, calendar: _Calendar, day: dt.date) -> bool:
    if mode not in ("daily", "weekly"):
        raise ValueError(f"unknown rebalance mode {mode!r}")
    if not calendar.is_trading_day(day):
        return False
    if mode == "daily":
        return True
    previous = calendar.previous_trading_day(day)
    return previous.isocalendar()[:2] != day.isocalendar()[:2]


def next_rebalance_session(mode: str, calendar: _Calendar, after: dt.date) -> dt.date:
    """The first rebalance session strictly after ``after``."""
    day = calendar.next_trading_day(after)
    for _ in range(20):
        if is_rebalance_session(mode, calendar, day):
            return day
        day = calendar.next_trading_day(day)
    raise ValueError(f"no rebalance session within 20 sessions of {after}")
