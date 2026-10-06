"""Which sessions the bot trades on (``bot.rebalance`` in settings.yaml).

``daily`` trades every session. ``weekly`` trades on the first session of
each ISO week: Monday, or Tuesday when Monday is a holiday, and so on. The
walk-forward backtest rebalanced every fifth session of a fold; the first
session of the week is the same cadence in a form a person can plan a
login around. A week whose first session passes without a rebalance catches
up on its next session (``rebalance_due``).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
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


def rebalance_due(
    mode: str, calendar: _Calendar, day: dt.date, completed: Iterable[dt.date]
) -> bool:
    """Whether ``day`` should run a rebalance, given the sessions that already did.

    The scheduled session is the first of the ISO week, but a missed one -- no
    Kite login that morning, a failed start -- is not dropped for the week: the
    next session of the same week runs it instead. Without this one missed
    login leaves the book in cash, or a week stale, until the following week.
    A first build of an empty account is the same case: no rebalance yet this
    week, so the next session with a login builds it.
    """
    if mode not in ("daily", "weekly"):
        raise ValueError(f"unknown rebalance mode {mode!r}")
    if not calendar.is_trading_day(day):
        return False
    if mode == "daily":
        return True
    week = day.isocalendar()[:2]
    return not any(d <= day and d.isocalendar()[:2] == week for d in completed)


def next_rebalance_due(
    mode: str, calendar: _Calendar, after: dt.date, completed: Iterable[dt.date]
) -> dt.date:
    """The first session strictly after ``after`` on which a rebalance is due,
    assuming none completes in between."""
    done = list(completed)
    day = calendar.next_trading_day(after)
    for _ in range(20):
        if rebalance_due(mode, calendar, day, done):
            return day
        day = calendar.next_trading_day(day)
    raise ValueError(f"no rebalance due within 20 sessions of {after}")
