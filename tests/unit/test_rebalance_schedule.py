from __future__ import annotations

import datetime as dt

import pytest

from orchestration.rebalance_schedule import is_rebalance_session, next_rebalance_session


class _Calendar:
    """Weekdays trade, except the listed holidays."""

    def __init__(self, holidays: set[dt.date]) -> None:
        self.holidays = holidays

    def is_trading_day(self, day: dt.date) -> bool:
        return day.weekday() < 5 and day not in self.holidays

    def previous_trading_day(self, day: dt.date) -> dt.date:
        day -= dt.timedelta(days=1)
        while not self.is_trading_day(day):
            day -= dt.timedelta(days=1)
        return day

    def next_trading_day(self, day: dt.date) -> dt.date:
        day += dt.timedelta(days=1)
        while not self.is_trading_day(day):
            day += dt.timedelta(days=1)
        return day


MON = dt.date(2026, 9, 28)


def test_weekly_trades_on_monday_only() -> None:
    cal = _Calendar(set())
    week = [MON + dt.timedelta(days=i) for i in range(5)]
    assert [is_rebalance_session("weekly", cal, d) for d in week] == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_weekly_moves_to_tuesday_when_monday_is_a_holiday() -> None:
    cal = _Calendar({MON})
    assert not is_rebalance_session("weekly", cal, MON)
    assert is_rebalance_session("weekly", cal, MON + dt.timedelta(days=1))
    assert not is_rebalance_session("weekly", cal, MON + dt.timedelta(days=2))


def test_weekends_and_holidays_never_rebalance() -> None:
    cal = _Calendar({MON})
    for mode in ("daily", "weekly"):
        assert not is_rebalance_session(mode, cal, MON)
        assert not is_rebalance_session(mode, cal, MON - dt.timedelta(days=1))


def test_daily_trades_every_session() -> None:
    cal = _Calendar(set())
    assert all(is_rebalance_session("daily", cal, MON + dt.timedelta(days=i)) for i in range(5))


def test_next_rebalance_session_skips_to_the_following_week() -> None:
    cal = _Calendar({MON + dt.timedelta(days=7)})
    assert next_rebalance_session("weekly", cal, MON) == MON + dt.timedelta(days=8)
    assert next_rebalance_session("weekly", cal, MON - dt.timedelta(days=3)) == MON


def test_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError):
        is_rebalance_session("monthly", _Calendar(set()), MON)
