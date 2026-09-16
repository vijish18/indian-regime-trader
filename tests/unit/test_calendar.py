"""Calendar behavior: holidays, weekends, special sessions, and the
fail-closed coverage rule.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from data.calendar import NSETradingCalendar
from data.errors import CalendarCoverageError
from data.models import SessionType

IST = ZoneInfo("Asia/Kolkata")


def test_weekday_is_a_trading_day(calendar: NSETradingCalendar) -> None:
    assert calendar.is_trading_day(dt.date(2024, 1, 2))  # Tuesday


@pytest.mark.parametrize("day", [dt.date(2024, 1, 6), dt.date(2024, 1, 7)])
def test_weekend_is_not_a_trading_day(calendar: NSETradingCalendar, day: dt.date) -> None:
    assert not calendar.is_trading_day(day)


def test_holiday_is_not_a_trading_day(calendar: NSETradingCalendar) -> None:
    assert not calendar.is_trading_day(dt.date(2024, 1, 26))  # Republic Day, a Friday


def test_special_session_on_a_sunday_is_a_trading_day(
    calendar: NSETradingCalendar,
) -> None:
    """Muhurat trading falls on Diwali, which can be a weekend -- so trading
    days cannot be derived from the weekday plus a holiday list.
    """
    muhurat = dt.date(2024, 11, 3)
    assert muhurat.weekday() == 6  # Sunday
    assert calendar.is_trading_day(muhurat)
    assert calendar.session(muhurat).session_type is SessionType.SPECIAL


def test_session_boundaries_are_timezone_aware(calendar: NSETradingCalendar) -> None:
    session = calendar.session(dt.date(2024, 1, 2))
    assert session.regular_open.tzinfo is not None
    assert session.regular_open == dt.datetime(2024, 1, 2, 9, 15, tzinfo=IST)
    assert session.regular_close == dt.datetime(2024, 1, 2, 15, 30, tzinfo=IST)
    assert session.pre_open_start == dt.datetime(2024, 1, 2, 9, 0, tzinfo=IST)


def test_closed_day_yields_a_closed_session(calendar: NSETradingCalendar) -> None:
    session = calendar.session(dt.date(2024, 1, 26))
    assert session.session_type is SessionType.CLOSED
    assert not session.is_trading_day


def test_trading_days_between_skips_weekends_and_holidays(
    calendar: NSETradingCalendar,
) -> None:
    days = calendar.trading_days_between(dt.date(2024, 1, 25), dt.date(2024, 1, 30))
    assert days == [
        dt.date(2024, 1, 25),  # Thu
        # 26th is Republic Day, 27-28 is the weekend
        dt.date(2024, 1, 29),  # Mon
        dt.date(2024, 1, 30),  # Tue
    ]


def test_trading_days_between_rejects_inverted_range(
    calendar: NSETradingCalendar,
) -> None:
    with pytest.raises(ValueError, match="precedes"):
        calendar.trading_days_between(dt.date(2024, 3, 1), dt.date(2024, 2, 1))


def test_next_and_previous_trading_day_skip_closures(
    calendar: NSETradingCalendar,
) -> None:
    assert calendar.next_trading_day(dt.date(2024, 1, 25)) == dt.date(2024, 1, 29)
    assert calendar.previous_trading_day(dt.date(2024, 1, 29)) == dt.date(2024, 1, 25)


def test_sessions_offset_moves_by_sessions_not_calendar_days(
    calendar: NSETradingCalendar,
) -> None:
    """Next-session execution means the next *session*: from Thursday the 25th,
    t+1 is Monday the 29th, because Friday was a holiday.
    """
    assert calendar.sessions_offset(dt.date(2024, 1, 25), 1) == dt.date(2024, 1, 29)
    assert calendar.sessions_offset(dt.date(2024, 1, 25), 2) == dt.date(2024, 1, 30)
    assert calendar.sessions_offset(dt.date(2024, 1, 30), -2) == dt.date(2024, 1, 25)


def test_sessions_offset_zero_requires_a_trading_day(
    calendar: NSETradingCalendar,
) -> None:
    assert calendar.sessions_offset(dt.date(2024, 1, 25), 0) == dt.date(2024, 1, 25)
    with pytest.raises(ValueError, match="not a trading day"):
        calendar.sessions_offset(dt.date(2024, 1, 26), 0)


def test_calendar_refuses_dates_outside_its_coverage(
    calendar: NSETradingCalendar,
) -> None:
    """Answering for an uncovered year would silently assume "no holidays",
    turning missing data into wrong data.
    """
    assert calendar.covered_years == frozenset({2024})
    with pytest.raises(CalendarCoverageError, match="2025"):
        calendar.is_trading_day(dt.date(2025, 1, 2))
    with pytest.raises(CalendarCoverageError):
        calendar.trading_days_between(dt.date(2024, 12, 20), dt.date(2025, 1, 10))


def test_calendar_loads_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "holidays.csv"
    path.write_text(
        "date,description,session_type\n"
        "2024-01-26,Republic Day,closed\n"
        "2024-11-03,Muhurat Trading,special\n",
        encoding="utf-8",
    )
    loaded = NSETradingCalendar.from_file(path)
    assert not loaded.is_trading_day(dt.date(2024, 1, 26))
    assert loaded.is_trading_day(dt.date(2024, 11, 3))
    assert loaded.covered_years == frozenset({2024})


def test_calendar_file_rejects_duplicate_dates(tmp_path: Path) -> None:
    path = tmp_path / "holidays.csv"
    path.write_text(
        "date,description,session_type\n"
        "2024-01-26,Republic Day,closed\n"
        "2024-01-26,Duplicated Entry,closed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate calendar entry"):
        NSETradingCalendar.from_file(path)


def test_empty_calendar_file_is_rejected_with_guidance(tmp_path: Path) -> None:
    path = tmp_path / "holidays.csv"
    path.write_text("date,description,session_type\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no calendar entries"):
        NSETradingCalendar.from_file(path)


def test_calendar_file_rejects_invalid_dates(tmp_path: Path) -> None:
    path = tmp_path / "holidays.csv"
    path.write_text(
        "date,description,session_type\n2024-02-31,Not A Real Date,closed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ISO-8601"):
        NSETradingCalendar.from_file(path)


def test_shipped_calendar_file_is_populated() -> None:
    """The repository used to ship an unpopulated calendar on purpose --
    fabricating holiday dates would have been worse than refusing to run --
    and the test that documented that state said it should be replaced
    "when the real list is loaded". This is that replacement.

    ``config/nse_holidays.csv`` is now transcribed from NSE's own
    circulars and live holiday master by ``scripts/build_nse_holidays.py``.
    The substantive checks on its *contents* live in
    ``tests/unit/test_nse_holidays.py``; this one only asserts that the
    file the rest of the system loads is no longer empty, so that a
    regression to the placeholder state fails here rather than at 09:15.
    """
    path = Path(__file__).resolve().parents[2] / "config" / "nse_holidays.csv"
    calendar = NSETradingCalendar.from_file(path)
    assert calendar.covered_years
