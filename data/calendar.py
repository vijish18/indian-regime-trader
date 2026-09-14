"""NSE trading calendar.

The calendar is the source of truth for whether the exchange was open. Two
properties matter more than they look:

- **It fails closed outside its coverage.** Asking about a year the holiday
  dataset does not cover raises ``CalendarCoverageError`` rather than assuming
  "no holidays". Assuming would turn missing data into wrong data: a backtest
  would happily trade on Republic Day, and a live loop would expect fills on a
  closed exchange.
- **Special sessions are first-class.** Muhurat trading falls on Diwali, which
  can land on a weekend, so "is it a trading day" cannot be derived from the
  weekday plus a holiday list (docs/SPECIFICATION.md section 14).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from pathlib import Path
from zoneinfo import ZoneInfo

from data.errors import CalendarCoverageError
from data.interfaces import TradingCalendar
from data.models import SessionType, TradingSession
from data.storage import parse_date, parse_optional_str, read_table, require_columns

_MAX_SEARCH_DAYS = 400
"""Upper bound when walking forward/backward for the next trading day. A gap
this long means the calendar data is wrong, not that the exchange was shut."""


class NSETradingCalendar(TradingCalendar):
    """Exchange calendar backed by a maintained holiday dataset.

    Coverage is tracked per calendar year: a year is considered covered if the
    dataset contains at least one entry for it. NSE has holidays in every
    year, so a year with no entries means "not maintained yet", not "no
    holidays" -- and the calendar refuses to answer for it.
    """

    def __init__(
        self,
        holidays: Mapping[dt.date, str],
        special_sessions: Mapping[dt.date, str] | None = None,
        *,
        timezone: str = "Asia/Kolkata",
        pre_open_start: dt.time = dt.time(9, 0),
        pre_open_end: dt.time = dt.time(9, 8),
        regular_open: dt.time = dt.time(9, 15),
        regular_close: dt.time = dt.time(15, 30),
        covered_years: frozenset[int] | None = None,
    ) -> None:
        self._holidays = dict(holidays)
        self._special_sessions = dict(special_sessions or {})
        self._zone = ZoneInfo(timezone)
        self._pre_open_start = pre_open_start
        self._pre_open_end = pre_open_end
        self._regular_open = regular_open
        self._regular_close = regular_close
        self._covered_years = (
            covered_years
            if covered_years is not None
            else frozenset(
                day.year
                for day in list(self._holidays) + list(self._special_sessions)
            )
        )

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        timezone: str = "Asia/Kolkata",
        pre_open_start: dt.time = dt.time(9, 0),
        pre_open_end: dt.time = dt.time(9, 8),
        regular_open: dt.time = dt.time(9, 15),
        regular_close: dt.time = dt.time(15, 30),
    ) -> NSETradingCalendar:
        """Load from a CSV/Parquet holiday dataset.

        Required columns: ``date``, ``description``. Optional ``session_type``
        marks special sessions (``special``); anything else is treated as a
        full closure.
        """
        frame = read_table(path)
        require_columns(frame, ("date", "description"), str(path))

        holidays: dict[dt.date, str] = {}
        special: dict[dt.date, str] = {}
        years: set[int] = set()
        for position, row in enumerate(frame.to_dict("records"), start=2):
            day = parse_date(row["date"])
            description = parse_optional_str(row["description"]) or ""
            session_type = (
                parse_optional_str(row.get("session_type")) or SessionType.CLOSED.value
            ).lower()
            if day in holidays or day in special:
                raise ValueError(
                    f"{path} line {position}: duplicate calendar entry for {day}"
                )
            if session_type == SessionType.SPECIAL.value:
                special[day] = description
            else:
                holidays[day] = description
            years.add(day.year)

        if not years:
            raise ValueError(
                f"{path} contains no calendar entries. Populate it from NSE's "
                "published holiday list (one row per closure, plus session_type="
                "'special' for Muhurat sessions) before running anything that "
                "depends on the calendar. An empty calendar is refused rather "
                "than treated as 'no holidays'."
            )
        return cls(
            holidays=holidays,
            special_sessions=special,
            timezone=timezone,
            pre_open_start=pre_open_start,
            pre_open_end=pre_open_end,
            regular_open=regular_open,
            regular_close=regular_close,
            covered_years=frozenset(years),
        )

    @property
    def covered_years(self) -> frozenset[int]:
        return self._covered_years

    @property
    def timezone(self) -> ZoneInfo:
        return self._zone

    def is_trading_day(self, day: dt.date) -> bool:
        self._require_coverage(day)
        return self._is_trading_day_unchecked(day)

    def session(self, day: dt.date) -> TradingSession:
        self._require_coverage(day)
        if day in self._special_sessions:
            session_type = SessionType.SPECIAL
        elif self._is_trading_day_unchecked(day):
            session_type = SessionType.REGULAR
        else:
            session_type = SessionType.CLOSED
        return TradingSession(
            session_date=day,
            session_type=session_type,
            regular_open=self._at(day, self._regular_open),
            regular_close=self._at(day, self._regular_close),
            pre_open_start=self._at(day, self._pre_open_start),
            pre_open_end=self._at(day, self._pre_open_end),
        )

    def sessions_between(self, start: dt.date, end: dt.date) -> list[TradingSession]:
        return [self.session(day) for day in self.trading_days_between(start, end)]

    def trading_days_between(self, start: dt.date, end: dt.date) -> list[dt.date]:
        if end < start:
            raise ValueError(f"end {end} precedes start {start}")
        self._require_coverage(start)
        self._require_coverage(end)
        days: list[dt.date] = []
        day = start
        while day <= end:
            if self._is_trading_day_unchecked(day):
                days.append(day)
            day += dt.timedelta(days=1)
        return days

    def next_trading_day(self, day: dt.date) -> dt.date:
        return self._step(day, forward=True)

    def previous_trading_day(self, day: dt.date) -> dt.date:
        return self._step(day, forward=False)

    def sessions_offset(self, day: dt.date, sessions: int) -> dt.date:
        """Move ``sessions`` trading sessions from ``day``.

        ``sessions=0`` returns ``day`` itself and requires it to be a trading
        day, so callers cannot silently anchor a window on a closed date.
        """
        self._require_coverage(day)
        if sessions == 0:
            if not self._is_trading_day_unchecked(day):
                raise ValueError(f"{day} is not a trading day; cannot use it as anchor")
            return day
        current = day
        for _ in range(abs(sessions)):
            current = self._step(current, forward=sessions > 0)
        return current

    def _step(self, day: dt.date, *, forward: bool) -> dt.date:
        self._require_coverage(day)
        delta = dt.timedelta(days=1 if forward else -1)
        candidate = day + delta
        for _ in range(_MAX_SEARCH_DAYS):
            self._require_coverage(candidate)
            if self._is_trading_day_unchecked(candidate):
                return candidate
            candidate += delta
        raise CalendarCoverageError(
            f"no trading day found within {_MAX_SEARCH_DAYS} days "
            f"{'after' if forward else 'before'} {day}; calendar data looks wrong"
        )

    def _is_trading_day_unchecked(self, day: dt.date) -> bool:
        if day in self._special_sessions:
            return True
        if day.weekday() >= 5:  # Saturday, Sunday
            return False
        return day not in self._holidays

    def _require_coverage(self, day: dt.date) -> None:
        if day.year not in self._covered_years:
            raise CalendarCoverageError(
                f"trading calendar has no data for {day.year} "
                f"(covered years: {sorted(self._covered_years)}); refusing to guess "
                "whether the exchange was open"
            )

    def _at(self, day: dt.date, moment: dt.time) -> dt.datetime:
        return dt.datetime.combine(day, moment, tzinfo=self._zone)
