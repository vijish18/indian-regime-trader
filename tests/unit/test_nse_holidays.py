"""Invariants on the *shipped* market calendar (``config/nse_holidays.csv``).

Most tests in this repository check code. These check data -- because for
a trading calendar, wrong data and wrong code fail identically, and the
data is the part with no type checker.

The failure mode these are built around is the **silently partial list**.
While assembling this file, a web-search summary of NSE's 2026 holidays
returned five dates, four of which were weekends. It looked like an
answer. Had it been used, the system would have believed Republic Day,
Holi, Good Friday, Ambedkar Jayanti, Dussehra and Diwali-Balipratipada
were ordinary trading days, and would have sat waiting for fills from a
closed exchange on each of them. Nothing would have raised: a holiday
file with entries in it is a "covered" year as far as
``NSETradingCalendar`` is concerned, and every missing closure just looks
like a normal open day.

So the tests below are mostly *shape* assertions -- a year with too few
weekday closures is rejected even though every row in it is individually
correct. That is the only kind of check that catches a truncated source.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import pytest

from data.calendar import NSETradingCalendar

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLIDAY_FILE = REPO_ROOT / "config" / "nse_holidays.csv"

MIN_WEEKDAY_CLOSURES_PER_YEAR = 10
"""NSE has closed on 12-17 weekdays in each recent year. Ten is a floor
set below every observed value, so it flags a truncated list without
failing on a genuinely light year."""

MAX_WEEKDAY_CLOSURES_PER_YEAR = 25
"""And an upper bound, because a parse that accidentally swept in the
Futures & Options or Currency segment's list -- or duplicated a table --
would show up as an implausible count rather than as anything wrong with
an individual row."""


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    with HOLIDAY_FILE.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def calendar() -> NSETradingCalendar:
    return NSETradingCalendar.from_file(HOLIDAY_FILE)


# ---------------------------------------------------------------------------
# The file loads at all
# ---------------------------------------------------------------------------


def test_the_shipped_calendar_loads(calendar: NSETradingCalendar) -> None:
    """Phase 23 shipped this file containing only a header row, which made
    every calendar call raise. The system failed closed -- correctly -- but
    it also meant nothing could run at all."""
    assert calendar.covered_years


def test_every_row_carries_its_provenance(rows: list[dict[str, str]]) -> None:
    """A holiday file nobody can trace is one nobody can re-verify, and
    re-verification is the only defence against a quietly wrong date.

    Three provenance kinds, and the distinction matters when auditing a
    date: ``NSE/CMTR/...`` is transcribed from that circular,
    ``nse-api/...`` came from the live holiday master, and
    ``derived/nifty50-no-bar`` means the exchange printed no NIFTY 50 bar
    that weekday -- a fact from price data rather than a document, used
    for years with no circular on file and for closures announced after
    one. ``scripts/build_nse_holidays.py`` regenerates them all.
    """
    assert rows
    allowed = ("NSE/CMTR/", "nse-api/", "derived/")
    for row in rows:
        source = row["source"].strip()
        assert source, f"{row['date']} has no source"
        assert source.startswith(allowed), f"{row['date']}: {source!r}"


def test_the_reconciled_closures_that_the_circulars_missed_are_present() -> None:
    """Regression test for three days the shipped calendar got wrong.

    Reconciling against NIFTY 50's actual bar history found three
    weekdays the exchange was shut and this calendar said were open --
    the dangerous direction, where the system waits all day for fills
    from a closed market:

        2023-06-29  Bakri Id, moved from the 28th after the circular
        2024-01-22  Ram Mandir consecration
        2024-11-20  Maharashtra assembly elections

    None appears in its year's December circular.
    """
    calendar = NSETradingCalendar.from_file(HOLIDAY_FILE)
    for day in (dt.date(2023, 6, 29), dt.date(2024, 1, 22), dt.date(2024, 11, 20)):
        assert day.weekday() < 5
        assert calendar.is_trading_day(day) is False, f"{day} should be closed"

    # ...and the date the circular wrongly listed is open again, because
    # the index demonstrably traded that day.
    assert calendar.is_trading_day(dt.date(2023, 6, 28)) is True


# ---------------------------------------------------------------------------
# Shape: the checks that catch a truncated source
# ---------------------------------------------------------------------------


def _weekday_closures_by_year(rows: list[dict[str, str]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        day = dt.date.fromisoformat(row["date"])
        if day.weekday() < 5:
            counts[day.year] = counts.get(day.year, 0) + 1
    return counts


def test_each_covered_year_has_a_plausible_number_of_closures(
    rows: list[dict[str, str]],
) -> None:
    """The partial-list guard. Every individual row can be correct and the
    year still be dangerously incomplete."""
    counts = _weekday_closures_by_year(rows)
    assert counts, "no weekday closures at all -- the file is empty or unparsed"
    for year, count in sorted(counts.items()):
        assert count >= MIN_WEEKDAY_CLOSURES_PER_YEAR, (
            f"{year} has only {count} weekday closures, which is too few to be a "
            f"complete NSE year. A truncated holiday list makes the system treat "
            f"real closures as trading days. Re-run scripts/build_nse_holidays.py."
        )
        assert count <= MAX_WEEKDAY_CLOSURES_PER_YEAR, (
            f"{year} has {count} weekday closures, more than NSE has ever had in "
            f"the cash segment -- check for a duplicated table or a non-CM segment."
        )


def test_covered_years_are_contiguous(calendar: NSETradingCalendar) -> None:
    """A gap in the middle is almost certainly an extraction failure rather
    than a deliberate choice. Missing years at the *ends* are fine and
    expected (see the 2019-2021 note in docs/MARKET_CALENDAR.md)."""
    years = sorted(calendar.covered_years)
    assert years == list(range(years[0], years[-1] + 1)), f"gap in coverage: {years}"


def test_no_duplicate_dates(rows: list[dict[str, str]]) -> None:
    dates = [row["date"] for row in rows]
    assert len(dates) == len(set(dates)), "duplicate calendar entries"


def test_rows_are_sorted_by_date(rows: list[dict[str, str]]) -> None:
    """Keeps the committed diff readable when a mid-year closure is added,
    so a reviewer sees one inserted line rather than a reshuffled file."""
    dates = [row["date"] for row in rows]
    assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# Muhurat: why nothing is marked `special`
# ---------------------------------------------------------------------------


def test_nothing_is_marked_as_a_special_session(rows: list[dict[str, str]]) -> None:
    """``NSETradingCalendar.session()`` returns the regular 09:15-15:30
    window for anything marked ``special``. It has no way to express
    Muhurat's actual ~1 hour ceremonial timings, which NSE notifies
    separately and sometimes only days in advance.

    So marking a Muhurat day ``special`` would tell this system that a
    Sunday in November is a normal full trading day -- worse than not
    knowing. Until the calendar can carry per-session times, these days
    are recorded closed and the strategy does not trade them.
    """
    special = [row["date"] for row in rows if row["session_type"].strip().lower() == "special"]
    assert special == [], (
        f"{special} are marked 'special', but session() would give them regular "
        f"trading hours. See docs/MARKET_CALENDAR.md."
    )


def test_muhurat_days_are_recorded_as_closed(rows: list[dict[str, str]]) -> None:
    """NSE marks the Muhurat row with an asterisk. Those days have no
    regular session, so they must not be trading days."""
    muhurat = [row for row in rows if "*" in row["description"]]
    assert muhurat, "expected at least one asterisked Muhurat row"
    for row in muhurat:
        assert row["session_type"].strip().lower() == "closed"


def test_a_weekday_muhurat_day_is_not_a_trading_day(calendar: NSETradingCalendar) -> None:
    """The case that matters, because the weekend ones are closed anyway.

    2025-10-21 was a Tuesday: an ordinary working day on which the
    exchange held only the Muhurat session. If this returned True the
    system would have traded a full day into a market that was not open.
    """
    day = dt.date(2025, 10, 21)
    assert day.weekday() < 5
    assert calendar.is_trading_day(day) is False


# ---------------------------------------------------------------------------
# Drift: the annual circular is not the last word
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "what"),
    [
        (dt.date(2024, 5, 20), "Parliamentary Elections, added by NSE/CMTR/61518"),
        (dt.date(2026, 1, 15), "Municipal Corporation Election, absent from NSE/CMTR/71775"),
    ],
)
def test_mid_year_closures_are_present(
    calendar: NSETradingCalendar, day: dt.date, what: str
) -> None:
    """NSE adds closures during the year by partial modification, and both
    of these are weekdays absent from their year's December circular. A
    build that read only the annual circulars would have the system
    trading on them."""
    assert day.weekday() < 5
    assert calendar.is_trading_day(day) is False, f"{day} ({what}) should be closed"


# ---------------------------------------------------------------------------
# Spot checks against the source documents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "day",
    [
        dt.date(2026, 1, 26),   # Republic Day, Monday
        dt.date(2026, 3, 3),    # Holi, Tuesday
        dt.date(2026, 4, 3),    # Good Friday
        dt.date(2026, 10, 2),   # Gandhi Jayanti, Friday
        dt.date(2026, 12, 25),  # Christmas, Friday
        dt.date(2025, 8, 15),   # Independence Day, Friday
    ],
)
def test_known_weekday_closures(calendar: NSETradingCalendar, day: dt.date) -> None:
    """Every one of these is a weekday the exchange is shut, and every one
    was missing from the truncated search-summary list described in this
    module's docstring."""
    assert day.weekday() < 5
    assert calendar.is_trading_day(day) is False


@pytest.mark.parametrize(
    "day",
    [
        dt.date(2026, 9, 16),   # ordinary Wednesday
        dt.date(2026, 1, 27),   # the day after Republic Day
        dt.date(2025, 1, 2),    # ordinary Thursday
    ],
)
def test_known_trading_days(calendar: NSETradingCalendar, day: dt.date) -> None:
    """The other half. A calendar that called every day a holiday would
    pass every test above and trade nothing."""
    assert calendar.is_trading_day(day) is True


def test_an_uncovered_year_still_refuses_to_answer(calendar: NSETradingCalendar) -> None:
    """Populating the file must not have weakened the refusal that
    protects the years it does not cover."""
    from data.errors import CalendarCoverageError

    uncovered = min(calendar.covered_years) - 1
    with pytest.raises(CalendarCoverageError):
        calendar.is_trading_day(dt.date(uncovered, 6, 3))


# ---------------------------------------------------------------------------
# Is this calendar usable *now*?
# ---------------------------------------------------------------------------


def test_the_calendar_covers_the_current_year(calendar: NSETradingCalendar) -> None:
    """This test is designed to start failing.

    When the shipped calendar stops covering today, every calendar call
    raises ``CalendarCoverageError`` and the system fails closed -- which
    is correct, but is a thing to discover in December, not at 09:15 on
    the second of January. NSE publishes each year's circular the
    preceding December.

    If this fails: add next year's circular reference to
    ``ANNUAL_CIRCULARS`` in scripts/build_nse_holidays.py and re-run it.
    """
    today = dt.date.today()
    assert today.year in calendar.covered_years, (
        f"the shipped calendar does not cover {today.year}. Add that year's NSE "
        f"circular to scripts/build_nse_holidays.py and re-run it; until then the "
        f"calendar refuses to answer and nothing can trade."
    )
