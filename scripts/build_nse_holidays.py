"""Regenerate ``config/nse_holidays.csv`` from NSE's own published sources.

    python scripts/build_nse_holidays.py            # rebuild from cached/downloaded sources
    python scripts/build_nse_holidays.py --check    # verify the committed file matches

This exists so the calendar data has *provenance*. A holiday file that
someone typed from memory is indistinguishable, on inspection, from one
transcribed from the exchange's circulars -- right up until the day it is
wrong and the system trades into a closed market, or sits out a real
session without anyone noticing. Every row this writes carries the NSE
circular reference it came from.

Two source kinds, deliberately both:

* **Annual circulars** (``https://nsearchives.nseindia.com/content/circulars/CMTR*.pdf``)
  are the authoritative statement for a calendar year, published each
  December for the year ahead. These are what historical years come from.

* **The live holiday-master API** (``/api/holiday-master?type=trading``,
  ``CM`` segment) is the authoritative statement for *right now*. It
  matters because the annual circular is not the last word: NSE adds
  closures during the year by partial modification. Two real examples
  this file already contains --

      2024-05-20  Parliamentary Elections in Mumbai   (NSE/CMTR/61518)
      2026-01-15  Municipal Corporation Election      (absent from NSE/CMTR/71775)

  -- neither of which appears in the December circular for its year. A
  system that trusted the circular alone would have believed the exchange
  was open on both days.

**Requires** ``pypdf`` (``pip install pypdf``) and network access. It is
deliberately NOT a project dependency: this is a maintenance tool run a
few times a year by a human, not something the trading system imports.

**The output is not automatically trustworthy.** Read
``docs/MARKET_CALENDAR.md`` before relying on a year for live trading.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "config" / "nse_holidays.csv"
CACHE = REPO_ROOT / ".holiday-cache"

CIRCULAR_URLS = (
    "https://nsearchives.nseindia.com/content/circulars/{ref}.pdf",
    "https://archives.nseindia.com/content/circulars/{ref}.pdf",
)

API_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
API_SEGMENT = "CM"
"""Capital market segment. The endpoint returns a dozen segments (FO, CD,
COM, ...) whose holidays differ; this system trades cash equities, so
reading any other segment's list would be quietly wrong."""

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

ANNUAL_CIRCULARS: dict[int, str] = {
    2022: "CMTR50560",
    2023: "CMTR54757",
    2024: "CMTR59722",
    2025: "CMTR65587",
    2026: "CMTR71775",
}
"""Year -> the December circular that notified that year's holidays.

2019-2021 are deliberately absent rather than guessed: their circular
references were not located from NSE's own archive. See
docs/MARKET_CALENDAR.md -- a year missing from this file is a year the
calendar refuses to answer for, which is the correct behavior."""

AMENDMENT_CIRCULARS: tuple[str, ...] = ("CMTR61518",)
"""Partial-modification circulars that add a closure mid-year. Each is
parsed for the single date it notifies."""

ROW = re.compile(
    r"(?P<sr>\d{1,2})\s+(?P<date>[A-Z][a-z]+ \d{1,2}\s*,\s*\d{4})[\s\d,]*?"
    r"(?P<day>Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
)
"""Matches only a row's *anchor* -- serial number, date, weekday.

It deliberately does not try to capture the description, and it runs over
whitespace-collapsed text rather than line by line. Both choices are
fixes for a real bug: NSE's 2022 circular renders row 13 as

    13 November 08,2022
    08,2022
    Tuesday Gurunanak Jayanti

with the date fragment repeated on its own line. A line-anchored regex
requiring date and weekday together silently skipped it, so the shipped
calendar said NSE was open on Guru Nanak Jayanti 2022. Nothing caught it
-- the day-of-week cross-check only validates rows that *did* parse, and
12 closures is a plausible count. It was found by reconciling against
NIFTY 50 price data, which had no bar that day.

``[\\s\\d,]*?`` tolerates the repeated fragment; the description is taken
as the text between one anchor and the next (see
:func:`parse_annual_circular`), so a description can wrap lines too.
"""

TABLE_SPLIT = "The holidays falling on"
"""The circular prints two tables -- trading holidays, then those falling
on a weekend -- and each restarts its numbering at 1. Splitting here lets
the completeness check below assert 1..N per table."""

BOILERPLATE = re.compile(
    r"National Stock Exchange.*|Page \d+ of \d+.*|\*Muhurat.*|The holidays falling.*",
    re.S,
)
AMENDMENT_DATE = re.compile(
    r"notifies\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s*"
    r"([A-Z][a-z]+ \d{1,2},\s*\d{4})\s+as a trading holiday on account of\s+(.+?)(?:\.|\s+in\s)",
    re.DOTALL,
)


@dataclass(frozen=True)
class Entry:
    day: dt.date
    description: str
    source: str

    @property
    def is_muhurat(self) -> bool:
        """NSE marks the Diwali Laxmi Pujan row with an asterisk to mean
        "closed for regular trading, but a Muhurat session is held"."""
        return "*" in self.description


def _fetch(url: str, *, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return bytes(response.read())


def _circular_text(ref: str) -> str:
    try:
        import pypdf
    except ImportError:  # pragma: no cover - a maintenance-tool dependency
        raise SystemExit(
            "this script needs pypdf to read NSE's circular PDFs: pip install pypdf"
        ) from None

    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{ref}.pdf"
    if not cached.is_file():
        last_error: Exception | None = None
        for template in CIRCULAR_URLS:
            try:
                cached.write_bytes(_fetch(template.format(ref=ref)))
                break
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                last_error = exc
        else:
            raise SystemExit(f"could not download circular {ref}: {last_error}")

    reader = pypdf.PdfReader(cached)
    return "\n".join(page.extract_text() for page in reader.pages)


def parse_annual_circular(ref: str, expected_year: int) -> list[Entry]:
    """Every dated row in a year's circular, both tables.

    The circular prints two tables: the trading holidays proper, and the
    ones that fall on a weekend. Both are captured. The weekend rows are
    redundant for the calendar's own logic (it closes weekends anyway),
    but transcribing the circular in full is what makes the committed
    file checkable against the source by eye.
    """
    text = _circular_text(ref)

    stated = re.search(r"calendar year\s*(\d{3}\s?\d)", text)
    if stated:
        # NSE's PDFs sometimes render "2026" with a stray space ("202 6").
        if int(stated.group(1).replace(" ", "")) != expected_year:
            raise SystemExit(
                f"{ref} says calendar year {stated.group(1)!r}, expected {expected_year}"
            )

    collapsed = " ".join(text.split())
    source = f"NSE/{ref[:4]}/{ref[4:]}"
    entries: list[Entry] = []

    for table in collapsed.split(TABLE_SPLIT):
        anchors = list(ROW.finditer(table))
        if not anchors:
            continue

        # --- completeness, the check that would have caught the 2022 bug ---
        #
        # NSE numbers each table's rows 1..N. That is the document stating
        # its own row count, so a dropped row is detectable without any
        # external source: the serials simply stop being contiguous.
        #
        # This matters more than validating the rows that did parse. A
        # missing closure does not look wrong -- it looks like an ordinary
        # trading day, and the system waits all day for fills from a shut
        # exchange.
        serials = [int(a.group("sr")) for a in anchors]
        if serials != list(range(1, len(serials) + 1)):
            raise SystemExit(
                f"{ref}: row serial numbers are {serials}, expected 1..{len(serials)}. "
                "A row was dropped or double-counted; refusing to write a calendar "
                "that may be missing a closure."
            )

        for index, anchor in enumerate(anchors):
            end = anchors[index + 1].start() if index + 1 < len(anchors) else len(table)
            description = BOILERPLATE.sub("", table[anchor.end() : end]).strip(" .,")

            day = dt.datetime.strptime(
                re.sub(r"\s*,\s*", ", ", anchor.group("date")), "%B %d, %Y"
            ).date()

            # Cross-check the date against the weekday NSE printed beside
            # it. Two independent statements of the same fact; if they
            # disagree, the parse is wrong (or the circular is), and
            # guessing which would be exactly the wrong instinct.
            if day.strftime("%A") != anchor.group("day"):
                raise SystemExit(
                    f"{ref}: {day} is a {day.strftime('%A')} but the circular prints "
                    f"{anchor.group('day')}; refusing to guess which is right"
                )
            if day.year != expected_year:
                raise SystemExit(f"{ref}: parsed {day}, which is not in {expected_year}")
            if not description:
                raise SystemExit(f"{ref}: {day} parsed with an empty description")

            entries.append(Entry(day, description, source))

    if not entries:
        raise SystemExit(f"{ref}: no holiday rows parsed -- the PDF layout may have changed")
    return entries


def parse_amendment_circular(ref: str) -> list[Entry]:
    text = _circular_text(ref)
    match = AMENDMENT_DATE.search(" ".join(text.split()))
    if not match:
        raise SystemExit(f"{ref}: could not find the notified date in this amendment")
    day = dt.datetime.strptime(re.sub(r",\s*", ", ", match.group(1)), "%B %d, %Y").date()
    return [Entry(day, match.group(2).strip(), f"NSE/{ref[:4]}/{ref[4:]}")]


def fetch_api_entries() -> list[Entry]:
    """The live holiday master, for the current year.

    This is what catches a closure added after the annual circular. A
    failure here is reported, not swallowed: silently falling back to the
    circular alone would reintroduce exactly the gap this call exists to
    close.
    """
    payload = json.loads(_fetch(API_URL).decode("utf-8"))
    rows = payload.get(API_SEGMENT)
    if not rows:
        raise SystemExit(
            f"holiday-master API returned no {API_SEGMENT} segment "
            f"(got: {sorted(payload)}); refusing to treat that as 'no holidays'"
        )
    entries = []
    for row in rows:
        day = dt.datetime.strptime(row["tradingDate"], "%d-%b-%Y").date()
        entries.append(Entry(day, str(row["description"]).strip(), "nse-api/holiday-master"))
    return entries


def build(*, use_api: bool = True) -> list[Entry]:
    merged: dict[dt.date, Entry] = {}

    for year, ref in sorted(ANNUAL_CIRCULARS.items()):
        for entry in parse_annual_circular(ref, year):
            merged[entry.day] = entry

    for ref in AMENDMENT_CIRCULARS:
        for entry in parse_amendment_circular(ref):
            merged[entry.day] = entry

    if use_api:
        # `setdefault`, so the API can only ever *add* a closure the
        # circulars did not mention -- the mid-year modification case
        # this call exists for. It cannot remove one, and it cannot
        # overwrite a historical year's transcription with a live
        # response that does not cover that year.
        for entry in fetch_api_entries():
            merged.setdefault(entry.day, entry)

    return sorted(merged.values(), key=lambda e: e.day)


def index_trading_days(start: dt.date, end: dt.date) -> set[dt.date]:
    """Dates NIFTY 50 actually printed a bar, straight from Kite.

    This is the exchange's own record of what it did, as opposed to the
    circulars' record of what it intended to do the following year. Both
    are authoritative about different things, which is the whole point of
    comparing them.
    """
    from broker.zerodha.kite_historical import NIFTY_50_TOKEN, KiteHistoricalClient
    from broker.zerodha.kite_session import load_session

    session = load_session()
    client = KiteHistoricalClient(session.api_key, access_token=session.access_token)
    return {candle.session_date for candle in client.daily_candles(NIFTY_50_TOKEN, start, end)}


def reconcile(entries: list[Entry], traded: set[dt.date]) -> tuple[list[Entry], list[str]]:
    """Correct the circular-derived calendar against what the market did.

    An annual circular is published each December for the year *ahead*, so
    it is a forecast. NIFTY 50's bar history is the record. For any date
    that has already happened, the record wins -- and reconciling the two
    found three real errors in the shipped calendar over 2022-2025:

        2023-06-29  Bakri Id, moved from the 28th after the circular
        2024-01-22  Ram Mandir consecration
        2024-11-20  Maharashtra assembly elections

    Each was a weekday the exchange was shut and this calendar said was
    open, which is the dangerous direction: the system would have sat
    waiting for fills all day.

    Muhurat days are the deliberate exception. The index prints a bar
    because the ceremonial session trades, but there is no regular
    session, so they stay closed (see docs/MARKET_CALENDAR.md).
    """
    if not traded:
        raise SystemExit("no index trading days supplied; refusing to reconcile against nothing")

    horizon = max(traded)
    by_day = {entry.day: entry for entry in entries}
    notes: list[str] = []

    # 1. A past weekday with no index bar was a closure, whatever the
    #    circular said -- or did not say.
    day = min(traded)
    while day <= horizon:
        if day.weekday() < 5 and day not in traded and day not in by_day:
            by_day[day] = Entry(
                day,
                "closure recorded by the exchange (no NIFTY 50 session)",
                "derived/nifty50-no-bar",
            )
            notes.append(f"  + {day} ({day:%a}) closed: index printed no bar")
        day += dt.timedelta(days=1)

    # 2. A past closure the index actually traded through was superseded.
    #    Muhurat rows are exempt: they trade a ceremonial session only.
    for existing in list(by_day.values()):
        if existing.day > horizon or existing.day.weekday() >= 5:
            continue
        if existing.day in traded and not existing.is_muhurat:
            del by_day[existing.day]
            notes.append(
                f"  - {existing.day} ({existing.day:%a}) reopened: index traded "
                f"(circular said {existing.description!r})"
            )

    return sorted(by_day.values(), key=lambda e: e.day), notes


def to_rows(entries: list[Entry]) -> list[dict[str, str]]:
    rows = []
    for entry in entries:
        # Muhurat days are recorded as CLOSED, not as a special session.
        #
        # On a Muhurat day the exchange does not hold regular trading --
        # only a ~1 hour ceremonial session whose timings NSE notifies
        # separately, and sometimes not until days beforehand.
        # `NSETradingCalendar.session()` has no way to express those
        # hours: it returns the regular 09:15-15:30 window for anything
        # marked `special`. Marking these `special` would therefore tell
        # this system a Sunday in November is a normal full trading day.
        #
        # Recording them closed means the strategy simply does not trade
        # Muhurat. For a daily, long-only cash-equity system that is the
        # right answer anyway -- it is a low-liquidity ceremonial session
        # on a separate settlement schedule -- and it is the fail-closed
        # answer regardless. See docs/MARKET_CALENDAR.md.
        rows.append(
            {
                "date": entry.day.isoformat(),
                "description": entry.description,
                "session_type": "closed",
                "source": entry.source,
            }
        )
    return rows


def render(rows: list[dict[str, str]]) -> str:
    buffer: list[str] = []
    writer = csv.DictWriter(
        _Sink(buffer), fieldnames=["date", "description", "session_type", "source"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return "".join(buffer)


class _Sink:
    def __init__(self, into: list[str]) -> None:
        self._into = into

    def write(self, text: str) -> int:
        self._into.append(text)
        return len(text)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed file matches the sources instead of rewriting it",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="skip the live holiday-master cross-check (offline; historical years only)",
    )
    parser.add_argument(
        "--reconcile-with-kite",
        action="store_true",
        help=(
            "correct and extend the calendar against NIFTY 50's actual bar history "
            "(needs a Kite session: python scripts/kite_login.py)"
        ),
    )
    parser.add_argument(
        "--history-from",
        type=dt.date.fromisoformat,
        default=dt.date(2015, 1, 1),
        help="earliest date to reconcile from, with --reconcile-with-kite",
    )
    args = parser.parse_args(argv[1:])

    entries = build(use_api=not args.no_api)

    if args.reconcile_with_kite:
        print(f"reconciling against NIFTY 50 bars from {args.history_from}...")
        traded = index_trading_days(args.history_from, dt.date.today())
        entries, notes = reconcile(entries, traded)
        if notes:
            print(f"{len(notes)} correction(s) from the exchange's own record:")
            for note in notes:
                print(note)
        else:
            print("  no corrections needed")
        print()

    rendered = render(to_rows(entries))

    years = sorted({e.day.year for e in entries})
    print(f"{len(entries)} entries across {len(years)} year(s): {years}")
    for year in years:
        in_year = [e for e in entries if e.day.year == year]
        weekday = [e for e in in_year if e.day.weekday() < 5]
        print(f"  {year}: {len(in_year):2d} entries, {len(weekday):2d} on weekdays")

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if current != rendered:
            print(f"\n{OUTPUT} is OUT OF DATE with respect to NSE's sources.", file=sys.stderr)
            return 1
        print(f"\n{OUTPUT} matches its sources.")
        return 0

    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"\nwrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
