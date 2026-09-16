"""NSE's daily bhavcopy: one file per trading day, everything that traded.

This is the survivorship-bias fix, and it works by not reconstructing
anything. A company delisted in 2019 appears in 2018's file and is absent
from 2020's, because each file is a photograph of what actually traded
that day. There is no membership history to compile and therefore no
announcement that can be missed -- which is the failure mode of the
alternative (scraping NIFTY 50 constituent-change press releases), where
every omission is invisible.

docs/SPECIFICATION.md section 2.1 asks for exactly this: "use point-in-time
membership; do not backtest today's NIFTY 50 members all the way into the
past", over a "point-in-time liquid NSE equity universe".

**Two formats, one shape.** NSE changed the bhavcopy layout in 2024:

    old (2015 - 2024)  /content/historical/EQUITIES/YYYY/MON/cmDDMONYYYYbhav.csv.zip
                       SYMBOL, SERIES, OPEN..CLOSE, TOTTRDQTY, TOTTRDVAL, ISIN
    new (2024 - )      /content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
                       TckrSymb, SctySrs, OpnPric..ClsPric, TtlTradgVol, TtlTrfVal, ISIN

The two overlap for part of 2024, and on 2024-06-03 they agree exactly:
1,926 EQ symbols in both, with no disagreement in close, volume, traded
value or ISIN. That cross-check is what justifies treating them as one
series; the column mapping below was verified against the source rather
than inferred from the names.

The fetcher tries the new URL first and falls back to the old one, rather
than switching on a hardcoded cutover date. NSE has moved this boundary
once already, and a date constant would silently start returning nothing.

**ISIN is why this is usable.** Symbols get renamed; ISIN does not. A
universe keyed on symbol silently treats a rename as a delisting plus a
listing, which looks like a position that vanished and a new candidate
that appeared. Every row here carries both.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

NEW_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{day:%Y%m%d}_F_0000.csv.zip"
)
OLD_URL = (
    "https://nsearchives.nseindia.com/content/historical/EQUITIES/"
    "{day:%Y}/{month}/cm{day:%d}{month}{day:%Y}bhav.csv.zip"
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

EQUITY_SERIES = "EQ"
"""Only the plain cash-equity series. The file also carries BE (trade-for-
trade), SM/ST (SME platform) and debt series, which have different
settlement, different liquidity and, for SME, a different lot size. A
universe that swept them in would propose trades this system cannot
execute the way it believes it can."""

# Column mappings, verified against real files of each vintage.
_NEW_COLUMNS = {
    "symbol": "TckrSymb",
    "series": "SctySrs",
    "isin": "ISIN",
    "open": "OpnPric",
    "high": "HghPric",
    "low": "LwPric",
    "close": "ClsPric",
    "volume": "TtlTradgVol",
    "traded_value": "TtlTrfVal",
    "trades": "TtlNbOfTxsExctd",
}
_OLD_COLUMNS = {
    "symbol": "SYMBOL",
    "series": "SERIES",
    "isin": "ISIN",
    "open": "OPEN",
    "high": "HIGH",
    "low": "LOW",
    "close": "CLOSE",
    "volume": "TOTTRDQTY",
    "traded_value": "TOTTRDVAL",
    "trades": "TOTALTRADES",
}


class BhavcopyError(RuntimeError):
    """The file could not be fetched or did not have a recognised shape."""


class BhavcopyNotPublished(BhavcopyError):
    """No bhavcopy exists for this date in either format.

    Distinct from a general failure because the usual cause is benign --
    the exchange was closed. The caller decides whether that is expected
    (it should already know, from the trading calendar) or a real gap.
    """


@dataclass(frozen=True, slots=True)
class BhavcopyRow:
    """One instrument's trading on one day."""

    symbol: str
    isin: str
    session_date: dt.date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    traded_value: Decimal
    """Turnover in rupees. This, not volume, is what
    ``universe.min_avg_daily_value_inr`` filters on -- a million shares of
    a Rs 3 stock is not liquidity in any sense this system can use."""

    trades: int

    @property
    def instrument_id(self) -> str:
        return f"NSE:{self.symbol}"


def bhavcopy_urls(day: dt.date) -> tuple[str, ...]:
    """Candidate URLs, newest format first."""
    month = f"{day:%b}".upper()
    return (NEW_URL.format(day=day), OLD_URL.format(day=day, month=month))


def fetch_bhavcopy(
    day: dt.date, *, cache_dir: Path | None = None, timeout: int = 60
) -> bytes:
    """The raw zip for one day, from cache when available.

    Caching is not an optimisation here, it is the difference between a
    backfill that can be resumed and one that starts again from 2015 every
    time something goes wrong 2,000 files in.
    """
    cached = cache_dir / f"bhavcopy-{day:%Y-%m-%d}.zip" if cache_dir else None
    if cached is not None and cached.is_file():
        return cached.read_bytes()

    errors: list[str] = []
    for url in bhavcopy_urls(day):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                payload = bytes(response.read())
        except urllib.error.HTTPError as exc:
            errors.append(f"{exc.code} {url}")
            continue
        except Exception as exc:  # noqa: BLE001 - network of any kind
            raise BhavcopyError(f"could not fetch {url}: {exc}") from exc

        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(payload)
        return payload

    raise BhavcopyNotPublished(
        f"no bhavcopy published for {day} in either format ({'; '.join(errors)})"
    )


def parse_bhavcopy(
    payload: bytes, *, expected_date: dt.date | None = None
) -> tuple[BhavcopyRow, ...]:
    """Parse either format into one row shape.

    The format is detected from the header rather than from the date, so a
    file downloaded before NSE's cutover still parses after it.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
        names = archive.namelist()
        if not names:
            raise BhavcopyError("bhavcopy archive is empty")
        text = archive.read(names[0]).decode("utf-8", errors="replace")
    except zipfile.BadZipFile as exc:
        raise BhavcopyError(f"bhavcopy is not a readable zip: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text))
    fields = {name.strip() for name in (reader.fieldnames or [])}
    if _NEW_COLUMNS["symbol"] in fields:
        columns: dict[str, str] = _NEW_COLUMNS
        date_column = "TradDt"
        date_formats: tuple[str, ...] = ("%Y-%m-%d",)
    elif _OLD_COLUMNS["symbol"] in fields:
        # Two accepted spellings, because NSE used both. 2020-07-13 is
        # published with a two-digit year ("13-Jul-20") while every
        # neighbouring session uses four. Accepting only the common one
        # silently drops that session, and a missing session is a hole
        # in the universe that looks exactly like a quiet day.
        columns = _OLD_COLUMNS
        date_column = "TIMESTAMP"
        date_formats = ("%d-%b-%Y", "%d-%b-%y")
    else:
        raise BhavcopyError(
            f"unrecognised bhavcopy layout; columns were {sorted(fields)}. "
            "Refusing to guess at a mapping -- a mis-mapped price column "
            "produces a plausible and entirely wrong price series."
        )

    missing = {v for v in columns.values()} - fields
    if missing:
        raise BhavcopyError(f"bhavcopy is missing expected column(s): {sorted(missing)}")

    rows: list[BhavcopyRow] = []
    for line_number, record in enumerate(reader, start=2):
        if str(record.get(columns["series"], "")).strip().upper() != EQUITY_SERIES:
            continue
        try:
            session_date = _parse_session_date(
                str(record[date_column]).strip(), date_formats
            )
            rows.append(
                BhavcopyRow(
                    symbol=str(record[columns["symbol"]]).strip(),
                    isin=str(record[columns["isin"]]).strip(),
                    session_date=session_date,
                    open=Decimal(str(record[columns["open"]]).strip()),
                    high=Decimal(str(record[columns["high"]]).strip()),
                    low=Decimal(str(record[columns["low"]]).strip()),
                    close=Decimal(str(record[columns["close"]]).strip()),
                    volume=int(float(str(record[columns["volume"]]).strip())),
                    traded_value=Decimal(str(record[columns["traded_value"]]).strip()),
                    trades=int(float(str(record[columns["trades"]]).strip() or 0)),
                )
            )
        except (KeyError, ValueError, ArithmeticError) as exc:
            raise BhavcopyError(f"bhavcopy line {line_number}: {exc}") from exc

    if not rows:
        raise BhavcopyError("bhavcopy contained no EQ-series rows")

    if expected_date is not None:
        actual = {row.session_date for row in rows}
        if actual != {expected_date}:
            # NSE has served a file under the wrong date before. Trusting
            # the filename over the contents would silently shift a whole
            # day of prices, which no downstream check would catch.
            raise BhavcopyError(
                f"bhavcopy requested for {expected_date} contains {sorted(actual)}"
            )

    return tuple(rows)


def _parse_session_date(raw: str, formats: tuple[str, ...]) -> dt.date:
    for date_format in formats:
        try:
            return dt.datetime.strptime(raw, date_format).date()
        except ValueError:
            continue
    raise ValueError(f"unparseable session date {raw!r} (tried {list(formats)})")


def load_bhavcopy(
    day: dt.date, *, cache_dir: Path | None = None, timeout: int = 60
) -> tuple[BhavcopyRow, ...]:
    """Fetch and parse one day, verifying the file is for the day asked for."""
    return parse_bhavcopy(
        fetch_bhavcopy(day, cache_dir=cache_dir, timeout=timeout), expected_date=day
    )
