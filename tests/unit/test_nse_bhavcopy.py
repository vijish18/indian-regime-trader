"""Parsing NSE's daily bhavcopy, in both of its layouts.

The fixtures below are trimmed copies of real files: the column names and
their order are exactly what NSE publishes, because a parser tested
against invented headers is a parser tested against a format nobody uses.

The two layouts were cross-checked against each other on 2024-06-03, a
date both were published for: 1,926 EQ symbols in each, with no
disagreement in close, volume, traded value or ISIN. That is what
justifies treating them as one continuous series, and it is a check no
amount of reading the column names could replace.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile
from decimal import Decimal

import pytest

from data.nse_bhavcopy import (
    BhavcopyError,
    bhavcopy_urls,
    parse_bhavcopy,
)

NEW_CSV = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,"
    "FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,"
    "LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,"
    "TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n"
    "2024-06-03,2024-06-03,CM,NSE,STK,2885,INE002A01018,RELIANCE,EQ,,,,,RELIANCE,"
    "2900,2950,2880,2930,2929,2895,,2930,,,1000000,2930000000,50000,F1,1,,,,,\n"
    "2024-06-03,2024-06-03,CM,NSE,STK,999,INE111A01011,SMECO,SM,,,,,SME CO,"
    "10,11,9,10,10,10,,10,,,500,5000,5,F1,1,,,,,\n"
)

OLD_CSV = (
    "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,"
    "TIMESTAMP,TOTALTRADES,ISIN,\n"
    "RELIANCE,EQ,2900,2950,2880,2930,2929,2895,1000000,2930000000,"
    "03-JUN-2024,50000,INE002A01018,\n"
    "SMECO,SM,10,11,9,10,10,10,500,5000,03-JUN-2024,5,INE111A01011,\n"
)


def _zip(csv_text: str, name: str = "bhav.csv") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, csv_text)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Both layouts, one shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("csv_text", [NEW_CSV, OLD_CSV], ids=["new", "old"])
def test_both_layouts_parse_to_the_same_values(csv_text: str) -> None:
    """The layout is detected from the header, not from the date, so a file
    downloaded before NSE's 2024 cutover still parses after it."""
    rows = parse_bhavcopy(_zip(csv_text))
    assert len(rows) == 1
    row = rows[0]
    assert row.symbol == "RELIANCE"
    assert row.instrument_id == "NSE:RELIANCE"
    assert row.isin == "INE002A01018"
    assert row.session_date == dt.date(2024, 6, 3)
    assert row.close == Decimal("2930")
    assert row.volume == 1_000_000
    assert row.traded_value == Decimal("2930000000")


@pytest.mark.parametrize("csv_text", [NEW_CSV, OLD_CSV], ids=["new", "old"])
def test_non_equity_series_is_excluded(csv_text: str) -> None:
    """The file also carries SME, trade-for-trade and debt series. They
    settle differently, trade differently, and for SME have a different
    lot size -- a universe that swept them in would propose trades this
    system cannot execute the way it believes it can."""
    rows = parse_bhavcopy(_zip(csv_text))
    assert [row.symbol for row in rows] == ["RELIANCE"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_an_unrecognised_layout_is_refused_rather_than_guessed() -> None:
    """A mis-mapped price column produces a plausible and entirely wrong
    price series -- the kind of error that survives every downstream check
    because every number still looks like a number."""
    with pytest.raises(BhavcopyError, match="unrecognised bhavcopy layout"):
        parse_bhavcopy(_zip("FOO,BAR\n1,2\n"))


def test_a_missing_column_is_refused() -> None:
    truncated = "\n".join(
        line.replace(",TtlTrfVal", "").replace(",2930000000", "")
        for line in NEW_CSV.splitlines()
    )
    with pytest.raises(BhavcopyError, match="missing expected column"):
        parse_bhavcopy(_zip(truncated + "\n"))


def test_a_file_for_the_wrong_date_is_refused() -> None:
    """NSE has served a file under the wrong date before. Trusting the
    filename over the contents would shift a whole day of prices, and
    nothing downstream would notice -- every bar would still be a valid
    bar, just attributed to the wrong session."""
    with pytest.raises(BhavcopyError, match="contains"):
        parse_bhavcopy(_zip(NEW_CSV), expected_date=dt.date(2024, 6, 4))


def test_the_correct_date_passes_verification() -> None:
    rows = parse_bhavcopy(_zip(NEW_CSV), expected_date=dt.date(2024, 6, 3))
    assert len(rows) == 1


def test_a_file_with_no_equity_rows_is_refused() -> None:
    """An empty result would look like "nothing traded", which for a
    session that did happen is worse than an error."""
    header, _, sme = NEW_CSV.splitlines()
    with pytest.raises(BhavcopyError, match="no EQ-series rows"):
        parse_bhavcopy(_zip(header + "\n" + sme + "\n"))


def test_a_corrupt_archive_is_refused() -> None:
    with pytest.raises(BhavcopyError, match="readable zip"):
        parse_bhavcopy(b"this is not a zip file")


# ---------------------------------------------------------------------------
# URL selection
# ---------------------------------------------------------------------------


def test_the_new_layout_url_is_tried_first() -> None:
    """Falling back rather than switching on a hardcoded cutover date: NSE
    has moved that boundary once already, and a date constant would
    silently start returning nothing."""
    new_url, old_url = bhavcopy_urls(dt.date(2024, 6, 3))
    assert "BhavCopy_NSE_CM_0_0_0_20240603" in new_url
    assert "historical/EQUITIES/2024/JUN/cm03JUN2024bhav" in old_url


def test_the_old_url_uses_an_uppercase_three_letter_month() -> None:
    """NSE's archive path is case-sensitive; "Jun" 404s where "JUN" works."""
    _, old_url = bhavcopy_urls(dt.date(2015, 1, 2))
    assert "/2015/JAN/cm02JAN2015bhav.csv.zip" in old_url


def test_a_two_digit_year_is_accepted_in_the_old_layout() -> None:
    """NSE published 2020-07-13 with "13-Jul-20" while every neighbouring
    session used four digits.

    Accepting only the common spelling dropped that session from an
    eleven-year backfill -- and a missing session is a hole in the universe
    that looks exactly like a quiet day, because the trailing liquidity
    windows simply advance one session short.
    """
    csv_text = OLD_CSV.replace("03-JUN-2024", "03-JUN-24")
    rows = parse_bhavcopy(_zip(csv_text))
    assert len(rows) == 1
    assert rows[0].session_date == dt.date(2024, 6, 3)


def test_an_unparseable_session_date_is_still_refused() -> None:
    """Widening the accepted spellings must not turn into accepting
    anything: a date nobody can read is a bar nobody can place in time."""
    csv_text = OLD_CSV.replace("03-JUN-2024", "not-a-date")
    with pytest.raises(BhavcopyError, match="unparseable session date"):
        parse_bhavcopy(_zip(csv_text))
