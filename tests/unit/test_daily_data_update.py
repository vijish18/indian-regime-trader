from __future__ import annotations

import datetime as dt

import pytest

from scripts.daily_data_update import parse_index_close

HEADER = (
    "Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,"
    "Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield\n"
)
FILE = HEADER + (
    "Nifty Next 50,25-09-2026,1,2,0.5,1.5,0,0,1,1,1,1,1\n"
    "Nifty 50,25-09-2026,23035,23162.7,23020.95,23140.5,77.4,.34,242720711,18949.25,"
    "19.56,2.8,1.22\n"
    "India VIX,25-09-2026,12.6875,12.835,11.77,12.16,-0.53,-4.16,-,-,-,-,-\n"
)
DAY = dt.date(2026, 9, 25)


def test_parses_nifty_and_vix_into_store_rows() -> None:
    rows = parse_index_close(FILE, DAY)
    assert rows["NIFTY50"] == [
        "NIFTY50",
        "2026-09-25",
        "23035",
        "23162.7",
        "23020.95",
        "23140.5",
        "",
    ]
    assert rows["INDIAVIX"] == ["INDIAVIX", "2026-09-25", "12.6875", "12.835", "11.77", "12.16", ""]
    assert set(rows) == {"NIFTY50", "INDIAVIX"}


def test_a_file_for_another_day_is_refused() -> None:
    with pytest.raises(ValueError, match="dated"):
        parse_index_close(FILE, DAY + dt.timedelta(days=1))


def test_a_missing_index_is_refused() -> None:
    with pytest.raises(ValueError, match="lacks"):
        parse_index_close(HEADER + FILE.splitlines(keepends=True)[2], DAY)


def test_a_non_numeric_close_is_refused() -> None:
    with pytest.raises(ValueError):
        parse_index_close(FILE.replace("23140.5", "-"), DAY)
