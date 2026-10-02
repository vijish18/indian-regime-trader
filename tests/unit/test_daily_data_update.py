from __future__ import annotations

import datetime as dt

import pytest

from scripts.daily_data_update import merge_recent_actions, parse_index_close

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


def _act(iid: str, kind: str, ex: str) -> dict[str, str]:
    return {"instrument_id": iid, "action_type": kind, "ex_date": ex, "ratio_new": ""}


def test_merge_keeps_history_and_appends_only_new_recent_events() -> None:
    existing = [_act("NSE:HEG", "dividend", "2015-09-14"), _act("NSE:A", "bonus", "2026-09-10")]
    fresh = [
        _act("NSE:HEGAM", "dividend", "2015-09-14"),  # old, re-filed: ignored
        _act("NSE:A", "bonus", "2026-09-10"),  # already present
        _act("NSE:B", "split", "2026-09-28"),  # new and recent: added
        _act("NSE:C", "dividend", "2026-06-01"),  # new but older than the window
    ]
    merged, added = merge_recent_actions(existing, fresh, dt.date(2026, 8, 3))
    assert added == 1
    assert merged == existing + [_act("NSE:B", "split", "2026-09-28")]
