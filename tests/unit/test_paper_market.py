from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from app import paper_market
from app.paper_market import IST, PaperMarket
from app.paper_runtime import _SignalDaySelector


def _row(price: float, when: dt.datetime) -> dict[str, Any]:
    return {
        "exchange_timestamp": when.astimezone(IST).replace(tzinfo=None).isoformat(),
        "last": price,
        "depth": {
            "buy": [{"price": price - 0.05, "quantity": 100}],
            "sell": [{"price": price + 0.05, "quantity": 100}],
        },
    }


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    seen: list[list[str]] = []

    def fake_live_quotes(ids: list[str], session_path: Path) -> dict[str, Any]:
        seen.append(list(ids))
        now = dt.datetime.now(dt.UTC)
        return {"available": True, "quotes": {i: _row(100.0 + len(seen), now) for i in ids}}

    monkeypatch.setattr(paper_market, "live_quotes", fake_live_quotes)
    return seen


def test_a_fresh_quote_is_served_without_another_fetch(calls: list[list[str]]) -> None:
    market = PaperMarket(history=None, session_path=Path("x"))  # type: ignore[arg-type]
    market.refresh(["NSE:A", "NSE:B"], dt.datetime.now(dt.UTC))
    market.get_quote("NSE:A")
    assert calls == [["NSE:A", "NSE:B"]]


def test_an_old_fetch_is_refetched_for_that_instrument_only(calls: list[list[str]]) -> None:
    market = PaperMarket(history=None, session_path=Path("x"))  # type: ignore[arg-type]
    market.refresh(["NSE:A", "NSE:B"], dt.datetime.now(dt.UTC))
    market.fetched_at["NSE:A"] -= dt.timedelta(seconds=30)
    quote = market.get_quote("NSE:A")
    assert calls[-1] == ["NSE:A"]
    assert float(quote.last_price) == 102.0
    assert "NSE:B" in market.quotes  # the targeted refetch keeps the others


def test_an_unfetched_instrument_is_fetched_on_demand(calls: list[list[str]]) -> None:
    market = PaperMarket(history=None, session_path=Path("x"))  # type: ignore[arg-type]
    market.get_quote("NSE:C")
    assert calls == [["NSE:C"]]


class _Inner:
    def __init__(self) -> None:
        self.calls: list[dt.date] = []
        self.universe_provider = "universe"

    def select(self, as_of: dt.date) -> list[str]:
        self.calls.append(as_of)
        return ["other-day"]


def test_the_signal_day_ranking_is_reused_and_other_days_pass_through() -> None:
    inner = _Inner()
    day = dt.date(2026, 10, 1)
    selector = _SignalDaySelector(inner, day, ["CUPID", "SBC"])
    assert selector.select(day) == ["CUPID", "SBC"]
    assert inner.calls == []
    assert selector.select(day - dt.timedelta(days=1)) == ["other-day"]
    assert selector.universe_provider == "universe"


def test_one_unquotable_name_does_not_stop_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_live_quotes(ids: list[str], session_path: Path) -> dict[str, Any]:
        now = dt.datetime.now(dt.UTC)
        rows = {i: _row(100.0, now) for i in ids}
        if "NSE:LOCKED" in rows:  # upper circuit: buyers only, no sellers
            rows["NSE:LOCKED"]["depth"]["sell"] = [{"price": 0, "quantity": 0}]
        return {"available": True, "quotes": rows}

    monkeypatch.setattr(paper_market, "live_quotes", fake_live_quotes)
    market = PaperMarket(history=None, session_path=Path("x"))  # type: ignore[arg-type]
    market.refresh(["NSE:A", "NSE:LOCKED"], dt.datetime.now(dt.UTC))
    assert set(market.quotes) == {"NSE:A"}
    assert "crossed" in market.unavailable["NSE:LOCKED"]
    with pytest.raises(paper_market.DataNotAvailableError):
        market.get_quote("NSE:LOCKED")
    assert float(market.get_quote("NSE:A").last_price) == 100.0


def test_a_failed_fetch_as_a_whole_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        paper_market, "live_quotes", lambda ids, path: {"available": False, "reason": "down"}
    )
    market = PaperMarket(history=None, session_path=Path("x"))  # type: ignore[arg-type]
    with pytest.raises(paper_market.DataNotAvailableError, match="down"):
        market.refresh(["NSE:A"], dt.datetime.now(dt.UTC))
