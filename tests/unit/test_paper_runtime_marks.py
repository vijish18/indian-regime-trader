from __future__ import annotations

import datetime as dt
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

from app.paper_runtime import PaperRuntime
from data.errors import DataNotAvailableError

DAY = dt.date(2026, 10, 6)


def _runtime(quotes: dict[str, float], closes: dict[str, float], held: list[str]) -> Any:
    def bars(instrument_id: str, start: dt.date, end: dt.date) -> list[Any]:
        if instrument_id == "NSE:GONE":
            raise DataNotAvailableError("no file")
        close = closes.get(instrument_id)
        return [] if close is None else [SimpleNamespace(close=Decimal(str(close)))]

    return SimpleNamespace(
        as_of=DAY,
        orchestrator=SimpleNamespace(
            position_tracker=SimpleNamespace(
                current_positions=lambda: [SimpleNamespace(instrument_id=i) for i in held]
            )
        ),
        market=SimpleNamespace(
            quotes={i: SimpleNamespace(last_price=Decimal(str(p))) for i, p in quotes.items()},
            get_equity_bars=bars,
        ),
    )


def test_held_names_are_valued_at_the_live_quote_else_the_close() -> None:
    runtime = _runtime(
        quotes={"NSE:A": 101.5, "NSE:NOTHELD": 5.0},
        closes={"NSE:A": 99.0, "NSE:B": 250.0},
        held=["NSE:A", "NSE:B", "NSE:C", "NSE:GONE"],
    )
    assert PaperRuntime.marks(runtime) == {"NSE:A": 101.5, "NSE:B": 250.0}


def test_live_book_carries_rank_returns_orders_and_unbought_names() -> None:
    from app.paper_market import IST
    from execution.order_manager import OrderState

    now = dt.datetime.now(IST)
    position = SimpleNamespace(
        instrument_id="NSE:B",
        quantity=10,
        avg_price=100.0,
        current_price=103.0,
        unrealized_pnl=30.0,
    )
    order = SimpleNamespace(
        instrument_id="NSE:B",
        side="buy",
        quantity=10,
        filled_quantity=10,
        limit_price=100.0,
        avg_fill_price=100.0,
        state=OrderState.FILLED,
        created_at=now,
        reject_reason=None,
    )
    old_resting = SimpleNamespace(
        **{
            **vars(order),
            "instrument_id": "NSE:C",
            "filled_quantity": 2,
            "state": OrderState.PARTIALLY_FILLED,
            "created_at": now - dt.timedelta(days=3),
        }
    )
    old_done = SimpleNamespace(
        **{**vars(order), "instrument_id": "NSE:D", "created_at": now - dt.timedelta(days=3)}
    )
    runtime = SimpleNamespace(
        candidates=[SimpleNamespace(instrument_id="NSE:A"), SimpleNamespace(instrument_id="NSE:B")],
        settings=SimpleNamespace(risk=SimpleNamespace(stop_loss=SimpleNamespace(enabled=False))),
        budget=2000.0,
        last_quote_at="t",
        market=SimpleNamespace(
            rows={"NSE:B": {"day_pct": 0.01}},
            unavailable={"NSE:A": "NSE:A: missing or crossed bid/ask"},
        ),
        orchestrator=SimpleNamespace(
            order_manager=SimpleNamespace(all_orders=lambda: [old_done, old_resting, order])
        ),
    )
    book = PaperRuntime.live_book(runtime, [position], 970.0, 2000.0, 1030.0)
    (row,) = book["positions"]
    assert row["rank"] == 2 and row["pnl_pct"] == 0.03 and row["stop"] == {"available": False}
    assert book["pnl_pct"] == 0.03 and book["winners"] == 1 and book["losers"] == 0
    assert [o["symbol"] for o in book["orders"]] == ["C", "B"]  # today's, plus still working
    assert book["unquoted"] == [{"symbol": "A", "reason": "NSE:A: missing or crossed bid/ask"}]
