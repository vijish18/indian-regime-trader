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
