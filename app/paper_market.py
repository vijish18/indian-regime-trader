"""Historical data plus timestamped, read-only Kite quotes for the paper broker."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from data.errors import DataNotAvailableError
from data.interfaces import MarketDataProvider
from data.models import DailyBar, IndexObservation, PriceBasis, Quote
from scripts._dashboard_live import live_quotes

IST = ZoneInfo("Asia/Kolkata")


def parse_quote(instrument_id: str, row: dict[str, Any], now: dt.datetime) -> Quote:
    timestamp = dt.datetime.fromisoformat(row["exchange_timestamp"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=IST)
    age = (now - timestamp).total_seconds()
    if age < -5 or age > 30:
        raise DataNotAvailableError(f"{instrument_id}: quote timestamp is stale or in the future")
    depth = row["depth"]
    bid, ask = depth["buy"][0], depth["sell"][0]
    quote = Quote(
        instrument_id,
        Decimal(str(bid["price"])),
        Decimal(str(ask["price"])),
        Decimal(str(row["last"])),
        timestamp,
        int(bid["quantity"]),
        int(ask["quantity"]),
    )
    if quote.bid <= 0 or quote.ask <= 0 or quote.is_crossed():
        raise DataNotAvailableError(f"{instrument_id}: missing or crossed bid/ask")
    return quote


REFETCH_AFTER_SECONDS = 5.0
"""A quote fetched longer ago than this is fetched again when asked for. The
broker refuses quotes older than ``execution.stale_quote_seconds`` (15 s), and
a daily cycle can take longer than that between the first fetch and its last
order, so serving the first fetch would get every late order rejected."""


class PaperMarket(MarketDataProvider):
    def __init__(self, history: MarketDataProvider, session_path: Path) -> None:
        self.history = history
        self.session_path = session_path
        self.quotes: dict[str, Quote] = {}
        self.rows: dict[str, Any] = {}
        self.fetched_at: dict[str, dt.datetime] = {}
        self.unavailable: dict[str, str] = {}
        """Instruments the last fetch returned no usable quote for, and why."""

    def refresh(self, instrument_ids: list[str], now: dt.datetime, *, replace: bool = True) -> None:
        """Fetch ``instrument_ids``. ``replace`` drops every other quote first
        (the per-tick refresh); ``replace=False`` updates just these.

        One instrument without a usable quote -- locked at a price band with
        one side of the book empty, suspended, a stale tick -- is recorded in
        ``unavailable`` and left unquoted; it does not stop the others. Only a
        failed fetch as a whole raises. ``get_quote`` raises for that one name.
        """
        if replace:
            self.quotes, self.rows, self.fetched_at, self.unavailable = {}, {}, {}, {}
        payload = live_quotes(instrument_ids, self.session_path)
        if not payload.get("available"):
            raise DataNotAvailableError(str(payload.get("reason", "No quotes")))
        self.rows.update(payload["quotes"])
        for instrument_id in instrument_ids:
            self.fetched_at[instrument_id] = now
            try:
                self.quotes[instrument_id] = parse_quote(
                    instrument_id, self.rows[instrument_id], now
                )
            except DataNotAvailableError as exc:
                reason = str(exc)
            except (KeyError, TypeError, ValueError, IndexError):
                reason = f"Incomplete quote for {instrument_id}"
            else:
                self.unavailable.pop(instrument_id, None)
                continue
            self.quotes.pop(instrument_id, None)
            self.unavailable[instrument_id] = reason

    def get_quote(self, instrument_id: str) -> Quote:
        now = dt.datetime.now(dt.UTC)
        fetched = self.fetched_at.get(instrument_id)
        if fetched is None or (now - fetched).total_seconds() > REFETCH_AFTER_SECONDS:
            self.refresh([instrument_id], now, replace=False)
        if instrument_id not in self.quotes:
            raise DataNotAvailableError(
                self.unavailable.get(instrument_id, f"{instrument_id}: no quote")
            )
        return self.quotes[instrument_id]

    def get_equity_bars(
        self,
        instrument_id: str,
        start: dt.date,
        end: dt.date,
        price_basis: PriceBasis = PriceBasis.RAW,
    ) -> list[DailyBar]:
        return self.history.get_equity_bars(instrument_id, start, end, price_basis)

    def get_index_observations(
        self, index_symbol: str, start: dt.date, end: dt.date
    ) -> list[IndexObservation]:
        return self.history.get_index_observations(index_symbol, start, end)

    def available_range(self, instrument_id: str) -> tuple[dt.date, dt.date] | None:
        return self.history.available_range(instrument_id)
