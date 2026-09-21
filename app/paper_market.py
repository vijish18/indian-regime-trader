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


class PaperMarket(MarketDataProvider):
    def __init__(self, history: MarketDataProvider, session_path: Path) -> None:
        self.history = history
        self.session_path = session_path
        self.quotes: dict[str, Quote] = {}
        self.rows: dict[str, Any] = {}

    def refresh(self, instrument_ids: list[str], now: dt.datetime) -> None:
        self.quotes = {}
        payload = live_quotes(instrument_ids, self.session_path)
        if not payload.get("available"):
            raise DataNotAvailableError(str(payload.get("reason", "No quotes")))
        self.rows = payload["quotes"]
        for instrument_id in instrument_ids:
            try:
                self.quotes[instrument_id] = parse_quote(
                    instrument_id, self.rows[instrument_id], now
                )
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise DataNotAvailableError(f"Incomplete quote for {instrument_id}") from exc

    def get_quote(self, instrument_id: str) -> Quote:
        try:
            return self.quotes[instrument_id]
        except KeyError:
            raise DataNotAvailableError(f"No fresh quote for {instrument_id}") from None

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
