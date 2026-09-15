"""A quote feed for paper mode (Phase 21).

``data.market_data.LocalMarketDataProvider`` serves historical bars and
deliberately refuses to serve quotes -- a backtest must never be able to
consult a live book. But ``broker.adapters.paper_broker.PaperBroker``
genuinely needs quotes: it prices fills from a real bid/ask and rejects
stale or crossed markets, which is most of what makes paper trading a
meaningful rehearsal rather than a backtest with extra steps.

``ReplayQuoteFeed`` closes that gap in the only honest way available
without a live feed: it synthesizes a quote from the most recent ingested
close at or before the session date, stamped with the current clock. That
is precisely what "paper trading against historical data" means, and its
limits should be stated plainly rather than discovered later:

- the spread is a configured constant, not a measured one. This dataset
  has no bid/ask history (the same gap ``backtest/engine.py`` documents
  for its own assumed spread), so spread-sensitive conclusions from a
  paper run are not evidence about real spreads.
- there is no intraday path. Every quote within a session is the same
  close, so a paper session cannot say anything about intraday timing.
- depth is a configured constant, so partial fills happen where the
  configured depth says they do, not where the real book would have.

What it *does* exercise honestly is every path that depends on a quote
existing, being fresh, and being consistent with the order being priced:
the price guard, the staleness check, partial fills against displayed
depth, and the whole downstream accounting chain.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from decimal import Decimal

from data.errors import DataNotAvailableError
from data.interfaces import MarketDataProvider
from data.models import DailyBar, IndexObservation, PriceBasis, Quote

_LOOKBACK_DAYS = 30
"""How far back to search for a close when the session date itself has no
bar (a holiday, or an instrument that stopped trading). Bounded so a
missing series fails fast instead of scanning the whole history."""


class ReplayQuoteFeed(MarketDataProvider):
    """Wraps a historical provider, delegating everything except
    :meth:`get_quote`, which it synthesizes from the latest close.
    """

    def __init__(
        self,
        inner: MarketDataProvider,
        session_date: dt.date,
        *,
        spread_bps: float = 10.0,
        depth: int = 1_000_000,
        clock: Callable[[], dt.datetime] | None = None,
    ) -> None:
        if spread_bps < 0:
            raise ValueError(f"spread_bps must be >= 0, got {spread_bps}")
        self._inner = inner
        self.session_date = session_date
        self.spread_bps = spread_bps
        self.depth = depth
        self._clock: Callable[[], dt.datetime] = clock or (lambda: dt.datetime.now(dt.UTC))

    def advance_to(self, session_date: dt.date) -> None:
        """Move the feed to a new session. A paper session that runs more
        than one day moves the feed with it, so quotes never come from a
        date the rest of the system has not reached."""
        self.session_date = session_date

    # -- the synthesized part ------------------------------------------

    def get_quote(self, instrument_id: str) -> Quote:
        close = self._latest_close(instrument_id)
        half_spread = close * Decimal(str(self.spread_bps)) / Decimal("20000")
        return Quote(
            instrument_id=instrument_id,
            bid=close - half_spread,
            ask=close + half_spread,
            last_price=close,
            as_of=self._clock(),
            bid_quantity=self.depth,
            ask_quantity=self.depth,
        )

    def _latest_close(self, instrument_id: str) -> Decimal:
        start = self.session_date - dt.timedelta(days=_LOOKBACK_DAYS)
        bars = self._inner.get_equity_bars(instrument_id, start, self.session_date)
        if not bars:
            raise DataNotAvailableError(
                f"no bar for {instrument_id} within {_LOOKBACK_DAYS} days of "
                f"{self.session_date}; cannot synthesize a paper quote"
            )
        return bars[-1].close

    # -- everything else is the real provider --------------------------

    def get_equity_bars(
        self,
        instrument_id: str,
        start: dt.date,
        end: dt.date,
        price_basis: PriceBasis = PriceBasis.RAW,
    ) -> list[DailyBar]:
        return self._inner.get_equity_bars(instrument_id, start, end, price_basis)

    def get_index_observations(
        self, index_symbol: str, start: dt.date, end: dt.date
    ) -> list[IndexObservation]:
        return self._inner.get_index_observations(index_symbol, start, end)

    def available_range(self, instrument_id: str) -> tuple[dt.date, dt.date] | None:
        return self._inner.available_range(instrument_id)
