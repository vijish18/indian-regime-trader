"""Unit tests for ``validation/paper_feed.py`` (Phase 21):
``ReplayQuoteFeed`` synthesizes a quote from the latest ingested close and
delegates everything historical unchanged.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from data.errors import DataNotAvailableError
from tests.unit._wf_support import FakeMarketDataProvider
from validation.paper_feed import ReplayQuoteFeed

_INSTRUMENT = "NSE:INFY"


class _ClockBox:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now


@pytest.fixture
def inner() -> FakeMarketDataProvider:
    provider = FakeMarketDataProvider()
    dates = [dt.date(2024, 6, day) for day in range(1, 6)]
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    provider.add_equity(_INSTRUMENT, dates, closes)
    provider.add_index("NIFTY50", dates, [23000.0 + i for i in range(5)])
    return provider


def test_get_quote_uses_the_latest_close_at_or_before_the_session_date(
    inner: FakeMarketDataProvider,
) -> None:
    feed = ReplayQuoteFeed(inner, dt.date(2024, 6, 3))
    quote = feed.get_quote(_INSTRUMENT)
    assert quote.last_price == Decimal("102.0")


def test_get_quote_falls_back_to_the_most_recent_prior_close_on_a_gap(
    inner: FakeMarketDataProvider,
) -> None:
    feed = ReplayQuoteFeed(inner, dt.date(2024, 6, 4))
    feed.session_date = dt.date(2024, 6, 10)  # no bar on this exact date
    quote = feed.get_quote(_INSTRUMENT)
    assert quote.last_price == Decimal("104.0")  # the last bar within the lookback


def test_get_quote_raises_when_nothing_is_within_the_lookback_window(
    inner: FakeMarketDataProvider,
) -> None:
    feed = ReplayQuoteFeed(inner, dt.date(2025, 1, 1))
    with pytest.raises(DataNotAvailableError):
        feed.get_quote(_INSTRUMENT)


def test_bid_ask_straddle_the_close_by_the_configured_spread(
    inner: FakeMarketDataProvider,
) -> None:
    feed = ReplayQuoteFeed(inner, dt.date(2024, 6, 3), spread_bps=100.0)
    quote = feed.get_quote(_INSTRUMENT)
    assert quote.bid < quote.last_price < quote.ask
    half_spread = quote.last_price * Decimal("100") / Decimal("20000")
    assert float(quote.ask - quote.last_price) == pytest.approx(float(half_spread), abs=1e-6)


def test_zero_spread_collapses_bid_and_ask_to_the_close(inner: FakeMarketDataProvider) -> None:
    feed = ReplayQuoteFeed(inner, dt.date(2024, 6, 3), spread_bps=0.0)
    quote = feed.get_quote(_INSTRUMENT)
    assert quote.bid == quote.ask == quote.last_price


def test_negative_spread_is_refused() -> None:
    with pytest.raises(ValueError, match="spread_bps"):
        ReplayQuoteFeed(FakeMarketDataProvider(), dt.date(2024, 6, 3), spread_bps=-1.0)


def test_depth_is_reported_on_both_sides(inner: FakeMarketDataProvider) -> None:
    feed = ReplayQuoteFeed(inner, dt.date(2024, 6, 3), depth=12_345)
    quote = feed.get_quote(_INSTRUMENT)
    assert quote.bid_quantity == 12_345
    assert quote.ask_quantity == 12_345


def test_quote_is_stamped_with_the_clock_not_the_bar_date(inner: FakeMarketDataProvider) -> None:
    clock = _ClockBox(dt.datetime(2024, 6, 3, 15, 30, tzinfo=dt.UTC))
    feed = ReplayQuoteFeed(inner, dt.date(2024, 6, 3), clock=clock)
    quote = feed.get_quote(_INSTRUMENT)
    assert quote.as_of == clock.now


def test_advance_to_moves_the_session_date(inner: FakeMarketDataProvider) -> None:
    feed = ReplayQuoteFeed(inner, dt.date(2024, 6, 1))
    assert feed.get_quote(_INSTRUMENT).last_price == Decimal("100.0")
    feed.advance_to(dt.date(2024, 6, 5))
    assert feed.get_quote(_INSTRUMENT).last_price == Decimal("104.0")


def test_historical_methods_delegate_unchanged(inner: FakeMarketDataProvider) -> None:
    feed = ReplayQuoteFeed(inner, dt.date(2024, 6, 3))
    bars = feed.get_equity_bars(_INSTRUMENT, dt.date(2024, 6, 1), dt.date(2024, 6, 5))
    assert bars == inner.get_equity_bars(_INSTRUMENT, dt.date(2024, 6, 1), dt.date(2024, 6, 5))

    observations = feed.get_index_observations(
        "NIFTY50", dt.date(2024, 6, 1), dt.date(2024, 6, 5)
    )
    assert observations == inner.get_index_observations(
        "NIFTY50", dt.date(2024, 6, 1), dt.date(2024, 6, 5)
    )

    assert feed.available_range(_INSTRUMENT) == inner.available_range(_INSTRUMENT)


def test_a_genuinely_missing_instrument_still_raises_through_the_feed(
    inner: FakeMarketDataProvider,
) -> None:
    feed = ReplayQuoteFeed(inner, dt.date(2024, 6, 3))
    with pytest.raises(DataNotAvailableError):
        feed.get_quote("NSE:NOT-INGESTED")
