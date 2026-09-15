"""Unit tests for ``orchestration/fill_tracker.py`` (Phase 19): fills are
deduplicated by ``trade_id`` and applied to the canonical ``PositionTracker``
exactly once, no matter how many times ``poll`` sees the same broker fill.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping

import pytest

from broker.base import (
    Broker,
    BrokerAccount,
    BrokerCapabilities,
    BrokerFill,
    BrokerOrder,
    BrokerPosition,
    BrokerQuote,
    HealthStatus,
)
from execution.position_tracker import PositionTracker
from orchestration.fill_tracker import FillTracker

_T0 = dt.datetime(2024, 6, 3, 9, 30, tzinfo=dt.UTC)


class _StubBroker(Broker):
    def __init__(self) -> None:
        self.trades: list[BrokerFill] = []

    def capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        raise NotImplementedError

    def get_account(self) -> BrokerAccount:
        raise NotImplementedError

    def get_positions(self) -> list[BrokerPosition]:
        raise NotImplementedError

    def get_open_orders(self) -> list[BrokerOrder]:
        raise NotImplementedError

    def get_order(self, order_id: str) -> BrokerOrder:
        raise NotImplementedError

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        return list(self.trades)

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        raise NotImplementedError

    def subscribe_market_data(
        self, instrument_ids: list[str], on_tick: Callable[[BrokerQuote], None]
    ) -> Callable[[], None]:
        raise NotImplementedError

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        raise NotImplementedError

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> BrokerOrder:
        raise NotImplementedError

    def close_position(self, instrument_id: str) -> BrokerOrder:
        raise NotImplementedError

    def close_all_positions(self) -> list[BrokerOrder]:
        raise NotImplementedError

    def health_check(self) -> HealthStatus:
        raise NotImplementedError


def _fill(trade_id: str, instrument_id: str = "NSE:INFY", side: str = "buy") -> BrokerFill:
    return BrokerFill(
        trade_id=trade_id,
        order_id="c-1",
        instrument_id=instrument_id,
        side=side,
        quantity=10,
        price=1500.0,
        product="CNC",
        as_of=_T0,
    )


@pytest.fixture
def broker() -> _StubBroker:
    return _StubBroker()


@pytest.fixture
def position_tracker() -> PositionTracker:
    return PositionTracker()


@pytest.fixture
def tracker(position_tracker: PositionTracker) -> FillTracker:
    return FillTracker(position_tracker)


def test_poll_applies_a_new_fill(
    tracker: FillTracker, broker: _StubBroker, position_tracker: PositionTracker
) -> None:
    broker.trades = [_fill("T-1")]
    new_fills = tracker.poll(broker)
    assert [f.trade_id for f in new_fills] == ["T-1"]
    assert position_tracker.held_quantity("NSE:INFY") == 10


def test_poll_never_applies_the_same_fill_twice(
    tracker: FillTracker, broker: _StubBroker, position_tracker: PositionTracker
) -> None:
    broker.trades = [_fill("T-1")]
    tracker.poll(broker)
    second = tracker.poll(broker)
    assert second == []
    assert position_tracker.held_quantity("NSE:INFY") == 10


def test_poll_applies_only_the_new_fill_when_broker_reports_more(
    tracker: FillTracker, broker: _StubBroker, position_tracker: PositionTracker
) -> None:
    broker.trades = [_fill("T-1")]
    tracker.poll(broker)
    broker.trades = [_fill("T-1"), _fill("T-2")]
    new_fills = tracker.poll(broker)
    assert [f.trade_id for f in new_fills] == ["T-2"]
    assert position_tracker.held_quantity("NSE:INFY") == 20


def test_sell_fill_reduces_the_position(
    tracker: FillTracker, broker: _StubBroker, position_tracker: PositionTracker
) -> None:
    broker.trades = [_fill("T-1", side="buy")]
    tracker.poll(broker)
    broker.trades = [_fill("T-1", side="buy"), _fill("T-2", side="sell")]
    tracker.poll(broker)
    assert position_tracker.held_quantity("NSE:INFY") == 0


def test_mark_seen_prevents_a_pre_existing_fill_from_being_re_applied(
    broker: _StubBroker, position_tracker: PositionTracker
) -> None:
    tracker = FillTracker(position_tracker)
    tracker.mark_seen({"T-1"})
    broker.trades = [_fill("T-1")]
    new_fills = tracker.poll(broker)
    assert new_fills == []
    assert position_tracker.held_quantity("NSE:INFY") == 0


def test_seen_fill_ids_reflects_applied_fills(tracker: FillTracker, broker: _StubBroker) -> None:
    broker.trades = [_fill("T-1")]
    tracker.poll(broker)
    assert tracker.seen_fill_ids() == {"T-1"}


def test_a_fill_redelivered_twice_in_one_poll_is_applied_only_once(
    tracker: FillTracker, broker: _StubBroker, position_tracker: PositionTracker
) -> None:
    """Regression for a real bug Phase 21's failure injection found: a
    brand-new trade_id appearing twice in a single ``get_trades()``
    response (the same shape a redelivered broker event has) must not
    slip past a membership check computed once against
    ``_applied_fill_ids`` before either copy has been recorded.
    """
    new_fill = _fill("T-1")
    broker.trades = [new_fill, new_fill]
    new_fills = tracker.poll(broker)
    assert [f.trade_id for f in new_fills] == ["T-1"]
    assert position_tracker.held_quantity("NSE:INFY") == 10
