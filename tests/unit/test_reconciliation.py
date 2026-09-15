"""Unit tests for ``execution/reconciliation.py`` (Phase 18):
``ReconciliationEngine``'s position-level and order-level comparisons,
against a fully controllable stub broker.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping

import pytest

from backtest.costs import TradeSide
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
from execution.order_manager import OrderManager, OrderState
from execution.position_tracker import PositionTracker
from execution.reconciliation import ReconciliationEngine, ReconciliationStatus

_T0 = dt.datetime(2024, 6, 3, 9, 30, tzinfo=dt.UTC)


class _ClockBox:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now


class _StubBroker(Broker):
    def __init__(self) -> None:
        self.positions_response: list[BrokerPosition] = []
        self.open_orders_response: list[BrokerOrder] = []
        self.orders: dict[str, BrokerOrder] = {}

    def capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        raise NotImplementedError

    def get_account(self) -> BrokerAccount:
        raise NotImplementedError

    def get_positions(self) -> list[BrokerPosition]:
        return list(self.positions_response)

    def get_open_orders(self) -> list[BrokerOrder]:
        return list(self.open_orders_response)

    def get_order(self, order_id: str) -> BrokerOrder:
        try:
            return self.orders[order_id]
        except KeyError:
            raise RuntimeError(f"unknown order: {order_id}") from None

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        raise NotImplementedError

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


@pytest.fixture
def clock() -> _ClockBox:
    return _ClockBox(_T0)


@pytest.fixture
def position_tracker() -> PositionTracker:
    return PositionTracker()


@pytest.fixture
def order_manager(clock: _ClockBox) -> OrderManager:
    return OrderManager(clock=clock)


@pytest.fixture
def broker() -> _StubBroker:
    return _StubBroker()


@pytest.fixture
def engine(
    position_tracker: PositionTracker,
    order_manager: OrderManager,
    broker: _StubBroker,
    clock: _ClockBox,
) -> ReconciliationEngine:
    return ReconciliationEngine(position_tracker, order_manager, broker, clock=clock)


# --------------------------------------------------------------------------
# reconcile_positions()
# --------------------------------------------------------------------------


def test_reconcile_positions_is_clean_when_both_sides_agree(
    engine: ReconciliationEngine, position_tracker: PositionTracker, broker: _StubBroker
) -> None:
    position_tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    broker.positions_response = [
        BrokerPosition(instrument_id="NSE:INFY", quantity=10, avg_price=1500.0)
    ]
    report = engine.reconcile_positions()
    assert report.status is ReconciliationStatus.CLEAN
    assert report.mismatches == ()


def test_reconcile_positions_is_clean_when_both_sides_hold_nothing(
    engine: ReconciliationEngine,
) -> None:
    report = engine.reconcile_positions()
    assert report.status is ReconciliationStatus.CLEAN


def test_reconcile_positions_flags_a_quantity_mismatch(
    engine: ReconciliationEngine, position_tracker: PositionTracker, broker: _StubBroker
) -> None:
    position_tracker.apply_fill("NSE:INFY", 8, 1500.0, TradeSide.BUY, _T0)
    broker.positions_response = [
        BrokerPosition(instrument_id="NSE:INFY", quantity=10, avg_price=1500.0)
    ]
    report = engine.reconcile_positions()
    assert report.status is ReconciliationStatus.QUARANTINED
    assert len(report.mismatches) == 1
    mismatch = report.mismatches[0]
    assert mismatch.instrument_id == "NSE:INFY"
    assert mismatch.local_quantity == 8
    assert mismatch.broker_quantity == 10
    assert "mismatch" in mismatch.detail


def test_reconcile_positions_flags_a_broker_position_with_no_local_record(
    engine: ReconciliationEngine, broker: _StubBroker
) -> None:
    """"Missing local record": the broker reports a position this
    system's local state has never seen at all."""
    broker.positions_response = [
        BrokerPosition(instrument_id="NSE:TCS", quantity=5, avg_price=3500.0)
    ]
    report = engine.reconcile_positions()
    assert report.status is ReconciliationStatus.QUARANTINED
    mismatch = report.mismatches[0]
    assert mismatch.local_quantity == 0
    assert mismatch.broker_quantity == 5
    assert "no local record" in mismatch.detail


def test_reconcile_positions_flags_a_local_position_the_broker_no_longer_reports(
    engine: ReconciliationEngine, position_tracker: PositionTracker
) -> None:
    position_tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    report = engine.reconcile_positions()
    assert report.status is ReconciliationStatus.QUARANTINED
    mismatch = report.mismatches[0]
    assert mismatch.local_quantity == 10
    assert mismatch.broker_quantity == 0
    assert "no longer reports" in mismatch.detail


def test_reconcile_positions_updates_quarantined_instruments(
    engine: ReconciliationEngine, broker: _StubBroker
) -> None:
    assert engine.quarantined_instruments() == set()
    broker.positions_response = [
        BrokerPosition(instrument_id="NSE:TCS", quantity=5, avg_price=3500.0)
    ]
    engine.reconcile_positions()
    assert engine.quarantined_instruments() == {"NSE:TCS"}


def test_reconcile_positions_clears_quarantine_once_resolved(
    engine: ReconciliationEngine, position_tracker: PositionTracker, broker: _StubBroker
) -> None:
    broker.positions_response = [
        BrokerPosition(instrument_id="NSE:TCS", quantity=5, avg_price=3500.0)
    ]
    engine.reconcile_positions()
    assert engine.quarantined_instruments() == {"NSE:TCS"}

    position_tracker.apply_fill("NSE:TCS", 5, 3500.0, TradeSide.BUY, _T0)
    engine.reconcile_positions()
    assert engine.quarantined_instruments() == set()


def test_reconcile_positions_multiple_instruments_reports_only_the_mismatched_ones(
    engine: ReconciliationEngine, position_tracker: PositionTracker, broker: _StubBroker
) -> None:
    position_tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    position_tracker.apply_fill("NSE:TCS", 5, 3500.0, TradeSide.BUY, _T0)
    broker.positions_response = [
        BrokerPosition(instrument_id="NSE:INFY", quantity=10, avg_price=1500.0),
        BrokerPosition(instrument_id="NSE:TCS", quantity=4, avg_price=3500.0),  # mismatched
    ]
    report = engine.reconcile_positions()
    assert [m.instrument_id for m in report.mismatches] == ["NSE:TCS"]


# --------------------------------------------------------------------------
# reconcile_open_orders() -- delegates to OrderReconciler
# --------------------------------------------------------------------------


def test_reconcile_open_orders_is_clean_with_nothing_outstanding(
    engine: ReconciliationEngine,
) -> None:
    report = engine.reconcile_open_orders()
    assert report.status is ReconciliationStatus.CLEAN
    assert report.mismatches == ()


def test_reconcile_open_orders_resolves_an_unknown_order_without_flagging_it(
    engine: ReconciliationEngine, order_manager: OrderManager, broker: _StubBroker
) -> None:
    """An UNKNOWN order OrderReconciler can resolve safely by querying
    the broker is a resolution, not a discrepancy -- it must not appear
    in the mismatches this engine reports."""
    result = order_manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0,
        idempotency_key="req-1", signal_id="sig-1", risk_decision_id="rd-1",
    )
    client_order_id = result.order.client_order_id
    order_manager.transition(client_order_id, OrderState.SUBMITTED)
    order_manager.transition(client_order_id, OrderState.UNKNOWN)
    broker.orders[client_order_id] = BrokerOrder(
        client_order_id=client_order_id,
        broker_order_id="B-1",
        instrument_id="NSE:INFY",
        side="buy",
        quantity=10,
        order_type="limit",
        limit_price=1500.0,
        status=OrderState.OPEN.value,
    )

    report = engine.reconcile_open_orders()
    assert report.status is ReconciliationStatus.CLEAN
    assert order_manager.get(client_order_id).state is OrderState.OPEN


def test_reconcile_open_orders_flags_an_orphaned_broker_order(
    engine: ReconciliationEngine, broker: _StubBroker
) -> None:
    """"Unknown local order" from the broker's perspective: an open
    order the broker reports that this system's OrderManager has never
    heard of at all."""
    broker.open_orders_response = [
        BrokerOrder(
            client_order_id="not-ours",
            broker_order_id="B-ORPHAN",
            instrument_id="NSE:INFY",
            side="buy",
            quantity=7,
            order_type="limit",
            limit_price=1500.0,
            status=OrderState.OPEN.value,
        )
    ]
    report = engine.reconcile_open_orders()
    assert report.status is ReconciliationStatus.QUARANTINED
    assert len(report.mismatches) == 1
    mismatch = report.mismatches[0]
    assert mismatch.broker_quantity == 7
    assert "no local record" in mismatch.detail
