"""Unit tests for the order state machine and idempotent creation
(``execution/order_manager.py``, Phase 14) -- isolated from any real broker
via a small controllable stub, so these tests are about the state machine
itself, not about paper-fill simulation (see test_paper_broker.py for
that).
"""

from __future__ import annotations

import datetime as dt

import pytest

from broker.base import Account, Broker, BrokerOrder, BrokerPosition, BrokerQuote, HealthStatus
from execution.order_manager import (
    OrderManager,
    OrderState,
    OrderStateError,
)


class _ClockBox:
    """A mutable, injectable clock -- lets a test advance "now" without a
    real sleep."""

    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now


class _StubBroker(Broker):
    """A minimal, fully controllable ``Broker`` double for testing
    ``OrderManager`` in isolation. ``next_place_order_result`` and
    ``next_place_order_error`` script exactly one call's outcome each.
    """

    def __init__(self) -> None:
        self.next_place_order_result: BrokerOrder | None = None
        self.next_place_order_error: Exception | None = None
        self.orders: dict[str, BrokerOrder] = {}
        self.placed: list[BrokerOrder] = []

    def get_account(self) -> Account:
        raise NotImplementedError

    def get_positions(self) -> list[BrokerPosition]:
        raise NotImplementedError

    def get_open_orders(self) -> list[BrokerOrder]:
        return list(self.orders.values())

    def get_order(self, order_id: str) -> BrokerOrder:
        try:
            return self.orders[order_id]
        except KeyError:
            raise RuntimeError(f"unknown order: {order_id}") from None

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        raise NotImplementedError

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        self.placed.append(order)
        if self.next_place_order_error is not None:
            error = self.next_place_order_error
            self.next_place_order_error = None
            raise error
        result = self.next_place_order_result if self.next_place_order_result is not None else order
        self.orders[order.client_order_id] = result
        return result

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
    return _ClockBox(dt.datetime(2024, 6, 3, 9, 30, tzinfo=dt.UTC))


@pytest.fixture
def manager(clock: _ClockBox) -> OrderManager:
    return OrderManager(clock=clock)


# --------------------------------------------------------------------------
# create() / idempotency
# --------------------------------------------------------------------------


def test_create_generates_a_unique_client_order_id(manager: OrderManager) -> None:
    first = manager.create("NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1")
    second = manager.create("NSE:TCS", "buy", 5, "limit", 3500.0, idempotency_key="req-2")
    assert first.order.client_order_id != second.order.client_order_id
    assert first.was_duplicate is False
    assert second.was_duplicate is False


def test_create_is_idempotent_by_key(manager: OrderManager) -> None:
    first = manager.create("NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1")
    replay = manager.create("NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1")
    assert replay.was_duplicate is True
    assert replay.order == first.order
    assert replay.order.client_order_id == first.order.client_order_id


def test_create_with_a_different_idempotency_key_is_a_separate_order(manager: OrderManager) -> None:
    first = manager.create("NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1")
    second = manager.create("NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-2")
    assert first.order.client_order_id != second.order.client_order_id


def test_create_result_starts_in_created_state(manager: OrderManager) -> None:
    result = manager.create("NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1")
    assert result.order.state is OrderState.CREATED
    assert result.order.filled_quantity == 0
    assert result.order.remaining_quantity() == 10


# --------------------------------------------------------------------------
# transition() -- the state machine
# --------------------------------------------------------------------------


def test_valid_transition_sequence(manager: OrderManager) -> None:
    order_id = manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1"
    ).order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    manager.transition(order_id, OrderState.OPEN)
    manager.transition(order_id, OrderState.PARTIALLY_FILLED, filled_quantity=4)
    record = manager.transition(order_id, OrderState.FILLED, filled_quantity=10)
    assert record.state is OrderState.FILLED
    assert record.filled_quantity == 10
    assert record.is_terminal()


def test_invalid_transition_is_rejected(manager: OrderManager) -> None:
    order_id = manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1"
    ).order.client_order_id
    with pytest.raises(OrderStateError, match="cannot transition"):
        manager.transition(order_id, OrderState.FILLED)


def test_terminal_state_accepts_no_further_transition(manager: OrderManager) -> None:
    order_id = manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1"
    ).order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    manager.transition(order_id, OrderState.REJECTED, reject_reason="no liquidity")
    with pytest.raises(OrderStateError):
        manager.transition(order_id, OrderState.OPEN)


def test_rejected_transition_requires_a_reason(manager: OrderManager) -> None:
    order_id = manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1"
    ).order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    with pytest.raises(OrderStateError, match="reject_reason"):
        manager.transition(order_id, OrderState.REJECTED)


def test_cancel_requested_can_still_resolve_to_filled(manager: OrderManager) -> None:
    """A cancel request can race a fill that was already in flight
    broker-side -- the state machine must not forbid that outcome."""
    order_id = manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1"
    ).order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    manager.transition(order_id, OrderState.OPEN)
    manager.transition(order_id, OrderState.CANCEL_REQUESTED)
    record = manager.transition(order_id, OrderState.FILLED, filled_quantity=10)
    assert record.state is OrderState.FILLED


def test_get_unknown_client_order_id_raises(manager: OrderManager) -> None:
    with pytest.raises(OrderStateError, match="unknown client_order_id"):
        manager.get("does-not-exist")


def test_open_orders_lists_only_non_terminal_orders(manager: OrderManager) -> None:
    open_id = manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-open"
    ).order.client_order_id
    filled_id = manager.create(
        "NSE:TCS", "buy", 5, "limit", 3500.0, idempotency_key="req-filled"
    ).order.client_order_id
    manager.transition(open_id, OrderState.SUBMITTED)
    manager.transition(open_id, OrderState.OPEN)
    manager.transition(filled_id, OrderState.SUBMITTED)
    manager.transition(filled_id, OrderState.FILLED, filled_quantity=5)

    open_ids = {order.client_order_id for order in manager.open_orders()}
    assert open_ids == {open_id}


# --------------------------------------------------------------------------
# submit() -- orchestrating a place_order call, including a lost response
# --------------------------------------------------------------------------


def test_submit_adopts_the_brokers_reported_status(
    manager: OrderManager, clock: _ClockBox
) -> None:
    broker = _StubBroker()
    result = manager.create("NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1")
    broker.next_place_order_result = BrokerOrder(
        client_order_id=result.order.client_order_id,
        broker_order_id="B-1",
        instrument_id="NSE:INFY",
        side="buy",
        quantity=10,
        order_type="limit",
        limit_price=1500.0,
        status=OrderState.FILLED.value,
        filled_quantity=10,
        avg_fill_price=1499.5,
    )
    record = manager.submit(result.order.client_order_id, broker)
    assert record.state is OrderState.FILLED
    assert record.broker_order_id == "B-1"
    assert record.filled_quantity == 10
    assert record.avg_fill_price == 1499.5


def test_submit_transitions_to_unknown_on_a_failed_response(
    manager: OrderManager, clock: _ClockBox
) -> None:
    """A submission whose response is lost (timeout, connection drop)
    must never be silently treated as failed or retried blindly -- it
    becomes UNKNOWN until resolved."""
    broker = _StubBroker()
    result = manager.create("NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1")
    broker.next_place_order_error = TimeoutError("simulated network timeout")
    record = manager.submit(result.order.client_order_id, broker)
    assert record.state is OrderState.UNKNOWN


def test_handle_ambiguous_response_resolves_via_broker_query(
    manager: OrderManager, clock: _ClockBox
) -> None:
    broker = _StubBroker()
    result = manager.create("NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1")
    client_order_id = result.order.client_order_id
    # The broker actually processed the order even though the response
    # that would have told the client so was lost.
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
    manager.transition(client_order_id, OrderState.SUBMITTED)
    manager.transition(client_order_id, OrderState.UNKNOWN)

    resolved = manager.handle_ambiguous_response(client_order_id, broker)
    assert resolved.state is OrderState.OPEN
    assert resolved.broker_order_id == "B-1"


def test_handle_ambiguous_response_is_a_noop_when_not_unknown(manager: OrderManager) -> None:
    broker = _StubBroker()
    result = manager.create("NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1")
    manager.transition(result.order.client_order_id, OrderState.SUBMITTED)
    manager.transition(result.order.client_order_id, OrderState.OPEN)
    resolved = manager.handle_ambiguous_response(result.order.client_order_id, broker)
    assert resolved.state is OrderState.OPEN


def test_handle_ambiguous_response_rejects_when_broker_never_saw_it(
    manager: OrderManager,
) -> None:
    broker = _StubBroker()
    result = manager.create("NSE:INFY", "buy", 10, "limit", 1500.0, idempotency_key="req-1")
    client_order_id = result.order.client_order_id
    manager.transition(client_order_id, OrderState.SUBMITTED)
    manager.transition(client_order_id, OrderState.UNKNOWN)

    resolved = manager.handle_ambiguous_response(client_order_id, broker)
    assert resolved.state is OrderState.REJECTED
    assert resolved.reject_reason is not None
