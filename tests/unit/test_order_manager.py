"""Unit tests for the order state machine and idempotent creation
(``execution/order_manager.py``, Phases 14 & 17) -- isolated from any real
broker via a small controllable stub, so these tests are about the state
machine and lifecycle bookkeeping themselves, not about paper-fill
simulation (see test_paper_broker.py for that) or the broker-reconciliation
sweep (see test_order_reconciler.py).
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
from execution.execution_journal import JournalEventType
from execution.order_manager import (
    DuplicateSignalError,
    ExecutionStateMachine,
    OrderCreationResult,
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
        self.next_cancel_order_result: BrokerOrder | None = None
        self.next_cancel_order_error: Exception | None = None
        self.orders: dict[str, BrokerOrder] = {}
        self.placed: list[BrokerOrder] = []
        self.cancelled: list[str] = []

    def capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        raise NotImplementedError

    def get_account(self) -> BrokerAccount:
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

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        raise NotImplementedError

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        raise NotImplementedError

    def subscribe_market_data(
        self, instrument_ids: list[str], on_tick: Callable[[BrokerQuote], None]
    ) -> Callable[[], None]:
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
        self.cancelled.append(order_id)
        if self.next_cancel_order_error is not None:
            error = self.next_cancel_order_error
            self.next_cancel_order_error = None
            raise error
        if self.next_cancel_order_result is not None:
            return self.next_cancel_order_result
        raise NotImplementedError("script next_cancel_order_result or next_cancel_order_error")

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


def create(
    manager: OrderManager,
    instrument_id: str = "NSE:INFY",
    side: str = "buy",
    quantity: int = 10,
    order_type: str = "limit",
    limit_price: float | None = 1500.0,
    *,
    idempotency_key: str,
    signal_id: str | None = None,
    risk_decision_id: str | None = None,
) -> OrderCreationResult:
    """Thin wrapper defaulting signal_id/risk_decision_id from
    idempotency_key when not given, since most tests only care about
    idempotency/state-machine behavior, not the specific trace IDs."""
    return manager.create(
        instrument_id,
        side,
        quantity,
        order_type,
        limit_price,
        idempotency_key=idempotency_key,
        signal_id=signal_id or f"sig-{idempotency_key}",
        risk_decision_id=risk_decision_id or f"rd-{idempotency_key}",
    )


# --------------------------------------------------------------------------
# ExecutionStateMachine -- the transition rules, in isolation
# --------------------------------------------------------------------------


def test_state_machine_allows_a_documented_transition() -> None:
    machine = ExecutionStateMachine()
    machine.validate_transition(OrderState.CREATED, OrderState.SUBMITTED)  # must not raise


def test_state_machine_rejects_an_undocumented_transition() -> None:
    machine = ExecutionStateMachine()
    with pytest.raises(OrderStateError, match="cannot transition"):
        machine.validate_transition(OrderState.CREATED, OrderState.FILLED)


def test_state_machine_reports_terminal_states() -> None:
    machine = ExecutionStateMachine()
    assert machine.is_terminal(OrderState.FILLED) is True
    assert machine.is_terminal(OrderState.CANCELLED) is True
    assert machine.is_terminal(OrderState.OPEN) is False


def test_state_machine_reports_open_states() -> None:
    machine = ExecutionStateMachine()
    assert machine.is_open(OrderState.OPEN) is True
    assert machine.is_open(OrderState.UNKNOWN) is False  # see OPEN_STATES' own docstring


def test_state_machine_no_transitions_leave_a_terminal_state() -> None:
    machine = ExecutionStateMachine()
    terminal_states = (
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
    )
    for terminal in terminal_states:
        assert machine.allowed_next_states(terminal) == frozenset()


# --------------------------------------------------------------------------
# create() / idempotency / duplicate-signal detection
# --------------------------------------------------------------------------


def test_create_generates_a_unique_client_order_id(manager: OrderManager) -> None:
    first = create(manager, "NSE:INFY", idempotency_key="req-1")
    second = create(manager, "NSE:TCS", idempotency_key="req-2")
    assert first.order.client_order_id != second.order.client_order_id
    assert first.was_duplicate is False
    assert second.was_duplicate is False


def test_create_is_idempotent_by_key(manager: OrderManager) -> None:
    first = create(manager, idempotency_key="req-1")
    replay = create(manager, idempotency_key="req-1")
    assert replay.was_duplicate is True
    assert replay.order == first.order
    assert replay.order.client_order_id == first.order.client_order_id


def test_create_with_a_different_idempotency_key_and_signal_is_a_separate_order(
    manager: OrderManager,
) -> None:
    first = create(manager, idempotency_key="req-1")
    second = create(manager, idempotency_key="req-2")
    assert first.order.client_order_id != second.order.client_order_id


def test_create_raises_duplicate_signal_error_for_a_reused_signal_under_a_new_key(
    manager: OrderManager,
) -> None:
    """A legitimate retry reuses the same idempotency_key (handled
    silently above). A *different* key for the same signal_id means a
    retry path minted a fresh key instead of reusing the original --
    exactly the caller bug this system must refuse, not quietly accept.
    """
    first = manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0,
        idempotency_key="req-1", signal_id="sig-A", risk_decision_id="rd-A",
    )
    with pytest.raises(DuplicateSignalError, match="sig-A"):
        manager.create(
            "NSE:INFY", "buy", 10, "limit", 1500.0,
            idempotency_key="req-2", signal_id="sig-A", risk_decision_id="rd-A",
        )
    # The original order is untouched.
    assert manager.get(first.order.client_order_id).client_order_id == first.order.client_order_id


def test_create_rejects_an_empty_signal_id(manager: OrderManager) -> None:
    with pytest.raises(ValueError, match="signal_id"):
        manager.create(
            "NSE:INFY", "buy", 10, "limit", 1500.0,
            idempotency_key="req-1", signal_id="", risk_decision_id="rd-1",
        )


def test_create_rejects_an_empty_risk_decision_id(manager: OrderManager) -> None:
    with pytest.raises(ValueError, match="risk_decision_id"):
        manager.create(
            "NSE:INFY", "buy", 10, "limit", 1500.0,
            idempotency_key="req-1", signal_id="sig-1", risk_decision_id="",
        )


def test_create_result_starts_in_created_state(manager: OrderManager) -> None:
    result = create(manager, idempotency_key="req-1")
    assert result.order.state is OrderState.CREATED
    assert result.order.filled_quantity == 0
    assert result.order.remaining_quantity() == 10


def test_create_carries_signal_and_risk_decision_ids_onto_the_record(
    manager: OrderManager,
) -> None:
    result = manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0,
        idempotency_key="req-1", signal_id="sig-42", risk_decision_id="rd-99",
    )
    assert result.order.signal_id == "sig-42"
    assert result.order.risk_decision_id == "rd-99"


# --------------------------------------------------------------------------
# transition() -- the state machine, as OrderManager applies it
# --------------------------------------------------------------------------


def test_valid_transition_sequence(manager: OrderManager) -> None:
    order_id = create(manager, idempotency_key="req-1").order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    manager.transition(order_id, OrderState.OPEN)
    manager.transition(order_id, OrderState.PARTIALLY_FILLED, filled_quantity=4)
    record = manager.transition(order_id, OrderState.FILLED, filled_quantity=10)
    assert record.state is OrderState.FILLED
    assert record.filled_quantity == 10
    assert record.is_terminal()


def test_invalid_transition_is_rejected(manager: OrderManager) -> None:
    order_id = create(manager, idempotency_key="req-1").order.client_order_id
    with pytest.raises(OrderStateError, match="cannot transition"):
        manager.transition(order_id, OrderState.FILLED)


def test_terminal_state_accepts_no_further_transition(manager: OrderManager) -> None:
    order_id = create(manager, idempotency_key="req-1").order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    manager.transition(order_id, OrderState.REJECTED, reject_reason="no liquidity")
    with pytest.raises(OrderStateError):
        manager.transition(order_id, OrderState.OPEN)


def test_rejected_transition_requires_a_reason(manager: OrderManager) -> None:
    order_id = create(manager, idempotency_key="req-1").order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    with pytest.raises(OrderStateError, match="reject_reason"):
        manager.transition(order_id, OrderState.REJECTED)


def test_cancel_requested_can_still_resolve_to_filled(manager: OrderManager) -> None:
    """A cancel request can race a fill that was already in flight
    broker-side -- the state machine must not forbid that outcome."""
    order_id = create(manager, idempotency_key="req-1").order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    manager.transition(order_id, OrderState.OPEN)
    manager.transition(order_id, OrderState.CANCEL_REQUESTED)
    record = manager.transition(order_id, OrderState.FILLED, filled_quantity=10)
    assert record.state is OrderState.FILLED


def test_get_unknown_client_order_id_raises(manager: OrderManager) -> None:
    with pytest.raises(OrderStateError, match="unknown client_order_id"):
        manager.get("does-not-exist")


def test_open_orders_lists_only_non_terminal_orders(manager: OrderManager) -> None:
    open_id = create(manager, "NSE:INFY", idempotency_key="req-open").order.client_order_id
    filled_id = create(manager, "NSE:TCS", idempotency_key="req-filled").order.client_order_id
    manager.transition(open_id, OrderState.SUBMITTED)
    manager.transition(open_id, OrderState.OPEN)
    manager.transition(filled_id, OrderState.SUBMITTED)
    manager.transition(filled_id, OrderState.FILLED, filled_quantity=5)

    open_ids = {order.client_order_id for order in manager.open_orders()}
    assert open_ids == {open_id}


def test_all_orders_includes_unknown_state_orders_unlike_open_orders(
    manager: OrderManager,
) -> None:
    order_id = create(manager, idempotency_key="req-1").order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    manager.transition(order_id, OrderState.UNKNOWN)

    assert order_id not in {o.client_order_id for o in manager.open_orders()}
    assert order_id in {o.client_order_id for o in manager.all_orders()}


# --------------------------------------------------------------------------
# submit() -- orchestrating a place_order call, including a lost response
# --------------------------------------------------------------------------


def test_submit_adopts_the_brokers_reported_status(
    manager: OrderManager, clock: _ClockBox
) -> None:
    broker = _StubBroker()
    result = create(manager, idempotency_key="req-1")
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
    result = create(manager, idempotency_key="req-1")
    broker.next_place_order_error = TimeoutError("simulated network timeout")
    record = manager.submit(result.order.client_order_id, broker)
    assert record.state is OrderState.UNKNOWN
    assert len(broker.placed) == 1  # the CRITICAL guarantee: exactly one attempt


def test_handle_ambiguous_response_resolves_via_broker_query(
    manager: OrderManager, clock: _ClockBox
) -> None:
    broker = _StubBroker()
    result = create(manager, idempotency_key="req-1")
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
    result = create(manager, idempotency_key="req-1")
    manager.transition(result.order.client_order_id, OrderState.SUBMITTED)
    manager.transition(result.order.client_order_id, OrderState.OPEN)
    resolved = manager.handle_ambiguous_response(result.order.client_order_id, broker)
    assert resolved.state is OrderState.OPEN


def test_handle_ambiguous_response_rejects_when_broker_never_saw_it(
    manager: OrderManager,
) -> None:
    broker = _StubBroker()
    result = create(manager, idempotency_key="req-1")
    client_order_id = result.order.client_order_id
    manager.transition(client_order_id, OrderState.SUBMITTED)
    manager.transition(client_order_id, OrderState.UNKNOWN)

    resolved = manager.handle_ambiguous_response(client_order_id, broker)
    assert resolved.state is OrderState.REJECTED
    assert resolved.reject_reason is not None


# --------------------------------------------------------------------------
# cancel() -- the same "never blind-retry" discipline as submit()
# --------------------------------------------------------------------------


def test_cancel_adopts_the_brokers_response(manager: OrderManager) -> None:
    broker = _StubBroker()
    order_id = create(manager, idempotency_key="req-1").order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    manager.transition(order_id, OrderState.OPEN)
    broker.next_cancel_order_result = BrokerOrder(
        client_order_id=order_id,
        broker_order_id="B-1",
        instrument_id="NSE:INFY",
        side="buy",
        quantity=10,
        order_type="limit",
        limit_price=1500.0,
        status=OrderState.CANCELLED.value,
    )
    record = manager.cancel(order_id, broker)
    assert record.state is OrderState.CANCELLED
    assert broker.cancelled == [order_id]


def test_cancel_transitions_to_unknown_on_a_failed_response(manager: OrderManager) -> None:
    broker = _StubBroker()
    order_id = create(manager, idempotency_key="req-1").order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    manager.transition(order_id, OrderState.OPEN)
    broker.next_cancel_order_error = TimeoutError("simulated network timeout")
    record = manager.cancel(order_id, broker)
    assert record.state is OrderState.UNKNOWN


# --------------------------------------------------------------------------
# Journal integration -- every lifecycle event is traceable without the
# caller having to remember to log it
# --------------------------------------------------------------------------


def test_create_records_an_order_created_journal_entry(manager: OrderManager) -> None:
    result = manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0,
        idempotency_key="req-1", signal_id="sig-1", risk_decision_id="rd-1",
    )
    entries = manager.journal.for_client_order_id(result.order.client_order_id)
    assert [e.event_type for e in entries] == [JournalEventType.ORDER_CREATED]
    assert entries[0].signal_id == "sig-1"
    assert entries[0].risk_decision_id == "rd-1"


def test_create_records_a_duplicate_suppressed_entry_on_idempotent_replay(
    manager: OrderManager,
) -> None:
    first = create(manager, idempotency_key="req-1")
    create(manager, idempotency_key="req-1")
    entries = manager.journal.for_client_order_id(first.order.client_order_id)
    assert JournalEventType.DUPLICATE_SUPPRESSED in [e.event_type for e in entries]


def test_create_records_a_duplicate_signal_rejected_entry(manager: OrderManager) -> None:
    first = manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0,
        idempotency_key="req-1", signal_id="sig-A", risk_decision_id="rd-A",
    )
    with pytest.raises(DuplicateSignalError):
        manager.create(
            "NSE:INFY", "buy", 10, "limit", 1500.0,
            idempotency_key="req-2", signal_id="sig-A", risk_decision_id="rd-A",
        )
    entries = manager.journal.for_signal("sig-A")
    assert JournalEventType.DUPLICATE_SIGNAL_REJECTED in [e.event_type for e in entries]
    assert first.order.client_order_id  # sanity: the original order still exists


def test_transition_records_a_state_changed_entry(manager: OrderManager) -> None:
    order_id = create(manager, idempotency_key="req-1").order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    entries = manager.journal.for_client_order_id(order_id)
    state_changes = [e for e in entries if e.event_type is JournalEventType.STATE_CHANGED]
    assert len(state_changes) == 1
    assert state_changes[0].detail == "created -> submitted"


def test_transition_records_a_fill_observed_entry_when_filled_quantity_increases(
    manager: OrderManager,
) -> None:
    order_id = create(manager, idempotency_key="req-1").order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    manager.transition(order_id, OrderState.OPEN)
    manager.transition(order_id, OrderState.PARTIALLY_FILLED, filled_quantity=4)
    manager.transition(order_id, OrderState.FILLED, filled_quantity=10)

    fills = [
        e
        for e in manager.journal.for_client_order_id(order_id)
        if e.event_type is JournalEventType.FILL_OBSERVED
    ]
    assert len(fills) == 2  # 0->4, then 4->10


def test_transition_does_not_record_a_fill_observed_entry_when_quantity_is_unchanged(
    manager: OrderManager,
) -> None:
    order_id = create(manager, idempotency_key="req-1").order.client_order_id
    manager.transition(order_id, OrderState.SUBMITTED)
    manager.transition(order_id, OrderState.OPEN)  # no filled_quantity given
    fills = [
        e
        for e in manager.journal.for_client_order_id(order_id)
        if e.event_type is JournalEventType.FILL_OBSERVED
    ]
    assert fills == []


def test_journal_trace_reconstructs_the_full_identity_chain(manager: OrderManager) -> None:
    broker = _StubBroker()
    result = manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0,
        idempotency_key="req-1", signal_id="sig-77", risk_decision_id="rd-88",
    )
    broker.next_place_order_result = BrokerOrder(
        client_order_id=result.order.client_order_id,
        broker_order_id="B-99",
        instrument_id="NSE:INFY",
        side="buy",
        quantity=10,
        order_type="limit",
        limit_price=1500.0,
        status=OrderState.FILLED.value,
        filled_quantity=10,
        avg_fill_price=1499.5,
    )
    manager.submit(result.order.client_order_id, broker)

    trace = manager.journal.trace(result.order.client_order_id)
    assert trace.signal_id == "sig-77"
    assert trace.risk_decision_id == "rd-88"
    assert trace.broker_order_id == "B-99"
    assert len(trace.fills()) == 1


def test_journal_trace_raises_for_an_id_it_has_never_seen(manager: OrderManager) -> None:
    with pytest.raises(KeyError):
        manager.journal.trace("does-not-exist")
