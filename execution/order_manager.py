"""Order state machine, lifecycle management, and idempotent order creation.
See docs/SPECIFICATION.md section 12.1.

    CREATED -> SUBMITTED -> OPEN -> PARTIALLY_FILLED -> FILLED
                                  -> FILLED
                          -> REJECTED
             SUBMITTED/OPEN/PARTIALLY_FILLED -> CANCEL_REQUESTED -> CANCELLED
             OPEN/PARTIALLY_FILLED -> EXPIRED
             any non-terminal state -> UNKNOWN -> (resolved by querying the broker)

This module's own state names (``OrderState``) are this system's canonical
order-lifecycle vocabulary -- broader than the literal sequence in
docs/SPECIFICATION.md section 12.1 (which has no ``OPEN``,
``PARTIALLY_FILLED``, or ``EXPIRED`` as distinct states), because a real
broker response needs to distinguish "resting, unfilled" from "resting,
partially filled" to treat a partial fill as the first-class event section
12.2 requires, and because a good-for-day order that never fills needs a
terminal state of its own rather than silently staying "submitted"
forever. ``broker.base.BrokerOrder.status`` is a raw string every adapter
reports in its own vocabulary; this module is the one place that turns it
into this typed state.

On any ambiguous broker response (timeout after submission, duplicate ack),
this module must query order status by client-side ID before ever retrying
-- never blind-retry a place_order call. See docs/ARCHITECTURE.md and
``execution.order_reconciler`` (Phase 17), which builds the fuller
reconciliation sweep (stale orders, submission timeouts, orphaned broker
orders) on top of the single-order resolution :meth:`OrderManager.handle_ambiguous_response`
provides here.

**Traceability (Phase 17).** Every order this system creates must be
traceable back to the signal and risk decision that produced it --
:meth:`OrderManager.create` requires ``signal_id`` and ``risk_decision_id``
for exactly this reason, and every create/transition is written to an
``execution.execution_journal.ExecutionJournal`` (owned by this manager,
never optional to have -- only optional to inject a specific instance)
so the full chain (signal -> risk decision -> client order -> broker
order -> fills) can always be reconstructed after the fact, not only
inferred from whichever fields happen to still be in memory.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

from broker.base import Broker, BrokerOrder
from execution.execution_journal import ExecutionJournal, JournalEventType


class OrderState(StrEnum):
    CREATED = "created"
    SUBMITTED = "submitted"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


TERMINAL_STATES: frozenset[OrderState] = frozenset(
    {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED}
)
"""States no further transition is ever allowed out of."""

OPEN_STATES: frozenset[OrderState] = frozenset(
    {
        OrderState.SUBMITTED,
        OrderState.OPEN,
        OrderState.PARTIALLY_FILLED,
        OrderState.CANCEL_REQUESTED,
    }
)
"""States that still represent live broker-side exposure -- what
``Broker.get_open_orders`` should return. Deliberately excludes
``UNKNOWN``: a broker never reports "unknown" about its own order (by
the time it can answer at all, it knows), only the client-side record
can be uncertain -- see ``execution.order_reconciler`` for how those are
found and resolved."""

_ALLOWED_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.CREATED: frozenset({OrderState.SUBMITTED}),
    # Every non-terminal, non-CREATED state below can reach *any* terminal
    # state directly, on top of its normal forward-progress edges. This is
    # not the "live flow" this system itself drives step by step (that
    # never jumps straight from SUBMITTED to CANCELLED) -- it is what
    # reconciliation needs: after a disconnect, the broker's own report is
    # authoritative and may reflect several events this system never
    # observed individually (e.g. an order this system last saw resting
    # OPEN that was cancelled through another channel, or by the broker's
    # own RMS, while disconnected). Refusing to adopt that reported state
    # just because it does not match this system's own step-by-step model
    # would defeat reconciliation's entire purpose -- see
    # ``execution.order_reconciler.OrderReconciler``.
    OrderState.SUBMITTED: frozenset(
        {OrderState.OPEN, OrderState.PARTIALLY_FILLED, OrderState.UNKNOWN}
    )
    | TERMINAL_STATES,
    OrderState.OPEN: frozenset(
        {OrderState.PARTIALLY_FILLED, OrderState.CANCEL_REQUESTED, OrderState.UNKNOWN}
    )
    | TERMINAL_STATES,
    OrderState.PARTIALLY_FILLED: frozenset(
        {OrderState.PARTIALLY_FILLED, OrderState.CANCEL_REQUESTED, OrderState.UNKNOWN}
    )
    | TERMINAL_STATES,
    OrderState.CANCEL_REQUESTED: frozenset({OrderState.PARTIALLY_FILLED, OrderState.UNKNOWN})
    | TERMINAL_STATES,
    OrderState.UNKNOWN: frozenset({OrderState.OPEN, OrderState.PARTIALLY_FILLED}) | TERMINAL_STATES,
    OrderState.FILLED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.EXPIRED: frozenset(),
}
"""Every allowed ``from -> {to, ...}`` edge. ``CANCEL_REQUESTED`` can still
resolve to ``FILLED``/``PARTIALLY_FILLED`` because a cancel can race a fill
that was already in flight broker-side -- a real broker does not guarantee
a cancel request beats a fill that already happened. ``UNKNOWN`` can
resolve to anything non-terminal-or-terminal because, by definition, this
system does not yet know which; resolving it always goes through
:meth:`OrderManager.handle_ambiguous_response`, never a guess."""


class OrderStateError(RuntimeError):
    """An order transition was attempted that the state machine does not
    allow -- fail closed rather than silently accepting an inconsistent
    order lifecycle."""


class DuplicateSignalError(RuntimeError):
    """A caller tried to create a second order for a ``signal_id`` that
    already has one, under a *different* ``idempotency_key`` than the
    first attempt used. A legitimate retry reuses the same
    ``idempotency_key`` (and is handled silently by :meth:`OrderManager.create`
    returning the existing order); a different key for the same signal
    means a retry path generated a fresh key instead of reusing the
    original one -- a caller bug, not a legitimate resubmission, and
    exactly the kind of duplicate this system must refuse rather than
    quietly create a second order for.
    """


class ExecutionStateMachine:
    """Owns the order-lifecycle transition rules only -- no orchestration,
    no broker calls, no idempotency, no journaling. A pure state machine:
    given a current state and a proposed next state, says whether the
    move is allowed. :class:`OrderManager` composes one of these rather
    than embedding the transition table itself, so "what states can an
    order legally move through" is independently readable and testable
    from "how this system actually manages one."
    """

    def validate_transition(self, current: OrderState, new: OrderState) -> None:
        allowed = _ALLOWED_TRANSITIONS[current]
        if new not in allowed:
            raise OrderStateError(
                f"cannot transition {current.value} -> {new.value} "
                f"(allowed: {sorted(s.value for s in allowed)})"
            )

    @staticmethod
    def is_terminal(state: OrderState) -> bool:
        return state in TERMINAL_STATES

    @staticmethod
    def is_open(state: OrderState) -> bool:
        return state in OPEN_STATES

    @staticmethod
    def allowed_next_states(current: OrderState) -> frozenset[OrderState]:
        return _ALLOWED_TRANSITIONS[current]


@dataclass(frozen=True, slots=True)
class OrderRecord:
    client_order_id: str
    idempotency_key: str
    signal_id: str
    """The trading signal/decision that produced this order -- the root
    of the traceability chain this phase requires. Opaque to this
    module; the caller (a future live trading loop) defines what it
    means and how it is generated."""

    risk_decision_id: str
    """The specific risk evaluation that approved this order. Also
    opaque here -- see ``risk.risk_manager.RiskManager.evaluate``, whose
    caller is responsible for minting and passing through an ID for its
    own decision."""

    broker_order_id: str | None
    instrument_id: str
    side: str
    quantity: int
    order_type: str
    limit_price: float | None
    state: OrderState
    filled_quantity: int
    avg_fill_price: float | None
    reject_reason: str | None
    created_at: dt.datetime
    updated_at: dt.datetime

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {self.quantity}")
        if not (0 <= self.filled_quantity <= self.quantity):
            raise ValueError(
                f"filled_quantity ({self.filled_quantity}) must be in [0, {self.quantity}]"
            )
        if not self.signal_id:
            raise ValueError("signal_id must not be empty")
        if not self.risk_decision_id:
            raise ValueError("risk_decision_id must not be empty")

    def is_open(self) -> bool:
        return self.state in OPEN_STATES

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity


@dataclass(frozen=True, slots=True)
class OrderCreationResult:
    order: OrderRecord
    was_duplicate: bool
    """``True`` when ``idempotency_key`` had already been used -- ``order``
    is the pre-existing record, unmodified, not a new one. The caller made
    the same trade request twice; this system created it once."""


class OrderManager:
    """Owns valid state transitions for every order this system creates,
    and is the single place a trade request's idempotency is enforced
    client-side (:meth:`create`) -- a resubmitted identical request must
    never produce a second order. Every state transition also writes to
    this manager's :class:`~execution.execution_journal.ExecutionJournal`,
    so "every signal must be traceable" is a structural property of
    calling this class, not something a caller can forget to do.
    """

    def __init__(
        self,
        clock: Callable[[], dt.datetime] | None = None,
        journal: ExecutionJournal | None = None,
    ) -> None:
        self._clock: Callable[[], dt.datetime] = clock or dt.datetime.now
        self._orders: dict[str, OrderRecord] = {}
        self._idempotency_index: dict[str, str] = {}
        self._signal_index: dict[str, str] = {}
        self._state_machine = ExecutionStateMachine()
        self.journal = journal or ExecutionJournal(clock=self._clock)

    def create(
        self,
        instrument_id: str,
        side: str,
        quantity: int,
        order_type: str,
        limit_price: float | None,
        idempotency_key: str,
        *,
        signal_id: str,
        risk_decision_id: str,
    ) -> OrderCreationResult:
        """Create an order in ``CREATED`` state with a unique, freshly
        generated client-side ID -- unless ``idempotency_key`` has already
        been used, in which case the existing order is returned unchanged
        and no new order is created.

        Raises :class:`DuplicateSignalError` if ``signal_id`` already has
        an order under a *different* ``idempotency_key`` -- see that
        exception's docstring for why this is refused rather than
        silently creating a second order.
        """
        if not signal_id:
            raise ValueError("signal_id must not be empty")
        if not risk_decision_id:
            raise ValueError("risk_decision_id must not be empty")

        existing_id = self._idempotency_index.get(idempotency_key)
        if existing_id is not None:
            existing = self._orders[existing_id]
            self.journal.record(
                JournalEventType.DUPLICATE_SUPPRESSED,
                signal_id=signal_id,
                risk_decision_id=risk_decision_id,
                client_order_id=existing_id,
                detail=f"idempotency_key {idempotency_key!r} already used; no new order created",
            )
            return OrderCreationResult(order=existing, was_duplicate=True)

        existing_for_signal = self._signal_index.get(signal_id)
        if existing_for_signal is not None:
            self.journal.record(
                JournalEventType.DUPLICATE_SIGNAL_REJECTED,
                signal_id=signal_id,
                risk_decision_id=risk_decision_id,
                client_order_id=existing_for_signal,
                detail=(
                    f"signal_id already produced order {existing_for_signal!r} under a "
                    f"different idempotency_key; refusing to create a second order"
                ),
            )
            raise DuplicateSignalError(
                f"signal_id {signal_id!r} already produced order {existing_for_signal!r}; "
                f"refusing to create a second order under a different idempotency_key "
                f"({idempotency_key!r}) -- reuse the original idempotency_key to retry safely"
            )

        now = self._clock()
        record = OrderRecord(
            client_order_id=str(uuid.uuid4()),
            idempotency_key=idempotency_key,
            signal_id=signal_id,
            risk_decision_id=risk_decision_id,
            broker_order_id=None,
            instrument_id=instrument_id,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            state=OrderState.CREATED,
            filled_quantity=0,
            avg_fill_price=None,
            reject_reason=None,
            created_at=now,
            updated_at=now,
        )
        self._orders[record.client_order_id] = record
        self._idempotency_index[idempotency_key] = record.client_order_id
        self._signal_index[signal_id] = record.client_order_id
        self.journal.record(
            JournalEventType.ORDER_CREATED,
            signal_id=signal_id,
            risk_decision_id=risk_decision_id,
            client_order_id=record.client_order_id,
            detail=f"{side} {quantity} {instrument_id} @ {order_type}",
        )
        return OrderCreationResult(order=record, was_duplicate=False)

    def get(self, client_order_id: str) -> OrderRecord:
        try:
            return self._orders[client_order_id]
        except KeyError:
            raise OrderStateError(f"unknown client_order_id: {client_order_id}") from None

    def all_orders(self) -> list[OrderRecord]:
        """Every order this manager has ever created, in any state --
        including ``UNKNOWN``, which :meth:`open_orders` deliberately
        excludes (see ``OPEN_STATES``). Used by
        ``execution.order_reconciler.OrderReconciler`` to find orders
        needing reconciliation.
        """
        return list(self._orders.values())

    def open_orders(self) -> list[OrderRecord]:
        return [order for order in self._orders.values() if order.is_open()]

    def transition(
        self,
        client_order_id: str,
        new_state: OrderState,
        *,
        broker_order_id: str | None = None,
        filled_quantity: int | None = None,
        avg_fill_price: float | None = None,
        reject_reason: str | None = None,
    ) -> OrderRecord:
        """Apply a state transition, rejecting any transition not allowed by
        the state machine above. Fields not supplied are carried over
        unchanged from the current record. Always writes a
        ``STATE_CHANGED`` journal entry, plus a ``FILL_OBSERVED`` one if
        ``filled_quantity`` increased -- every transition is traceable
        without the caller having to remember to log it.
        """
        current = self.get(client_order_id)
        self._state_machine.validate_transition(current.state, new_state)
        if new_state is OrderState.REJECTED and reject_reason is None:
            raise OrderStateError("a REJECTED transition must carry a reject_reason")

        new_filled_quantity = (
            filled_quantity if filled_quantity is not None else current.filled_quantity
        )
        updated = replace(
            current,
            state=new_state,
            broker_order_id=(
                broker_order_id if broker_order_id is not None else current.broker_order_id
            ),
            filled_quantity=new_filled_quantity,
            avg_fill_price=(
                avg_fill_price if avg_fill_price is not None else current.avg_fill_price
            ),
            reject_reason=(
                reject_reason if reject_reason is not None else current.reject_reason
            ),
            updated_at=self._clock(),
        )
        self._orders[client_order_id] = updated

        self.journal.record(
            JournalEventType.STATE_CHANGED,
            signal_id=updated.signal_id,
            risk_decision_id=updated.risk_decision_id,
            client_order_id=client_order_id,
            broker_order_id=updated.broker_order_id,
            detail=f"{current.state.value} -> {new_state.value}",
        )
        if new_filled_quantity > current.filled_quantity:
            self.journal.record(
                JournalEventType.FILL_OBSERVED,
                signal_id=updated.signal_id,
                risk_decision_id=updated.risk_decision_id,
                client_order_id=client_order_id,
                broker_order_id=updated.broker_order_id,
                detail=(
                    f"filled_quantity {current.filled_quantity} -> {new_filled_quantity} "
                    f"(avg_fill_price={updated.avg_fill_price})"
                ),
            )
        return updated

    def submit(self, client_order_id: str, broker: Broker) -> OrderRecord:
        """Submit a ``CREATED`` order to ``broker`` and adopt its reported
        status. A submission that raises (network failure, timeout -- the
        broker may or may not have actually received and processed the
        order) transitions the local record to ``UNKNOWN`` instead of
        propagating, so the caller can resolve it via
        :meth:`handle_ambiguous_response` (or
        ``execution.order_reconciler.OrderReconciler.resolve_unknown``)
        rather than blind-retrying.
        """
        record = self.transition(client_order_id, OrderState.SUBMITTED)
        request = BrokerOrder(
            client_order_id=record.client_order_id,
            broker_order_id=None,
            instrument_id=record.instrument_id,
            side=record.side,
            quantity=record.quantity,
            order_type=record.order_type,
            limit_price=record.limit_price,
            status=OrderState.SUBMITTED.value,
        )
        try:
            result = broker.place_order(request)
        except Exception:
            return self.transition(client_order_id, OrderState.UNKNOWN)
        return self.adopt_broker_order(client_order_id, result)

    def cancel(self, client_order_id: str, broker: Broker) -> OrderRecord:
        """Requests cancellation via the broker and adopts its response.
        Exactly like :meth:`submit`: a raised exception means the actual
        outcome is unknown (did the cancel reach the broker? did it
        apply before a fill?) -- transitions to ``UNKNOWN`` rather than
        assuming success or retrying blindly.
        """
        self.transition(client_order_id, OrderState.CANCEL_REQUESTED)
        try:
            result = broker.cancel_order(client_order_id)
        except Exception:
            return self.transition(client_order_id, OrderState.UNKNOWN)
        return self.adopt_broker_order(client_order_id, result)

    def handle_ambiguous_response(self, client_order_id: str, broker: Broker) -> OrderRecord:
        """Resolve an ``UNKNOWN``-state order by querying the broker for
        its actual status via ``client_order_id``, rather than retrying
        blindly. If the broker itself does not recognize the ID, the order
        never reached it and is treated as rejected.
        """
        record = self.get(client_order_id)
        if record.state is not OrderState.UNKNOWN:
            return record
        try:
            broker_order = broker.get_order(client_order_id)
        except Exception as exc:
            return self.transition(
                client_order_id,
                OrderState.REJECTED,
                reject_reason=f"broker does not recognize this order: {exc}",
            )
        return self.adopt_broker_order(client_order_id, broker_order)

    def adopt_broker_order(self, client_order_id: str, broker_order: BrokerOrder) -> OrderRecord:
        try:
            new_state = OrderState(broker_order.status)
        except ValueError:
            raise OrderStateError(
                f"broker reported an unrecognized status: {broker_order.status!r}"
            ) from None
        return self.transition(
            client_order_id,
            new_state,
            broker_order_id=broker_order.broker_order_id,
            filled_quantity=broker_order.filled_quantity,
            avg_fill_price=broker_order.avg_fill_price,
            reject_reason=broker_order.reject_reason,
        )
