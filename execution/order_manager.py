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
-- never blind-retry a place_order call. See docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

from broker.base import Broker, BrokerOrder


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
``Broker.get_open_orders`` should return."""

_ALLOWED_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.CREATED: frozenset({OrderState.SUBMITTED}),
    OrderState.SUBMITTED: frozenset(
        {
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.OPEN: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.EXPIRED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.CANCEL_REQUESTED: frozenset(
        {
            OrderState.CANCELLED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.UNKNOWN: frozenset(
        {
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        }
    ),
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


@dataclass(frozen=True, slots=True)
class OrderRecord:
    client_order_id: str
    idempotency_key: str
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
    never produce a second order.
    """

    def __init__(self, clock: Callable[[], dt.datetime] | None = None) -> None:
        self._clock: Callable[[], dt.datetime] = clock or dt.datetime.now
        self._orders: dict[str, OrderRecord] = {}
        self._idempotency_index: dict[str, str] = {}

    def create(
        self,
        instrument_id: str,
        side: str,
        quantity: int,
        order_type: str,
        limit_price: float | None,
        idempotency_key: str,
    ) -> OrderCreationResult:
        """Create an order in ``CREATED`` state with a unique, freshly
        generated client-side ID -- unless ``idempotency_key`` has already
        been used, in which case the existing order is returned unchanged
        and no new order is created.
        """
        existing_id = self._idempotency_index.get(idempotency_key)
        if existing_id is not None:
            return OrderCreationResult(order=self._orders[existing_id], was_duplicate=True)

        now = self._clock()
        record = OrderRecord(
            client_order_id=str(uuid.uuid4()),
            idempotency_key=idempotency_key,
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
        return OrderCreationResult(order=record, was_duplicate=False)

    def get(self, client_order_id: str) -> OrderRecord:
        try:
            return self._orders[client_order_id]
        except KeyError:
            raise OrderStateError(f"unknown client_order_id: {client_order_id}") from None

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
        unchanged from the current record.
        """
        current = self.get(client_order_id)
        allowed = _ALLOWED_TRANSITIONS[current.state]
        if new_state not in allowed:
            raise OrderStateError(
                f"order {client_order_id}: cannot transition {current.state.value} -> "
                f"{new_state.value} (allowed: {sorted(s.value for s in allowed)})"
            )
        if new_state is OrderState.REJECTED and reject_reason is None:
            raise OrderStateError("a REJECTED transition must carry a reject_reason")

        updated = replace(
            current,
            state=new_state,
            broker_order_id=(
                broker_order_id if broker_order_id is not None else current.broker_order_id
            ),
            filled_quantity=(
                filled_quantity if filled_quantity is not None else current.filled_quantity
            ),
            avg_fill_price=(
                avg_fill_price if avg_fill_price is not None else current.avg_fill_price
            ),
            reject_reason=(
                reject_reason if reject_reason is not None else current.reject_reason
            ),
            updated_at=self._clock(),
        )
        self._orders[client_order_id] = updated
        return updated

    def submit(self, client_order_id: str, broker: Broker) -> OrderRecord:
        """Submit a ``CREATED`` order to ``broker`` and adopt its reported
        status. A submission that raises (network failure, timeout -- the
        broker may or may not have actually received and processed the
        order) transitions the local record to ``UNKNOWN`` instead of
        propagating, so the caller can resolve it via
        :meth:`handle_ambiguous_response` rather than blind-retrying.
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
        return self._adopt_broker_order(client_order_id, result)

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
        return self._adopt_broker_order(client_order_id, broker_order)

    def _adopt_broker_order(self, client_order_id: str, broker_order: BrokerOrder) -> OrderRecord:
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
