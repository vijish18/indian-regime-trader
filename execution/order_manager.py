"""Order state machine and lifecycle management. See
docs/SPECIFICATION.md section 12.1:

    CREATED -> VALIDATED -> SUBMITTED -> ACKNOWLEDGED -> PARTIAL/FILLED
                                                       -> REJECTED
    SUBMITTED/ACKNOWLEDGED -> CANCEL_REQUESTED -> CANCELLED
    UNKNOWN state -> RECONCILIATION REQUIRED

On any ambiguous broker response (timeout after submission, duplicate ack),
this module must query order status by client-side ID before ever retrying
-- never blind-retry a place_order call. See docs/ARCHITECTURE.md.

Not implemented yet (Phase 10/11).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum


class OrderState(StrEnum):
    CREATED = "created"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    RECONCILIATION_REQUIRED = "reconciliation_required"


@dataclass(frozen=True)
class OrderRecord:
    client_order_id: str
    broker_order_id: str | None
    instrument_id: str
    side: str
    quantity: int
    order_type: str
    limit_price: float | None
    state: OrderState
    created_at: dt.datetime


class OrderManager:
    """Owns valid state transitions for every order this system creates."""

    def create(
        self,
        instrument_id: str,
        side: str,
        quantity: int,
        order_type: str,
        limit_price: float | None,
    ) -> OrderRecord:
        """Create an order in ``CREATED`` state with a unique client-side ID."""
        raise NotImplementedError("Phase 10/11: order management is not implemented yet.")

    def transition(self, client_order_id: str, new_state: OrderState) -> OrderRecord:
        """Apply a state transition, rejecting any transition not allowed by
        the state machine above.
        """
        raise NotImplementedError("Phase 10/11: order management is not implemented yet.")

    def handle_ambiguous_response(self, client_order_id: str) -> OrderRecord:
        """Resolve a timeout/duplicate-response case by querying the broker
        for the order's actual status via ``get_open_orders``, rather than
        retrying blindly.
        """
        raise NotImplementedError("Phase 10/11: order management is not implemented yet.")
