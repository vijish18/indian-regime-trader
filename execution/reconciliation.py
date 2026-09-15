"""Reconciles local *position and cash* state against the broker's actual
state, at startup and on a recurring schedule. See docs/SPECIFICATION.md
section 15. Implemented in Phase 18 as part of the restart-recovery
sequence (``execution/startup.py``) -- Phase 11b left this stubbed, and
this is where it stopped being deferred.

Order-level reconciliation (resolving an ``UNKNOWN`` order, stale-order
refresh, submission-timeout handling, and the full post-reconnect sweep)
is ``execution.order_reconciler.OrderReconciler`` (Phase 17), not this
module -- :meth:`ReconciliationEngine.reconcile_open_orders` delegates to
it and translates its report into this module's shape.

A mismatch must be quarantined, not silently resolved: the affected
instrument is frozen from new orders (:meth:`ReconciliationEngine.quarantined_instruments`)
while the rest of the reconciled portfolio continues to be managed
normally. See docs/ARCHITECTURE.md and ``execution/startup.py``'s own
"never auto-resolve a genuine discrepancy" rule -- this engine reports a
mismatch, it does not decide which side (local or broker) is right.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from broker.base import Broker
from execution.order_manager import OrderManager
from execution.order_reconciler import OrderReconciler
from execution.position_tracker import PositionTracker


class ReconciliationStatus(StrEnum):
    CLEAN = "clean"
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class ReconciliationMismatch:
    instrument_id: str
    local_quantity: int
    broker_quantity: int
    detail: str


@dataclass(frozen=True)
class ReconciliationReport:
    as_of: dt.datetime
    status: ReconciliationStatus
    mismatches: tuple[ReconciliationMismatch, ...]


class ReconciliationEngine:
    """Compares local position/cash/order state against the broker and
    produces a report; quarantined instruments must be excluded from new
    order generation until resolved.
    """

    def __init__(
        self,
        position_tracker: PositionTracker,
        order_manager: OrderManager,
        broker: Broker,
        order_reconciler: OrderReconciler | None = None,
        clock: Callable[[], dt.datetime] | None = None,
    ) -> None:
        self.position_tracker = position_tracker
        self.order_manager = order_manager
        self.broker = broker
        self.order_reconciler = order_reconciler or OrderReconciler(clock=clock)
        self._clock: Callable[[], dt.datetime] = clock or (lambda: dt.datetime.now(dt.UTC))
        self._quarantined: set[str] = set()

    def reconcile_positions(self) -> ReconciliationReport:
        """Compares every locally-tracked position's quantity against the
        broker's own reported quantity for the same instrument, in both
        directions -- an instrument the broker reports that local state
        has never seen ("missing local record") is exactly as much a
        mismatch as a quantity that merely disagrees.
        """
        local = {p.instrument_id: p.quantity for p in self.position_tracker.current_positions()}
        broker_positions = {p.instrument_id: p.quantity for p in self.broker.get_positions()}
        all_instrument_ids = sorted(set(local) | set(broker_positions))

        mismatches = []
        for instrument_id in all_instrument_ids:
            local_quantity = local.get(instrument_id, 0)
            broker_quantity = broker_positions.get(instrument_id, 0)
            if local_quantity == broker_quantity:
                continue
            if instrument_id not in local:
                detail = "broker reports a position this system has no local record of"
            elif instrument_id not in broker_positions:
                detail = "local state holds a position the broker no longer reports"
            else:
                detail = f"quantity mismatch: local={local_quantity} broker={broker_quantity}"
            mismatches.append(
                ReconciliationMismatch(
                    instrument_id=instrument_id,
                    local_quantity=local_quantity,
                    broker_quantity=broker_quantity,
                    detail=detail,
                )
            )

        self._quarantined = {m.instrument_id for m in mismatches}
        status = ReconciliationStatus.QUARANTINED if mismatches else ReconciliationStatus.CLEAN
        return ReconciliationReport(
            as_of=self._clock(), status=status, mismatches=tuple(mismatches)
        )

    def reconcile_open_orders(self) -> ReconciliationReport:
        """Delegates the actual reconciliation work to
        ``OrderReconciler.reconcile_after_reconnect`` (resolving every
        ``UNKNOWN`` order, refreshing stale ones, timing out stuck
        submissions -- all of that is a safe, broker-confirmed *resolution*,
        not a discrepancy) and reports only what that sweep could not
        safely resolve on its own: orders the broker reports that this
        system has no record of at all ("missing local record" /
        "unknown local order" from the broker's side).
        """
        report = self.order_reconciler.reconcile_after_reconnect(self.order_manager, self.broker)
        mismatches = tuple(
            ReconciliationMismatch(
                instrument_id=order.instrument_id,
                local_quantity=0,
                broker_quantity=order.quantity,
                detail=(
                    f"broker reports an open order this system has no local record of: "
                    f"{order.side} {order.quantity} (broker_order_id={order.broker_order_id})"
                ),
            )
            for order in report.orphaned_broker_orders
        )
        status = ReconciliationStatus.QUARANTINED if mismatches else ReconciliationStatus.CLEAN
        return ReconciliationReport(as_of=self._clock(), status=status, mismatches=mismatches)

    def quarantined_instruments(self) -> set[str]:
        """Instrument IDs currently frozen from new orders due to an
        unresolved mismatch -- reflects the most recent
        :meth:`reconcile_positions` call only; call it again to refresh.
        """
        return set(self._quarantined)
