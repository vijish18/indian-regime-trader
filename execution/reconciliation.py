"""Reconciles local *position and cash* state against the broker's actual
state, at startup and on a recurring schedule. See docs/SPECIFICATION.md
section 15.

Order-level reconciliation (resolving an ``UNKNOWN`` order, stale-order
refresh, submission-timeout handling, and the full post-reconnect sweep)
is ``execution.order_reconciler.OrderReconciler`` (Phase 17), not this
module -- that class is what ``reconcile_open_orders`` below would have
delegated to once implemented.

A mismatch must be quarantined, not silently resolved: the affected
instrument is frozen from new orders while the rest of the reconciled
portfolio continues to be managed normally. See docs/ARCHITECTURE.md.

Position/cash reconciliation itself is not implemented yet (Phase 11b).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum


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

    def reconcile_positions(self) -> ReconciliationReport:
        raise NotImplementedError("Phase 11b: reconciliation is not implemented yet.")

    def reconcile_open_orders(self) -> ReconciliationReport:
        """Once implemented, this should delegate to
        ``execution.order_reconciler.OrderReconciler.reconcile_after_reconnect``
        (Phase 17) and translate its ``OrderReconciliationReport`` into
        this module's ``ReconciliationReport``/``ReconciliationMismatch``
        shape, rather than re-implementing order-level reconciliation
        here."""
        raise NotImplementedError("Phase 11b: reconciliation is not implemented yet.")

    def quarantined_instruments(self) -> set[str]:
        """Instrument IDs currently frozen from new orders due to an
        unresolved mismatch.
        """
        raise NotImplementedError("Phase 11b: reconciliation is not implemented yet.")
