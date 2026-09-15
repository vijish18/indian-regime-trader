"""Reconciles local state (positions, cash, open orders) against the
broker's actual state, at startup and on a recurring schedule. See
docs/SPECIFICATION.md section 15.

A mismatch must be quarantined, not silently resolved: the affected
instrument is frozen from new orders while the rest of the reconciled
portfolio continues to be managed normally. See docs/ARCHITECTURE.md.

Not implemented yet (Phase 11b).
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
        raise NotImplementedError("Phase 11b: reconciliation is not implemented yet.")

    def quarantined_instruments(self) -> set[str]:
        """Instrument IDs currently frozen from new orders due to an
        unresolved mismatch.
        """
        raise NotImplementedError("Phase 11b: reconciliation is not implemented yet.")
