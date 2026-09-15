"""An append-only, structured record of every order-lifecycle event, so
a signal is always traceable end to end: ``signal_id -> risk_decision_id
-> client_order_id -> broker_order_id -> fills -> position`` (Phase 17).

``execution.order_manager.OrderManager`` writes to this on every
create/transition/duplicate-suppression -- a caller never has to
remember to log an event by hand, and can never accidentally produce an
order with no audit trail. Every entry is also emitted through the
standard ``logging`` module with a structured ``extra_fields`` payload,
the same convention ``risk/circuit_breaker.py`` and
``broker/compliance.py`` already use, so this journal's content reaches
whatever external log pipeline a deployment already has -- this class
itself is an in-memory, queryable index over the *current run*, not a
persistent store (see "What's not here" below).

**"fills" and "position" in the traceability chain.** A ``FILL_OBSERVED``
entry is written whenever ``OrderManager.transition`` reports a higher
``filled_quantity`` than the order previously had. The chain's final
link, "fills -> position", is satisfied structurally rather than
re-derived here: every fill this journal observes is the same fill
``broker.adapters.paper_broker.PaperBroker``/``broker.zerodha.kite_broker.KiteBroker``
already applies to ``execution.position_tracker.PositionTracker`` (Phase
14/15) -- reconstructing "which position resulted from which order" as
an automated query would mean correlating this journal against
``PositionTracker``'s own state by timestamp and instrument, which this
phase does not build; the two are linkable by a human reading both, not
yet by one call.

**What's not here.** No persistence -- this is an in-memory list for the
lifetime of the process (or test). A production deployment durably
persisting this journal (a database, an append-only file) is
``storage/``'s job in a later phase, not invented here.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class JournalEventType(StrEnum):
    ORDER_CREATED = "order_created"
    DUPLICATE_SUPPRESSED = "duplicate_suppressed"
    DUPLICATE_SIGNAL_REJECTED = "duplicate_signal_rejected"
    STATE_CHANGED = "state_changed"
    FILL_OBSERVED = "fill_observed"
    RECONCILIATION_STARTED = "reconciliation_started"
    RECONCILIATION_RESOLVED = "reconciliation_resolved"
    STALE_ORDER_DETECTED = "stale_order_detected"
    ORDER_TIMEOUT_DETECTED = "order_timeout_detected"
    ORPHAN_ORDER_DETECTED = "orphan_order_detected"
    RETRY_ATTEMPTED = "retry_attempted"
    RETRY_EXHAUSTED = "retry_exhausted"


@dataclass(frozen=True, slots=True)
class JournalEntry:
    sequence: int
    """Monotonically increasing within one journal -- the total order
    every entry was actually recorded in, independent of (and a
    tie-breaker for) ``as_of``, which two entries can share if the
    injected clock has coarser resolution than real time."""

    event_type: JournalEventType
    as_of: dt.datetime
    signal_id: str | None
    risk_decision_id: str | None
    client_order_id: str | None
    broker_order_id: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class OrderTrace:
    """The reconstructed identity chain for one order, built from every
    journal entry that ever carried its ``client_order_id`` -- see this
    module's docstring for what "fills" and "position" mean here.
    """

    client_order_id: str
    signal_id: str | None
    risk_decision_id: str | None
    broker_order_id: str | None
    events: tuple[JournalEntry, ...]

    def fills(self) -> tuple[JournalEntry, ...]:
        return tuple(e for e in self.events if e.event_type is JournalEventType.FILL_OBSERVED)


class ExecutionJournal:
    """An append-only, in-process, queryable record of order-lifecycle
    events. See this module's docstring for what is and is not
    persisted.
    """

    def __init__(self, clock: Callable[[], dt.datetime] | None = None) -> None:
        self._clock: Callable[[], dt.datetime] = clock or dt.datetime.now
        self._entries: list[JournalEntry] = []
        self._sequence = 0

    def record(
        self,
        event_type: JournalEventType,
        *,
        signal_id: str | None = None,
        risk_decision_id: str | None = None,
        client_order_id: str | None = None,
        broker_order_id: str | None = None,
        detail: str = "",
    ) -> JournalEntry:
        self._sequence += 1
        entry = JournalEntry(
            sequence=self._sequence,
            event_type=event_type,
            as_of=self._clock(),
            signal_id=signal_id,
            risk_decision_id=risk_decision_id,
            client_order_id=client_order_id,
            broker_order_id=broker_order_id,
            detail=detail,
        )
        self._entries.append(entry)
        logger.info(
            event_type.value,
            extra={
                "extra_fields": {
                    "event": "execution_journal",
                    "sequence": entry.sequence,
                    "event_type": event_type.value,
                    "signal_id": signal_id,
                    "risk_decision_id": risk_decision_id,
                    "client_order_id": client_order_id,
                    "broker_order_id": broker_order_id,
                    "detail": detail,
                }
            },
        )
        return entry

    def entries(self) -> tuple[JournalEntry, ...]:
        return tuple(self._entries)

    def for_client_order_id(self, client_order_id: str) -> tuple[JournalEntry, ...]:
        return tuple(e for e in self._entries if e.client_order_id == client_order_id)

    def for_signal(self, signal_id: str) -> tuple[JournalEntry, ...]:
        return tuple(e for e in self._entries if e.signal_id == signal_id)

    def trace(self, client_order_id: str) -> OrderTrace:
        """Reconstructs the full identity chain for one order. Raises
        ``KeyError`` if this journal has never seen the ID -- there is no
        such thing as an empty trace, only a missing one.
        """
        events = self.for_client_order_id(client_order_id)
        if not events:
            raise KeyError(f"no journal entries for client_order_id {client_order_id!r}")
        signal_id = next((e.signal_id for e in events if e.signal_id is not None), None)
        risk_decision_id = next(
            (e.risk_decision_id for e in events if e.risk_decision_id is not None), None
        )
        broker_order_id = next(
            (e.broker_order_id for e in events if e.broker_order_id is not None), None
        )
        return OrderTrace(
            client_order_id=client_order_id,
            signal_id=signal_id,
            risk_decision_id=risk_decision_id,
            broker_order_id=broker_order_id,
            events=events,
        )
