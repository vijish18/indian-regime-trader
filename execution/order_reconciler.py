"""Reconciles ``execution.order_manager.OrderManager``'s local order
state against a broker's actual state (Phase 17). Builds on top of
``OrderManager.handle_ambiguous_response`` (Phase 14) -- resolving one
``UNKNOWN`` order by querying the broker -- with the fuller set of
protections a production order-management system needs:

- **stale-order detection**: an ``OPEN``/``PARTIALLY_FILLED`` order this
  system hasn't heard about in a while is refreshed from the broker
  rather than trusted indefinitely.
- **order timeout**: an order stuck in ``SUBMITTED`` (never acknowledged
  as open, filled, or rejected) past a configured age is treated exactly
  like a submission that raised -- marked ``UNKNOWN`` and resolved by
  querying the broker, never assumed to have succeeded or silently
  retried.
- **broker reconnect**: :meth:`OrderReconciler.reconcile_after_reconnect`
  is the one entry point a live/paper trading loop calls after
  (re)establishing a broker connection -- it resolves every ``UNKNOWN``,
  refreshes every stale/timed-out order, and surfaces orders the broker
  reports that this system has no record of. It never resubmits
  anything.
- **retry policy, safely scoped**: :class:`RetryPolicy` retries read-only
  broker queries (``get_order``, ``get_open_orders``) a bounded number of
  times with backoff -- it is never used for ``place_order``'s initial
  submission, which ``OrderManager.submit`` deliberately does not retry
  at all (see that module's docstring, "no unsafe blind retry"). A query
  can be retried because asking twice changes nothing; placing an order
  twice can create a second position.

**The CRITICAL scenario this module exists for**: a broker accepts an
order but the response confirming it (with the broker's own order ID) is
lost before this system ever sees it -- a timeout, a dropped connection,
a crashed process. ``OrderManager.submit`` already marks the order
``UNKNOWN`` rather than assuming failure and retrying; ``OrderReconciler``
is what completes the sequence the phase 17 brief requires verbatim:
mark ``UNKNOWN`` -> query the broker -> reconcile -> determine the
actual state -> continue only after reconciliation. No code path in
either module ever re-submits the same order.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from broker.base import Broker, BrokerOrder
from execution.execution_journal import ExecutionJournal, JournalEventType
from execution.order_manager import OrderManager, OrderRecord, OrderState

_T = TypeVar("_T")

_DEFAULT_STALE_AFTER_SECONDS = 300.0
_DEFAULT_SUBMITTED_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BACKOFF_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retries a read-only or broker-deduplicated operation up to
    ``max_attempts`` times with a fixed backoff between attempts.

    **Never use this for an order's initial submission.** Retrying
    ``place_order`` blindly is exactly the unsafe behavior this phase
    forbids -- ``OrderManager.submit`` does not use this class at all;
    it marks an ambiguous submission ``UNKNOWN`` and leaves resolution to
    :meth:`OrderReconciler.resolve_unknown`, which itself only retries
    the read (``broker.get_order``), never the original write.
    """

    max_attempts: int = _DEFAULT_MAX_ATTEMPTS
    backoff_seconds: float = _DEFAULT_BACKOFF_SECONDS

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.backoff_seconds < 0:
            raise ValueError(f"backoff_seconds must be >= 0, got {self.backoff_seconds}")

    def execute_idempotent(
        self,
        operation: Callable[[], _T],
        *,
        on_retry: Callable[[int, Exception], None] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> _T:
        sleep_fn = sleep or time.sleep
        last_exception: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation()
            except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure retries
                last_exception = exc
                if on_retry is not None:
                    on_retry(attempt, exc)
                if attempt < self.max_attempts:
                    sleep_fn(self.backoff_seconds)
        assert last_exception is not None  # loop always sets this before falling through
        raise last_exception


@dataclass(frozen=True, slots=True)
class OrderReconciliationReport:
    as_of: dt.datetime
    resolved_unknown: tuple[OrderRecord, ...]
    timed_out: tuple[OrderRecord, ...]
    refreshed_stale: tuple[OrderRecord, ...]
    orphaned_broker_orders: tuple[BrokerOrder, ...]
    """Orders the broker reports as open that this ``OrderManager`` has no
    record of -- from a previous process, a manually-placed order, or a
    lost local-state file. Surfaced for manual review, never acted on
    automatically: this system cannot safely manage an order whose
    signal/risk lineage it does not know."""


class OrderReconciler:
    def __init__(
        self,
        journal: ExecutionJournal | None = None,
        retry_policy: RetryPolicy | None = None,
        clock: Callable[[], dt.datetime] | None = None,
        stale_after_seconds: float = _DEFAULT_STALE_AFTER_SECONDS,
        submitted_timeout_seconds: float = _DEFAULT_SUBMITTED_TIMEOUT_SECONDS,
    ) -> None:
        self._clock: Callable[[], dt.datetime] = clock or dt.datetime.now
        self.journal = journal or ExecutionJournal(clock=self._clock)
        self.retry_policy = retry_policy or RetryPolicy()
        self.stale_after_seconds = stale_after_seconds
        self.submitted_timeout_seconds = submitted_timeout_seconds

    def resolve_unknown(
        self, order_manager: OrderManager, client_order_id: str, broker: Broker
    ) -> OrderRecord:
        """Resolves one ``UNKNOWN`` order: query the broker (through the
        retry policy -- a read is safe to repeat), reconcile local state
        to match. Never re-submits the order. A no-op if the order is not
        (or is no longer) ``UNKNOWN``.
        """
        record = order_manager.get(client_order_id)
        if record.state is not OrderState.UNKNOWN:
            return record

        self.journal.record(
            JournalEventType.RECONCILIATION_STARTED,
            signal_id=record.signal_id,
            risk_decision_id=record.risk_decision_id,
            client_order_id=client_order_id,
            detail="resolving UNKNOWN order by querying the broker",
        )
        resolved = order_manager.handle_ambiguous_response(client_order_id, broker)
        self.journal.record(
            JournalEventType.RECONCILIATION_RESOLVED,
            signal_id=resolved.signal_id,
            risk_decision_id=resolved.risk_decision_id,
            client_order_id=client_order_id,
            broker_order_id=resolved.broker_order_id,
            detail=f"resolved to {resolved.state.value}",
        )
        return resolved

    def detect_and_handle_timeouts(self, order_manager: OrderManager) -> list[OrderRecord]:
        """An order stuck in ``SUBMITTED`` longer than
        ``submitted_timeout_seconds`` never received (or this system
        never observed) its acknowledgement -- treated as ambiguous,
        exactly like a submission call that raised, rather than assumed
        successful or silently retried.
        """
        now = self._clock()
        timed_out = []
        for record in order_manager.all_orders():
            if record.state is not OrderState.SUBMITTED:
                continue
            age = (now - record.updated_at).total_seconds()
            if age < self.submitted_timeout_seconds:
                continue
            self.journal.record(
                JournalEventType.ORDER_TIMEOUT_DETECTED,
                signal_id=record.signal_id,
                risk_decision_id=record.risk_decision_id,
                client_order_id=record.client_order_id,
                detail=(
                    f"stuck in SUBMITTED for {age:.1f}s "
                    f"(timeout={self.submitted_timeout_seconds}s)"
                ),
            )
            new_unknown = order_manager.transition(record.client_order_id, OrderState.UNKNOWN)
            timed_out.append(new_unknown)
        return timed_out

    def detect_stale_orders(self, order_manager: OrderManager) -> list[OrderRecord]:
        """``OPEN``/``PARTIALLY_FILLED`` orders this system hasn't heard
        an update about in a while -- flagged, not assumed still
        accurate. Pair with :meth:`refresh_from_broker` for each to
        actually resync local state.
        """
        now = self._clock()
        stale = []
        for record in order_manager.all_orders():
            if record.state not in (OrderState.OPEN, OrderState.PARTIALLY_FILLED):
                continue
            age = (now - record.updated_at).total_seconds()
            if age < self.stale_after_seconds:
                continue
            self.journal.record(
                JournalEventType.STALE_ORDER_DETECTED,
                signal_id=record.signal_id,
                risk_decision_id=record.risk_decision_id,
                client_order_id=record.client_order_id,
                detail=f"no update in {age:.1f}s (stale_after={self.stale_after_seconds}s)",
            )
            stale.append(record)
        return stale

    def refresh_from_broker(
        self, order_manager: OrderManager, client_order_id: str, broker: Broker
    ) -> OrderRecord:
        """Re-queries the broker for one order's current status and
        reconciles local state to match -- used for stale-order refresh
        and as part of :meth:`reconcile_after_reconnect`. Goes through
        the retry policy since this is a safe, idempotent read, never a
        resubmission.
        """
        broker_order = self.retry_policy.execute_idempotent(
            lambda: broker.get_order(client_order_id)
        )
        return order_manager.adopt_broker_order(client_order_id, broker_order)

    def find_orphaned_orders(
        self, order_manager: OrderManager, broker: Broker
    ) -> list[BrokerOrder]:
        """Orders the broker reports as open that this ``OrderManager``
        has no record of. Flagged for manual review; never acted on
        here."""
        known_broker_ids = {
            record.broker_order_id
            for record in order_manager.all_orders()
            if record.broker_order_id is not None
        }
        open_broker_orders = self.retry_policy.execute_idempotent(broker.get_open_orders)
        orphans = [
            order for order in open_broker_orders if order.broker_order_id not in known_broker_ids
        ]
        for orphan in orphans:
            self.journal.record(
                JournalEventType.ORPHAN_ORDER_DETECTED,
                broker_order_id=orphan.broker_order_id,
                detail=(
                    f"broker reports an open order this system has no record of: "
                    f"{orphan.instrument_id} ({orphan.side} {orphan.quantity})"
                ),
            )
        return orphans

    def reconcile_after_reconnect(
        self, order_manager: OrderManager, broker: Broker
    ) -> OrderReconciliationReport:
        """The full sweep a live/paper trading loop runs after
        (re)establishing a broker connection: times out stuck
        submissions, resolves every ``UNKNOWN`` order, refreshes every
        stale order, and surfaces orphans. Never resubmits anything --
        every step here either reads from the broker or applies a
        broker-confirmed state to local records.
        """
        now = self._clock()
        timed_out = self.detect_and_handle_timeouts(order_manager)

        unknown_ids = [
            record.client_order_id
            for record in order_manager.all_orders()
            if record.state is OrderState.UNKNOWN
        ]
        resolved_unknown = [
            self.resolve_unknown(order_manager, client_order_id, broker)
            for client_order_id in unknown_ids
        ]

        stale = self.detect_stale_orders(order_manager)
        refreshed = [
            self.refresh_from_broker(order_manager, record.client_order_id, broker)
            for record in stale
        ]

        orphans = self.find_orphaned_orders(order_manager, broker)

        return OrderReconciliationReport(
            as_of=now,
            resolved_unknown=tuple(resolved_unknown),
            timed_out=tuple(timed_out),
            refreshed_stale=tuple(refreshed),
            orphaned_broker_orders=tuple(orphans),
        )
