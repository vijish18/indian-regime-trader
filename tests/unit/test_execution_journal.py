"""Unit tests for ``execution/execution_journal.py`` (Phase 17): the
append-only, queryable record behind "every signal must be traceable".
"""

from __future__ import annotations

import datetime as dt

import pytest

from execution.execution_journal import ExecutionJournal, JournalEventType

_T0 = dt.datetime(2024, 6, 3, 9, 0, 0, tzinfo=dt.UTC)


class _ClockBox:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + dt.timedelta(**kwargs)


def test_record_appends_and_returns_the_entry() -> None:
    journal = ExecutionJournal(clock=lambda: _T0)
    entry = journal.record(
        JournalEventType.ORDER_CREATED,
        signal_id="sig-1",
        risk_decision_id="rd-1",
        client_order_id="c-1",
        detail="buy 10 NSE:INFY",
    )
    assert entry.sequence == 1
    assert entry.event_type is JournalEventType.ORDER_CREATED
    assert entry.as_of == _T0
    assert journal.entries() == (entry,)


def test_sequence_is_monotonically_increasing() -> None:
    journal = ExecutionJournal(clock=lambda: _T0)
    first = journal.record(JournalEventType.ORDER_CREATED, client_order_id="c-1")
    second = journal.record(JournalEventType.STATE_CHANGED, client_order_id="c-1")
    third = journal.record(JournalEventType.STATE_CHANGED, client_order_id="c-2")
    assert [e.sequence for e in (first, second, third)] == [1, 2, 3]


def test_for_client_order_id_filters_correctly() -> None:
    journal = ExecutionJournal(clock=lambda: _T0)
    journal.record(JournalEventType.ORDER_CREATED, client_order_id="c-1")
    journal.record(JournalEventType.ORDER_CREATED, client_order_id="c-2")
    journal.record(JournalEventType.STATE_CHANGED, client_order_id="c-1")

    entries = journal.for_client_order_id("c-1")
    assert len(entries) == 2
    assert all(e.client_order_id == "c-1" for e in entries)


def test_for_signal_filters_correctly() -> None:
    journal = ExecutionJournal(clock=lambda: _T0)
    journal.record(JournalEventType.ORDER_CREATED, signal_id="sig-1", client_order_id="c-1")
    journal.record(JournalEventType.ORDER_CREATED, signal_id="sig-2", client_order_id="c-2")

    entries = journal.for_signal("sig-1")
    assert len(entries) == 1
    assert entries[0].client_order_id == "c-1"


def test_trace_reconstructs_the_identity_chain() -> None:
    journal = ExecutionJournal(clock=lambda: _T0)
    journal.record(
        JournalEventType.ORDER_CREATED,
        signal_id="sig-1",
        risk_decision_id="rd-1",
        client_order_id="c-1",
        detail="buy 10 NSE:INFY",
    )
    journal.record(
        JournalEventType.STATE_CHANGED,
        signal_id="sig-1",
        risk_decision_id="rd-1",
        client_order_id="c-1",
        broker_order_id="B-1",
        detail="submitted -> open",
    )
    journal.record(
        JournalEventType.FILL_OBSERVED,
        signal_id="sig-1",
        risk_decision_id="rd-1",
        client_order_id="c-1",
        broker_order_id="B-1",
        detail="filled_quantity 0 -> 10",
    )

    trace = journal.trace("c-1")
    assert trace.client_order_id == "c-1"
    assert trace.signal_id == "sig-1"
    assert trace.risk_decision_id == "rd-1"
    assert trace.broker_order_id == "B-1"
    assert len(trace.events) == 3
    assert len(trace.fills()) == 1


def test_trace_raises_for_an_unknown_client_order_id() -> None:
    journal = ExecutionJournal(clock=lambda: _T0)
    with pytest.raises(KeyError, match="does-not-exist"):
        journal.trace("does-not-exist")


def test_trace_picks_up_the_broker_order_id_even_if_only_a_later_entry_carries_it() -> None:
    """The client_order_id is known from the first entry; the
    broker_order_id typically isn't known until a later one (the broker
    hasn't responded yet at creation time)."""
    journal = ExecutionJournal(clock=lambda: _T0)
    journal.record(JournalEventType.ORDER_CREATED, client_order_id="c-1")
    journal.record(JournalEventType.STATE_CHANGED, client_order_id="c-1", broker_order_id="B-1")

    trace = journal.trace("c-1")
    assert trace.broker_order_id == "B-1"


def test_entries_are_immutable_and_isolated_from_internal_state() -> None:
    journal = ExecutionJournal(clock=lambda: _T0)
    journal.record(JournalEventType.ORDER_CREATED, client_order_id="c-1")
    entries = journal.entries()
    journal.record(JournalEventType.STATE_CHANGED, client_order_id="c-1")
    # The tuple returned earlier does not grow when the journal does.
    assert len(entries) == 1
    assert len(journal.entries()) == 2
