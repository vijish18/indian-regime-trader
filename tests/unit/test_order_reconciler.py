"""Unit and failure-injection integration tests for
``execution/order_reconciler.py`` (Phase 17).

The integration test at the bottom of this file
(``test_critical_scenario_...``) is the literal scenario Phase 17's brief
states as CRITICAL: a broker accepts an order but the response confirming
it is lost before this system ever sees it. It runs against a real
``broker.adapters.paper_broker.PaperBroker`` (real cost model, real
position tracker, real market data fake) wrapped in a broker double that
drops exactly one response after the real broker has already processed
the order underneath -- proving ``OrderManager``/``OrderReconciler``
recover the broker's true state rather than losing track of it or,
worse, resubmitting and creating a second position.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from decimal import Decimal

import pytest

from backtest.cost_schedule import CostScheduleRepository
from backtest.costs import CostModel
from broker.adapters.paper_broker import PaperBroker
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
from config.models import ExecutionConfig, PaperTradingConfig
from data.errors import DataNotAvailableError
from data.interfaces import MarketDataProvider
from data.models import DailyBar, IndexObservation, PriceBasis, Quote
from execution.execution_journal import JournalEventType
from execution.order_manager import OrderManager, OrderState
from execution.order_reconciler import OrderReconciler, RetryPolicy
from execution.position_tracker import PositionTracker
from tests.unit._wf_support import cost_schedule, execution_config

_T0 = dt.datetime(2024, 6, 3, 9, 30, tzinfo=dt.UTC)


class _ClockBox:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + dt.timedelta(**kwargs)


# --------------------------------------------------------------------------
# RetryPolicy -- safe to use only for read-only/idempotent operations
# --------------------------------------------------------------------------


def test_retry_policy_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="backoff_seconds"):
        RetryPolicy(backoff_seconds=-1.0)


def test_retry_policy_returns_the_result_on_first_success() -> None:
    policy = RetryPolicy(max_attempts=3, backoff_seconds=0.0)
    calls = []

    def operation() -> str:
        calls.append(1)
        return "ok"

    result = policy.execute_idempotent(operation, sleep=lambda _: None)
    assert result == "ok"
    assert len(calls) == 1


def test_retry_policy_retries_and_eventually_succeeds() -> None:
    policy = RetryPolicy(max_attempts=3, backoff_seconds=0.0)
    attempts = {"count": 0}

    def operation() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ConnectionError("simulated network blip")
        return "ok"

    result = policy.execute_idempotent(operation, sleep=lambda _: None)
    assert result == "ok"
    assert attempts["count"] == 3


def test_retry_policy_exhausts_attempts_and_raises_the_last_exception() -> None:
    policy = RetryPolicy(max_attempts=3, backoff_seconds=0.0)

    def operation() -> str:
        raise ConnectionError("still failing")

    with pytest.raises(ConnectionError, match="still failing"):
        policy.execute_idempotent(operation, sleep=lambda _: None)


def test_retry_policy_calls_on_retry_with_attempt_number_and_exception() -> None:
    policy = RetryPolicy(max_attempts=3, backoff_seconds=0.0)
    seen: list[tuple[int, Exception]] = []

    def operation() -> str:
        raise ConnectionError("boom")

    def on_retry(attempt: int, exc: Exception) -> None:
        seen.append((attempt, exc))

    with pytest.raises(ConnectionError):
        policy.execute_idempotent(operation, on_retry=on_retry, sleep=lambda _: None)
    assert [attempt for attempt, _exc in seen] == [1, 2, 3]


def test_retry_policy_sleeps_between_attempts_but_not_after_the_last_one() -> None:
    policy = RetryPolicy(max_attempts=3, backoff_seconds=0.25)
    sleeps: list[float] = []

    def operation() -> str:
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        policy.execute_idempotent(operation, sleep=sleeps.append)
    assert sleeps == [0.25, 0.25]  # 2 sleeps for 3 attempts, none after the final failure


# --------------------------------------------------------------------------
# OrderReconciler -- unit tests against a controllable stub broker
# --------------------------------------------------------------------------


class _StubBroker(Broker):
    def __init__(self) -> None:
        self.orders: dict[str, BrokerOrder] = {}
        self.open_orders_response: list[BrokerOrder] = []
        self.get_order_call_count: dict[str, int] = {}

    def capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        raise NotImplementedError

    def get_account(self) -> BrokerAccount:
        raise NotImplementedError

    def get_positions(self) -> list[BrokerPosition]:
        raise NotImplementedError

    def get_open_orders(self) -> list[BrokerOrder]:
        return list(self.open_orders_response)

    def get_order(self, order_id: str) -> BrokerOrder:
        self.get_order_call_count[order_id] = self.get_order_call_count.get(order_id, 0) + 1
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
        raise NotImplementedError("this stub is for reconciliation, not fresh submission")

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> BrokerOrder:
        raise NotImplementedError

    def close_position(self, instrument_id: str) -> BrokerOrder:
        raise NotImplementedError

    def close_all_positions(self) -> list[BrokerOrder]:
        raise NotImplementedError

    def health_check(self) -> HealthStatus:
        raise NotImplementedError


def _broker_order(client_order_id: str, status: str, broker_order_id: str = "B-1") -> BrokerOrder:
    return BrokerOrder(
        client_order_id=client_order_id,
        broker_order_id=broker_order_id,
        instrument_id="NSE:INFY",
        side="buy",
        quantity=10,
        order_type="limit",
        limit_price=1500.0,
        status=status,
        filled_quantity=10 if status == OrderState.FILLED.value else 0,
    )


@pytest.fixture
def clock() -> _ClockBox:
    return _ClockBox(_T0)


@pytest.fixture
def order_manager(clock: _ClockBox) -> OrderManager:
    return OrderManager(clock=clock)


def _created_order(order_manager: OrderManager, idempotency_key: str = "req-1") -> str:
    result = order_manager.create(
        "NSE:INFY",
        "buy",
        10,
        "limit",
        1500.0,
        idempotency_key=idempotency_key,
        signal_id=f"sig-{idempotency_key}",
        risk_decision_id=f"rd-{idempotency_key}",
    )
    return result.order.client_order_id


def test_resolve_unknown_is_a_noop_for_a_non_unknown_order(
    order_manager: OrderManager, clock: _ClockBox
) -> None:
    reconciler = OrderReconciler(clock=clock)
    broker = _StubBroker()
    client_order_id = _created_order(order_manager)
    order_manager.transition(client_order_id, OrderState.SUBMITTED)
    order_manager.transition(client_order_id, OrderState.OPEN)

    resolved = reconciler.resolve_unknown(order_manager, client_order_id, broker)
    assert resolved.state is OrderState.OPEN
    assert broker.get_order_call_count == {}  # never queried -- nothing to resolve


def test_resolve_unknown_queries_and_reconciles(
    order_manager: OrderManager, clock: _ClockBox
) -> None:
    reconciler = OrderReconciler(clock=clock)
    broker = _StubBroker()
    client_order_id = _created_order(order_manager)
    order_manager.transition(client_order_id, OrderState.SUBMITTED)
    order_manager.transition(client_order_id, OrderState.UNKNOWN)
    broker.orders[client_order_id] = _broker_order(client_order_id, OrderState.FILLED.value)

    resolved = reconciler.resolve_unknown(order_manager, client_order_id, broker)
    assert resolved.state is OrderState.FILLED
    assert resolved.broker_order_id == "B-1"

    journal_events = [e.event_type for e in reconciler.journal.for_client_order_id(client_order_id)]
    assert JournalEventType.RECONCILIATION_STARTED in journal_events
    assert JournalEventType.RECONCILIATION_RESOLVED in journal_events


def test_detect_and_handle_timeouts_finds_stuck_submitted_orders(
    order_manager: OrderManager, clock: _ClockBox
) -> None:
    reconciler = OrderReconciler(clock=clock, submitted_timeout_seconds=10.0)
    client_order_id = _created_order(order_manager)
    order_manager.transition(client_order_id, OrderState.SUBMITTED)

    clock.advance(seconds=11)
    timed_out = reconciler.detect_and_handle_timeouts(order_manager)

    assert [r.client_order_id for r in timed_out] == [client_order_id]
    assert order_manager.get(client_order_id).state is OrderState.UNKNOWN
    events = [e.event_type for e in reconciler.journal.for_client_order_id(client_order_id)]
    assert JournalEventType.ORDER_TIMEOUT_DETECTED in events


def test_detect_and_handle_timeouts_ignores_orders_within_the_window(
    order_manager: OrderManager, clock: _ClockBox
) -> None:
    reconciler = OrderReconciler(clock=clock, submitted_timeout_seconds=30.0)
    client_order_id = _created_order(order_manager)
    order_manager.transition(client_order_id, OrderState.SUBMITTED)

    clock.advance(seconds=5)
    timed_out = reconciler.detect_and_handle_timeouts(order_manager)
    assert timed_out == []
    assert order_manager.get(client_order_id).state is OrderState.SUBMITTED


def test_detect_and_handle_timeouts_ignores_non_submitted_orders(
    order_manager: OrderManager, clock: _ClockBox
) -> None:
    reconciler = OrderReconciler(clock=clock, submitted_timeout_seconds=1.0)
    client_order_id = _created_order(order_manager)
    order_manager.transition(client_order_id, OrderState.SUBMITTED)
    order_manager.transition(client_order_id, OrderState.OPEN)

    clock.advance(seconds=5)
    timed_out = reconciler.detect_and_handle_timeouts(order_manager)
    assert timed_out == []


def test_detect_stale_orders_finds_open_orders_past_the_threshold(
    order_manager: OrderManager, clock: _ClockBox
) -> None:
    reconciler = OrderReconciler(clock=clock, stale_after_seconds=60.0)
    client_order_id = _created_order(order_manager)
    order_manager.transition(client_order_id, OrderState.SUBMITTED)
    order_manager.transition(client_order_id, OrderState.OPEN)

    clock.advance(seconds=61)
    stale = reconciler.detect_stale_orders(order_manager)
    assert [r.client_order_id for r in stale] == [client_order_id]
    events = [e.event_type for e in reconciler.journal.for_client_order_id(client_order_id)]
    assert JournalEventType.STALE_ORDER_DETECTED in events


def test_refresh_from_broker_updates_local_state(
    order_manager: OrderManager, clock: _ClockBox
) -> None:
    reconciler = OrderReconciler(clock=clock)
    broker = _StubBroker()
    client_order_id = _created_order(order_manager)
    order_manager.transition(client_order_id, OrderState.SUBMITTED)
    order_manager.transition(client_order_id, OrderState.OPEN)
    broker.orders[client_order_id] = _broker_order(client_order_id, OrderState.FILLED.value)

    refreshed = reconciler.refresh_from_broker(order_manager, client_order_id, broker)
    assert refreshed.state is OrderState.FILLED
    assert order_manager.get(client_order_id).state is OrderState.FILLED


def test_refresh_from_broker_retries_a_transient_failure(
    order_manager: OrderManager, clock: _ClockBox
) -> None:
    retry_policy = RetryPolicy(max_attempts=3, backoff_seconds=0.0)
    reconciler = OrderReconciler(clock=clock, retry_policy=retry_policy)
    broker = _StubBroker()
    client_order_id = _created_order(order_manager)
    order_manager.transition(client_order_id, OrderState.SUBMITTED)
    order_manager.transition(client_order_id, OrderState.OPEN)
    broker.orders[client_order_id] = _broker_order(client_order_id, OrderState.FILLED.value)

    call_count = {"n": 0}
    real_get_order = broker.get_order

    def flaky_get_order(order_id: str) -> BrokerOrder:
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise ConnectionError("transient")
        return real_get_order(order_id)

    broker.get_order = flaky_get_order  # type: ignore[method-assign]
    refreshed = reconciler.refresh_from_broker(order_manager, client_order_id, broker)
    assert refreshed.state is OrderState.FILLED
    assert call_count["n"] == 2


def test_find_orphaned_orders_flags_broker_orders_with_no_local_record(
    order_manager: OrderManager, clock: _ClockBox
) -> None:
    reconciler = OrderReconciler(clock=clock)
    broker = _StubBroker()
    orphan_order = _broker_order("unknown-to-us", OrderState.OPEN.value, "B-ORPHAN")
    broker.open_orders_response = [orphan_order]

    orphans = reconciler.find_orphaned_orders(order_manager, broker)
    assert [o.broker_order_id for o in orphans] == ["B-ORPHAN"]
    events = [e.event_type for e in reconciler.journal.entries()]
    assert JournalEventType.ORPHAN_ORDER_DETECTED in events


def test_find_orphaned_orders_excludes_known_orders(
    order_manager: OrderManager, clock: _ClockBox
) -> None:
    reconciler = OrderReconciler(clock=clock)
    broker = _StubBroker()
    client_order_id = _created_order(order_manager)
    order_manager.transition(client_order_id, OrderState.SUBMITTED)
    known = order_manager.transition(client_order_id, OrderState.OPEN, broker_order_id="B-KNOWN")
    broker.open_orders_response = [_broker_order(client_order_id, OrderState.OPEN.value, "B-KNOWN")]

    orphans = reconciler.find_orphaned_orders(order_manager, broker)
    assert orphans == []
    assert known.broker_order_id == "B-KNOWN"


def test_reconcile_after_reconnect_runs_the_full_sweep(
    order_manager: OrderManager, clock: _ClockBox
) -> None:
    reconciler = OrderReconciler(
        clock=clock, stale_after_seconds=60.0, submitted_timeout_seconds=10.0
    )
    broker = _StubBroker()

    # An order stuck SUBMITTED past the timeout.
    timeout_id = _created_order(order_manager, "req-timeout")
    order_manager.transition(timeout_id, OrderState.SUBMITTED)
    broker.orders[timeout_id] = _broker_order(timeout_id, OrderState.FILLED.value, "B-TIMEOUT")

    # An order that is already UNKNOWN from a prior lost response.
    unknown_id = _created_order(order_manager, "req-unknown")
    order_manager.transition(unknown_id, OrderState.SUBMITTED)
    order_manager.transition(unknown_id, OrderState.UNKNOWN)
    broker.orders[unknown_id] = _broker_order(unknown_id, OrderState.OPEN.value, "B-UNKNOWN")

    # A stale but locally-OPEN order that actually got cancelled broker-side.
    stale_id = _created_order(order_manager, "req-stale")
    order_manager.transition(stale_id, OrderState.SUBMITTED)
    order_manager.transition(stale_id, OrderState.OPEN, broker_order_id="B-STALE")
    broker.orders[stale_id] = _broker_order(stale_id, OrderState.CANCELLED.value, "B-STALE")

    # An orphan the broker reports with no local record at all.
    broker.open_orders_response = [_broker_order("orphan", OrderState.OPEN.value, "B-ORPHAN")]

    clock.advance(seconds=61)
    report = reconciler.reconcile_after_reconnect(order_manager, broker)

    assert {r.client_order_id for r in report.timed_out} == {timeout_id}
    assert {r.client_order_id for r in report.resolved_unknown} == {timeout_id, unknown_id}
    assert {r.client_order_id for r in report.refreshed_stale} == {stale_id}
    assert [o.broker_order_id for o in report.orphaned_broker_orders] == ["B-ORPHAN"]

    assert order_manager.get(timeout_id).state is OrderState.FILLED
    assert order_manager.get(unknown_id).state is OrderState.OPEN
    assert order_manager.get(stale_id).state is OrderState.CANCELLED


# --------------------------------------------------------------------------
# CRITICAL failure-injection integration test -- Phase 17's own scenario,
# run against a real PaperBroker (real cost model, real position tracker),
# not a stub.
# --------------------------------------------------------------------------


class _FakeLiveMarketData(MarketDataProvider):
    def __init__(self) -> None:
        self._quotes: dict[str, Quote] = {}
        self._bars: dict[str, list[DailyBar]] = {}

    def set_quote(self, quote: Quote) -> None:
        self._quotes[quote.instrument_id] = quote

    def get_equity_bars(
        self,
        instrument_id: str,
        start: dt.date,
        end: dt.date,
        price_basis: PriceBasis = PriceBasis.RAW,
    ) -> list[DailyBar]:
        bars = self._bars.get(instrument_id, [])
        return [bar for bar in bars if start <= bar.session_date <= end]

    def get_index_observations(
        self, index_symbol: str, start: dt.date, end: dt.date
    ) -> list[IndexObservation]:
        return []

    def available_range(self, instrument_id: str) -> tuple[dt.date, dt.date] | None:
        bars = self._bars.get(instrument_id)
        if not bars:
            return None
        return bars[0].session_date, bars[-1].session_date

    def get_quote(self, instrument_id: str) -> Quote:
        try:
            return self._quotes[instrument_id]
        except KeyError:
            raise DataNotAvailableError(f"no quote for {instrument_id}") from None


def _make_quote(ask_quantity: int = 1000) -> Quote:
    return Quote(
        instrument_id="NSE:INFY",
        bid=Decimal("99.70"),
        ask=Decimal("100.30"),
        last_price=Decimal("100.00"),
        as_of=_T0,
        ask_quantity=ask_quantity,
    )


def _make_paper_broker(clock: _ClockBox) -> PaperBroker:
    market_data = _FakeLiveMarketData()
    market_data.set_quote(_make_quote())
    cost_model = CostModel(
        CostScheduleRepository([cost_schedule()]), min_slippage_bps=2.0, impact_coefficient=10.0
    )
    exec_config: ExecutionConfig = execution_config(
        order_price_guard_bps=50, stale_quote_seconds=15
    )
    paper_config = PaperTradingConfig.model_validate(
        {
            "submission_latency_ms": 0,
            "max_fill_participation_pct": 1.0,
            "order_expiry_seconds": 3600,
            "default_avg_daily_value_inr": 0.0,
            "default_volatility": 0.30,
        }
    )
    return PaperBroker(
        market_data=market_data,
        cost_model=cost_model,
        execution_config=exec_config,
        paper_config=paper_config,
        position_tracker=PositionTracker(),
        initial_cash=1_000_000.0,
        clock=clock,
    )


class _LostResponseBroker(Broker):
    """Wraps a real ``Broker``. Drops the response to exactly one
    ``place_order`` call for a chosen ``client_order_id`` -- the order is
    actually processed underneath (a real fill happens), the caller just
    never learns the outcome from that specific call. Every other call
    passes straight through.
    """

    def __init__(self, inner: Broker, fail_client_order_id: str) -> None:
        self._inner = inner
        self._fail_for = fail_client_order_id
        self._already_failed = False
        self.place_order_call_count = 0

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        self.place_order_call_count += 1
        if order.client_order_id == self._fail_for and not self._already_failed:
            self._already_failed = True
            self._inner.place_order(order)  # the broker DOES process it
            raise TimeoutError("simulated network timeout waiting for broker ack")
        return self._inner.place_order(order)

    def capabilities(self) -> BrokerCapabilities:
        return self._inner.capabilities()

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        self._inner.authenticate(credentials)

    def get_account(self) -> BrokerAccount:
        return self._inner.get_account()

    def get_positions(self) -> list[BrokerPosition]:
        return self._inner.get_positions()

    def get_open_orders(self) -> list[BrokerOrder]:
        return self._inner.get_open_orders()

    def get_order(self, order_id: str) -> BrokerOrder:
        return self._inner.get_order(order_id)

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        return self._inner.get_trades(order_id)

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        return self._inner.get_quotes(instrument_ids)

    def subscribe_market_data(
        self, instrument_ids: list[str], on_tick: Callable[[BrokerQuote], None]
    ) -> Callable[[], None]:
        return self._inner.subscribe_market_data(instrument_ids, on_tick)

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        return self._inner.modify_order(order_id, changes)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        return self._inner.cancel_order(order_id)

    def close_position(self, instrument_id: str) -> BrokerOrder:
        return self._inner.close_position(instrument_id)

    def close_all_positions(self) -> list[BrokerOrder]:
        return self._inner.close_all_positions()

    def health_check(self) -> HealthStatus:
        return self._inner.health_check()


def test_critical_scenario_broker_accepts_order_but_response_is_lost() -> None:
    """The exact scenario Phase 17's brief states as CRITICAL: a broker
    accepts an order but times out before returning confirmation.

    Verifies, in order:
    1. OrderManager.submit does not raise -- it marks the order UNKNOWN.
    2. Exactly one place_order call reached the broker (no automatic
       resubmission).
    3. The order genuinely was filled broker-side already (the real
       PaperBroker's position/cash actually changed) -- proving the
       ambiguity is real, not a rejection this system could have
       inferred on its own.
    4. Before reconciliation, the local record is UNKNOWN -- a caller
       checking state has no way to mistake this for a known outcome
       ("continue only after reconciliation").
    5. OrderReconciler.resolve_unknown determines the actual state by
       querying the broker -- never guessing, never retrying the write.
    6. After reconciliation, exactly one fill/position exists -- the
       ambiguous submission was never duplicated.
    """
    clock = _ClockBox(_T0)
    paper_broker = _make_paper_broker(clock)
    order_manager = OrderManager(clock=clock)

    created = order_manager.create(
        "NSE:INFY",
        "buy",
        10,
        "limit",
        100.30,
        idempotency_key="strategy-signal-1",
        signal_id="sig-1",
        risk_decision_id="rd-1",
    )
    client_order_id = created.order.client_order_id
    flaky_broker = _LostResponseBroker(paper_broker, fail_client_order_id=client_order_id)

    # Step 1 & 2: submit -- must not raise, and the broker is called once.
    record = order_manager.submit(client_order_id, flaky_broker)
    assert record.state is OrderState.UNKNOWN
    assert flaky_broker.place_order_call_count == 1

    # Step 3: the broker actually filled the order underneath, unseen by
    # the client -- this is what makes the scenario genuinely ambiguous.
    assert paper_broker.position_tracker.held_quantity("NSE:INFY") == 10
    assert len(paper_broker.fills) == 1

    # Step 4: local state is UNKNOWN, not silently assumed successful.
    assert order_manager.get(client_order_id).state is OrderState.UNKNOWN

    # Step 5: reconcile -- query the broker, never resubmit.
    reconciler = OrderReconciler(clock=clock)
    resolved = reconciler.resolve_unknown(order_manager, client_order_id, flaky_broker)

    assert resolved.state is OrderState.FILLED
    assert resolved.filled_quantity == 10
    assert resolved.broker_order_id is not None

    # Step 6: still exactly one place_order call and one fill -- the
    # order was never duplicated by the reconciliation process either.
    assert flaky_broker.place_order_call_count == 1
    assert len(paper_broker.fills) == 1
    assert paper_broker.position_tracker.held_quantity("NSE:INFY") == 10

    # And the full chain is traceable end to end.
    trace = order_manager.journal.trace(client_order_id)
    assert trace.signal_id == "sig-1"
    assert trace.risk_decision_id == "rd-1"
    assert trace.broker_order_id == resolved.broker_order_id
    assert len(trace.fills()) == 1


def test_critical_scenario_a_second_reconciliation_pass_is_a_harmless_noop() -> None:
    """Resolving an already-resolved order must never re-touch the
    broker -- reconciliation itself must be safe to run repeatedly."""
    clock = _ClockBox(_T0)
    paper_broker = _make_paper_broker(clock)
    order_manager = OrderManager(clock=clock)
    created = order_manager.create(
        "NSE:INFY",
        "buy",
        10,
        "limit",
        100.30,
        idempotency_key="strategy-signal-1",
        signal_id="sig-1",
        risk_decision_id="rd-1",
    )
    client_order_id = created.order.client_order_id
    flaky_broker = _LostResponseBroker(paper_broker, fail_client_order_id=client_order_id)
    order_manager.submit(client_order_id, flaky_broker)

    reconciler = OrderReconciler(clock=clock)
    first = reconciler.resolve_unknown(order_manager, client_order_id, flaky_broker)
    second = reconciler.resolve_unknown(order_manager, client_order_id, flaky_broker)

    assert first == second
    assert flaky_broker.place_order_call_count == 1
