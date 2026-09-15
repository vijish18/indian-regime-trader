"""Unit and failure-injection integration tests for
``execution/startup.py`` (Phase 18): the full restart-recovery and
broker-reconciliation sequence, plus all named recovery scenarios from
the phase brief:

    clean restart, crash during order submission, crash after fill,
    database restart, broker disconnect, duplicate broker event,
    missing local record, unknown local order

Most scenarios use a fully controllable stub broker for precise,
deterministic control over exactly what "local" and "broker" each
believe. The crash-during-order-submission scenario is additionally
proven against a real ``broker.adapters.paper_broker.PaperBroker``
(mirroring Phase 17's own precedent for its CRITICAL scenario), so the
reconciliation this phase adds is shown working through the same real
fill/position mechanics a live run would actually use, not only through
a mock.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from decimal import Decimal
from pathlib import Path

import pytest

from backtest.cost_schedule import CostScheduleRepository
from backtest.costs import CostModel, TradeSide
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
from config.loader import ConfigError
from config.models import PaperTradingConfig
from core.regime.model_registry import ModelRegistry
from data.errors import DataNotAvailableError
from data.interfaces import MarketDataProvider
from data.models import DailyBar, IndexObservation, PriceBasis, Quote
from execution.order_manager import OrderManager, OrderState
from execution.position_tracker import PositionTracker
from execution.reconciliation import ReconciliationStatus
from execution.startup import StartupError, StartupSequence
from execution.system_state import APP_VERSION, STATE_SCHEMA_VERSION, SystemState, SystemStateStore
from risk.circuit_breaker import CircuitBreaker, CircuitState
from risk.portfolio_risk_state import PortfolioRiskState
from tests.unit._wf_support import cost_schedule, execution_config, risk_config

_T0 = dt.datetime(2024, 6, 3, 9, 30, tzinfo=dt.UTC)


class _ClockBox:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + dt.timedelta(**kwargs)


# --------------------------------------------------------------------------
# A fully controllable stub broker
# --------------------------------------------------------------------------


class _StubBroker(Broker):
    def __init__(self) -> None:
        self.healthy = True
        self.health_detail = "ok"
        self.positions_response: list[BrokerPosition] = []
        self.open_orders_response: list[BrokerOrder] = []
        self.trades_response: list[BrokerFill] = []
        self.orders: dict[str, BrokerOrder] = {}
        self.account_cash = 1_000_000.0
        self.get_positions_calls = 0

    def capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        raise NotImplementedError

    def get_account(self) -> BrokerAccount:
        return BrokerAccount(
            account_id="stub",
            equity=self.account_cash,
            cash=self.account_cash,
            buying_power=self.account_cash,
            as_of=_T0,
        )

    def get_positions(self) -> list[BrokerPosition]:
        self.get_positions_calls += 1
        return list(self.positions_response)

    def get_open_orders(self) -> list[BrokerOrder]:
        return list(self.open_orders_response)

    def get_order(self, order_id: str) -> BrokerOrder:
        try:
            return self.orders[order_id]
        except KeyError:
            raise RuntimeError(f"unknown order: {order_id}") from None

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        return list(self.trades_response)

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        raise NotImplementedError

    def subscribe_market_data(
        self, instrument_ids: list[str], on_tick: Callable[[BrokerQuote], None]
    ) -> Callable[[], None]:
        raise NotImplementedError

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        raise NotImplementedError

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> BrokerOrder:
        raise NotImplementedError

    def close_position(self, instrument_id: str) -> BrokerOrder:
        raise NotImplementedError

    def close_all_positions(self) -> list[BrokerOrder]:
        raise NotImplementedError

    def health_check(self) -> HealthStatus:
        return HealthStatus(
            healthy=self.healthy,
            detail=self.health_detail,
            checked_at=_T0,
            session_active=self.healthy,
            login_time=_T0 if self.healthy else None,
        )


def _make_sequence(
    tmp_path: Path,
    clock: _ClockBox,
    broker: Broker,
    *,
    position_tracker: PositionTracker | None = None,
    order_manager: OrderManager | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    model_registry: ModelRegistry | None = None,
    state_path: Path | None = None,
    settings_loader: Callable[[], object] | None = None,
) -> StartupSequence:
    kwargs: dict[str, object] = {}
    if settings_loader is not None:
        kwargs["settings_loader"] = settings_loader
    return StartupSequence(
        state_store=SystemStateStore(state_path or (tmp_path / "system_state.json")),
        position_tracker=position_tracker or PositionTracker(),
        order_manager=order_manager or OrderManager(clock=clock),
        broker=broker,
        circuit_breaker=circuit_breaker or CircuitBreaker(risk_config(), tmp_path / "cb.json"),
        model_registry=model_registry,
        strategy_version="strategy-v1",
        clock=clock,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def clock() -> _ClockBox:
    return _ClockBox(_T0)


# --------------------------------------------------------------------------
# 1. Clean restart
# --------------------------------------------------------------------------


def test_clean_restart_permits_strategy_execution(tmp_path: Path, clock: _ClockBox) -> None:
    broker = _StubBroker()
    report = _make_sequence(tmp_path, clock, broker).run()

    assert report.system_state is SystemState.READY
    assert report.permit_strategy_execution is True
    assert report.position_reconciliation is not None
    assert report.position_reconciliation.status is ReconciliationStatus.CLEAN
    assert report.order_reconciliation is not None
    assert report.order_reconciliation.status is ReconciliationStatus.CLEAN
    assert report.broker_connected is True
    assert report.database_verified is True
    assert report.schema_version_ok is True


def test_clean_restart_persists_state(tmp_path: Path, clock: _ClockBox) -> None:
    broker = _StubBroker()
    state_path = tmp_path / "system_state.json"
    _make_sequence(tmp_path, clock, broker, state_path=state_path).run()

    persisted = SystemStateStore(state_path).load()
    assert persisted is not None
    assert persisted.system_state is SystemState.READY
    assert persisted.app_version == APP_VERSION
    assert persisted.schema_version == STATE_SCHEMA_VERSION
    assert persisted.strategy_version == "strategy-v1"
    assert persisted.portfolio_snapshot is not None


def test_clean_restart_with_a_matching_position_on_both_sides_is_still_clean(
    tmp_path: Path, clock: _ClockBox
) -> None:
    broker = _StubBroker()
    broker.positions_response = [
        BrokerPosition(instrument_id="NSE:INFY", quantity=10, avg_price=1500.0)
    ]
    position_tracker = PositionTracker()
    position_tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)

    report = _make_sequence(tmp_path, clock, broker, position_tracker=position_tracker).run()
    assert report.system_state is SystemState.READY


# --------------------------------------------------------------------------
# 2. Crash during order submission -- OrderManager's in-memory record
# survives (e.g. a connectivity blip, not a full process kill), left in
# UNKNOWN; the broker already processed it. Resolved during startup,
# never resubmitted.
# --------------------------------------------------------------------------


def test_crash_during_order_submission_resolves_via_reconciliation(
    tmp_path: Path, clock: _ClockBox
) -> None:
    broker = _StubBroker()
    order_manager = OrderManager(clock=clock)
    created = order_manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0,
        idempotency_key="req-1", signal_id="sig-1", risk_decision_id="rd-1",
    )
    client_order_id = created.order.client_order_id
    order_manager.transition(client_order_id, OrderState.SUBMITTED)
    order_manager.transition(client_order_id, OrderState.UNKNOWN)  # the crash point

    broker.orders[client_order_id] = BrokerOrder(
        client_order_id=client_order_id,
        broker_order_id="B-1",
        instrument_id="NSE:INFY",
        side="buy",
        quantity=10,
        order_type="limit",
        limit_price=1500.0,
        status=OrderState.FILLED.value,
        filled_quantity=10,
        avg_fill_price=1499.5,
    )
    broker.positions_response = [
        BrokerPosition(instrument_id="NSE:INFY", quantity=10, avg_price=1499.5)
    ]
    position_tracker = PositionTracker()
    position_tracker.apply_fill("NSE:INFY", 10, 1499.5, TradeSide.BUY, _T0)

    report = _make_sequence(
        tmp_path, clock, broker, position_tracker=position_tracker, order_manager=order_manager
    ).run()

    assert order_manager.get(client_order_id).state is OrderState.FILLED
    assert report.system_state is SystemState.READY
    assert report.order_reconciliation is not None
    assert report.order_reconciliation.status is ReconciliationStatus.CLEAN


def test_critical_scenario_crash_during_order_submission_against_a_real_broker(
    tmp_path: Path, clock: _ClockBox
) -> None:
    """The literal Phase 17 CRITICAL scenario, now verified through the
    full Phase 18 startup sequence: a real PaperBroker fills the order,
    the response is lost, the process is treated as having 'restarted'
    with the same (surviving) OrderManager/PositionTracker, and startup
    must resolve it -- never resubmit, never silently ignore it.
    """
    market_data = _FakeLiveMarketData()
    market_data.set_quote(_make_quote())
    cost_model = CostModel(
        CostScheduleRepository([cost_schedule()]), min_slippage_bps=2.0, impact_coefficient=10.0
    )
    position_tracker = PositionTracker()
    paper_broker = PaperBroker(
        market_data=market_data,
        cost_model=cost_model,
        execution_config=execution_config(order_price_guard_bps=50, stale_quote_seconds=15),
        paper_config=PaperTradingConfig.model_validate(
            {
                "submission_latency_ms": 0,
                "max_fill_participation_pct": 1.0,
                "order_expiry_seconds": 3600,
                "default_avg_daily_value_inr": 0.0,
                "default_volatility": 0.30,
            }
        ),
        position_tracker=position_tracker,
        initial_cash=1_000_000.0,
        clock=clock,
    )
    order_manager = OrderManager(clock=clock)
    created = order_manager.create(
        "NSE:INFY", "buy", 10, "limit", 100.30,
        idempotency_key="req-1", signal_id="sig-1", risk_decision_id="rd-1",
    )
    client_order_id = created.order.client_order_id
    flaky_broker = _LostResponseBroker(paper_broker, fail_client_order_id=client_order_id)

    record = order_manager.submit(client_order_id, flaky_broker)
    assert record.state is OrderState.UNKNOWN
    assert flaky_broker.place_order_call_count == 1

    report = _make_sequence(
        tmp_path,
        clock,
        flaky_broker,
        position_tracker=position_tracker,
        order_manager=order_manager,
    ).run()

    assert order_manager.get(client_order_id).state is OrderState.FILLED
    assert flaky_broker.place_order_call_count == 1  # never resubmitted
    assert report.system_state is SystemState.READY


# --------------------------------------------------------------------------
# 3. Crash after fill -- a true process restart: the new process's
# PositionTracker/OrderManager start empty, and the broker already holds
# a position from before the crash. Must NOT be silently adopted.
# --------------------------------------------------------------------------


def test_crash_after_fill_with_no_surviving_local_record_requires_reconciliation(
    tmp_path: Path, clock: _ClockBox
) -> None:
    broker = _StubBroker()
    broker.positions_response = [
        BrokerPosition(instrument_id="NSE:INFY", quantity=10, avg_price=1499.5)
    ]
    # A fresh process: no local record survived the crash.
    report = _make_sequence(tmp_path, clock, broker).run()

    assert report.system_state is SystemState.RECONCILIATION_REQUIRED
    assert report.permit_strategy_execution is False
    assert report.position_reconciliation is not None
    assert report.position_reconciliation.status is ReconciliationStatus.QUARANTINED


# --------------------------------------------------------------------------
# 4. Database restart
# --------------------------------------------------------------------------


def test_database_restart_with_a_corrupted_state_file_raises(
    tmp_path: Path, clock: _ClockBox
) -> None:
    state_path = tmp_path / "system_state.json"
    state_path.write_text("{not valid json", encoding="utf-8")
    broker = _StubBroker()

    with pytest.raises(StartupError, match="state store verification failed"):
        _make_sequence(tmp_path, clock, broker, state_path=state_path).run()


def test_database_restart_recovers_once_the_file_is_repaired(
    tmp_path: Path, clock: _ClockBox
) -> None:
    state_path = tmp_path / "system_state.json"
    state_path.write_text("{not valid json", encoding="utf-8")
    broker = _StubBroker()
    with pytest.raises(StartupError):
        _make_sequence(tmp_path, clock, broker, state_path=state_path).run()

    state_path.unlink()
    report = _make_sequence(tmp_path, clock, broker, state_path=state_path).run()
    assert report.system_state is SystemState.READY


# --------------------------------------------------------------------------
# 5. Broker disconnect
# --------------------------------------------------------------------------


def test_broker_disconnect_blocks_startup_without_attempting_further_steps(
    tmp_path: Path, clock: _ClockBox
) -> None:
    broker = _StubBroker()
    broker.healthy = False
    broker.health_detail = "connection refused"

    report = _make_sequence(tmp_path, clock, broker).run()

    assert report.system_state is SystemState.RECONCILIATION_REQUIRED
    assert report.permit_strategy_execution is False
    assert report.broker_connected is False
    assert report.broker_health_detail == "connection refused"
    assert broker.get_positions_calls == 0


# --------------------------------------------------------------------------
# 6. Duplicate broker event
# --------------------------------------------------------------------------


def test_duplicate_broker_event_is_deduplicated(tmp_path: Path, clock: _ClockBox) -> None:
    broker = _StubBroker()
    fill = BrokerFill(
        trade_id="T-1",
        order_id="c-1",
        instrument_id="NSE:INFY",
        side="buy",
        quantity=10,
        price=1500.0,
        product="CNC",
        as_of=_T0,
    )
    broker.trades_response = [fill, fill]  # the exact same event, redelivered

    report = _make_sequence(tmp_path, clock, broker).run()

    assert any("duplicate broker fill events detected" in message for message in report.messages)
    assert report.system_state is SystemState.READY  # otherwise-clean run still succeeds


# --------------------------------------------------------------------------
# 7. Missing local record
# --------------------------------------------------------------------------


def test_missing_local_record_for_a_broker_position_requires_reconciliation(
    tmp_path: Path, clock: _ClockBox
) -> None:
    broker = _StubBroker()
    broker.positions_response = [
        BrokerPosition(instrument_id="NSE:TCS", quantity=5, avg_price=3500.0)
    ]
    report = _make_sequence(tmp_path, clock, broker).run()

    assert report.system_state is SystemState.RECONCILIATION_REQUIRED
    assert report.position_reconciliation is not None
    mismatch = report.position_reconciliation.mismatches[0]
    assert mismatch.instrument_id == "NSE:TCS"
    assert "no local record" in mismatch.detail


# --------------------------------------------------------------------------
# 8. Unknown local order
# --------------------------------------------------------------------------


def test_unknown_local_order_is_resolved_during_startup(tmp_path: Path, clock: _ClockBox) -> None:
    broker = _StubBroker()
    order_manager = OrderManager(clock=clock)
    created = order_manager.create(
        "NSE:INFY", "buy", 10, "limit", 1500.0,
        idempotency_key="req-1", signal_id="sig-1", risk_decision_id="rd-1",
    )
    client_order_id = created.order.client_order_id
    order_manager.transition(client_order_id, OrderState.SUBMITTED)
    order_manager.transition(client_order_id, OrderState.UNKNOWN)

    resolved_order = BrokerOrder(
        client_order_id=client_order_id,
        broker_order_id="B-1",
        instrument_id="NSE:INFY",
        side="buy",
        quantity=10,
        order_type="limit",
        limit_price=1500.0,
        status=OrderState.OPEN.value,
    )
    broker.orders[client_order_id] = resolved_order
    broker.open_orders_response = [resolved_order]

    report = _make_sequence(tmp_path, clock, broker, order_manager=order_manager).run()

    assert order_manager.get(client_order_id).state is OrderState.OPEN
    assert report.system_state is SystemState.READY


# --------------------------------------------------------------------------
# The remaining 13 steps not covered by a named scenario above
# --------------------------------------------------------------------------


def test_config_load_failure_raises_startup_error(tmp_path: Path, clock: _ClockBox) -> None:
    def failing_loader() -> object:
        raise ConfigError("settings.yaml is missing")

    broker = _StubBroker()
    with pytest.raises(StartupError, match="configuration failed to load"):
        _make_sequence(tmp_path, clock, broker, settings_loader=failing_loader).run()


def test_persisted_schema_version_mismatch_raises_startup_error(
    tmp_path: Path, clock: _ClockBox
) -> None:
    state_path = tmp_path / "system_state.json"
    from execution.system_state import PersistedState

    incompatible = PersistedState(
        schema_version=STATE_SCHEMA_VERSION + 1,
        app_version=APP_VERSION,
        system_state=SystemState.READY,
        model_version=None,
        strategy_version="strategy-v1",
        portfolio_snapshot=None,
        last_market_data_timestamp=None,
        last_broker_event_id=None,
        last_broker_event_timestamp=None,
        updated_at=_T0,
    )
    SystemStateStore(state_path).save(incompatible)

    broker = _StubBroker()
    with pytest.raises(StartupError, match="schema_version"):
        _make_sequence(tmp_path, clock, broker, state_path=state_path).run()


def test_application_version_change_is_reported_but_does_not_block(
    tmp_path: Path, clock: _ClockBox
) -> None:
    state_path = tmp_path / "system_state.json"
    from execution.system_state import PersistedState

    previous = PersistedState(
        schema_version=STATE_SCHEMA_VERSION,
        app_version="0.0.1-old",
        system_state=SystemState.READY,
        model_version=None,
        strategy_version="strategy-v1",
        portfolio_snapshot=None,
        last_market_data_timestamp=None,
        last_broker_event_id=None,
        last_broker_event_timestamp=None,
        updated_at=_T0,
    )
    SystemStateStore(state_path).save(previous)

    broker = _StubBroker()
    report = _make_sequence(tmp_path, clock, broker, state_path=state_path).run()

    assert report.system_state is SystemState.READY
    assert report.previous_app_version == "0.0.1-old"
    assert any("application version changed" in message for message in report.messages)


def test_no_approved_model_blocks_strategy_execution(tmp_path: Path, clock: _ClockBox) -> None:
    broker = _StubBroker()
    registry = ModelRegistry(artifact_root=tmp_path / "models")  # nothing approved

    report = _make_sequence(tmp_path, clock, broker, model_registry=registry).run()

    assert report.system_state is SystemState.RECONCILIATION_REQUIRED
    assert report.permit_strategy_execution is False
    assert report.risk_state_verified is False
    assert report.model_version is None


def test_halted_circuit_breaker_blocks_strategy_execution(tmp_path: Path, clock: _ClockBox) -> None:
    cb_path = tmp_path / "cb.json"
    circuit_breaker = CircuitBreaker(risk_config(), cb_path)
    halted_state = PortfolioRiskState(
        as_of=_T0,
        equity=10_000_000.0,
        positions=(),
        daily_pnl_pct=0.20,
        rolling_pnl_pct=0.20,
        peak_to_trough_drawdown_pct=0.0,
        daily_turnover_pct_so_far=0.0,
        max_pairwise_correlation=None,
        correlated_pair=None,
        system_healthy=True,
        system_detail=None,
        broker_connected=True,
        broker_detail=None,
    )
    circuit_breaker.evaluate(halted_state)
    assert circuit_breaker.current_status().state is CircuitState.HALTED

    broker = _StubBroker()
    report = _make_sequence(tmp_path, clock, broker, circuit_breaker=circuit_breaker).run()

    assert report.system_state is SystemState.HALTED
    assert report.permit_strategy_execution is False
    assert report.circuit_breaker_state is CircuitState.HALTED


def test_broker_state_retrieval_failure_requires_reconciliation(
    tmp_path: Path, clock: _ClockBox
) -> None:
    class _ExplodingBroker(_StubBroker):
        def get_positions(self) -> list[BrokerPosition]:
            raise ConnectionError("network reset")

    report = _make_sequence(tmp_path, clock, _ExplodingBroker()).run()
    assert report.system_state is SystemState.RECONCILIATION_REQUIRED
    assert any("failed to retrieve broker state" in message for message in report.messages)


# --------------------------------------------------------------------------
# Manual recovery mechanism
# --------------------------------------------------------------------------


def test_acknowledge_and_recover_requires_operator_and_reason(
    tmp_path: Path, clock: _ClockBox
) -> None:
    broker = _StubBroker()
    sequence = _make_sequence(tmp_path, clock, broker)
    with pytest.raises(StartupError, match="operator and reason"):
        sequence.acknowledge_and_recover("", "")


def test_acknowledge_and_recover_re_runs_and_can_reach_ready_once_fixed(
    tmp_path: Path, clock: _ClockBox
) -> None:
    broker = _StubBroker()
    broker.positions_response = [
        BrokerPosition(instrument_id="NSE:TCS", quantity=5, avg_price=3500.0)
    ]
    position_tracker = PositionTracker()
    sequence = _make_sequence(tmp_path, clock, broker, position_tracker=position_tracker)

    first = sequence.run()
    assert first.system_state is SystemState.RECONCILIATION_REQUIRED

    # An operator investigates and confirms the broker's position is
    # correct -- the resolution itself is a deliberate, external action
    # (here: bringing local state in line with the broker's), not
    # anything this module does on its own.
    position_tracker.apply_fill("NSE:TCS", 5, 3500.0, TradeSide.BUY, _T0)

    second = sequence.acknowledge_and_recover(
        operator="ops@example.com", reason="confirmed broker position is correct"
    )
    assert second.system_state is SystemState.READY


def test_acknowledge_and_recover_stays_in_reconciliation_required_if_still_broken(
    tmp_path: Path, clock: _ClockBox
) -> None:
    broker = _StubBroker()
    broker.positions_response = [
        BrokerPosition(instrument_id="NSE:TCS", quantity=5, avg_price=3500.0)
    ]
    sequence = _make_sequence(tmp_path, clock, broker)
    sequence.run()
    second = sequence.acknowledge_and_recover(operator="ops@example.com", reason="checking")
    assert second.system_state is SystemState.RECONCILIATION_REQUIRED


# --------------------------------------------------------------------------
# checkpoint_market_data
# --------------------------------------------------------------------------


def test_checkpoint_market_data_updates_only_that_field(tmp_path: Path, clock: _ClockBox) -> None:
    broker = _StubBroker()
    sequence = _make_sequence(tmp_path, clock, broker)
    sequence.run()

    checkpoint_time = _T0 + dt.timedelta(minutes=5)
    sequence.checkpoint_market_data(checkpoint_time)

    persisted = sequence.state_store.load()
    assert persisted is not None
    assert persisted.last_market_data_timestamp == checkpoint_time
    assert persisted.system_state is SystemState.READY  # unchanged


def test_checkpoint_market_data_before_any_run_raises(tmp_path: Path, clock: _ClockBox) -> None:
    broker = _StubBroker()
    sequence = _make_sequence(tmp_path, clock, broker)
    with pytest.raises(StartupError, match="before the startup sequence"):
        sequence.checkpoint_market_data(_T0)


# --------------------------------------------------------------------------
# Real-PaperBroker fixtures for the CRITICAL crash-during-submission test
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


class _LostResponseBroker(Broker):
    """Wraps a real ``Broker``. Drops the response to exactly one
    ``place_order`` call for a chosen ``client_order_id`` -- the order is
    actually processed underneath, the caller just never learns the
    outcome from that call. Identical to Phase 17's own test double.
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
            self._inner.place_order(order)
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
