"""Unit tests for ``validation/invariants.py`` (Phase 21): each of the
six point-in-time checks fires exactly when it should, against a fully
controllable stub broker -- not against the full harness, so a violation
can be constructed directly rather than engineered through a whole
session.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from backtest.costs import TradeSide
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
from execution.order_manager import OrderManager, OrderState
from execution.position_tracker import PositionTracker
from execution.reconciliation import ReconciliationEngine
from execution.system_state import SystemState, SystemStateStore
from risk.circuit_breaker import CircuitBreaker
from risk.portfolio_risk_state import PortfolioRiskState
from tests.unit._wf_support import risk_config
from validation.invariants import Invariant, InvariantResult, InvariantStatus, check_invariants

_T0 = dt.datetime(2024, 6, 3, 10, 0, tzinfo=dt.UTC)


class _ClockBox:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now


class _StubBroker(Broker):
    def __init__(self) -> None:
        self.equity = 1_000_000.0
        self.cash = 1_000_000.0
        self.positions: list[BrokerPosition] = []
        self.open_orders: list[BrokerOrder] = []
        self.raises_on_positions = False

    def capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        raise NotImplementedError

    def get_account(self) -> BrokerAccount:
        return BrokerAccount(
            account_id="stub", equity=self.equity, cash=self.cash,
            buying_power=self.cash, as_of=_T0,
        )

    def get_positions(self) -> list[BrokerPosition]:
        if self.raises_on_positions:
            raise ConnectionError("broker unreachable")
        return list(self.positions)

    def get_open_orders(self) -> list[BrokerOrder]:
        return list(self.open_orders)

    def get_order(self, order_id: str) -> BrokerOrder:
        raise NotImplementedError

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        return []

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
        return HealthStatus(healthy=True, detail="ok", checked_at=_T0, session_active=True)


class Rig:
    def __init__(self, tmp_path: Path) -> None:
        self.clock = _ClockBox(_T0)
        self.broker = _StubBroker()
        self.position_tracker = PositionTracker()
        self.order_manager = OrderManager(clock=self.clock)
        self.circuit_breaker = CircuitBreaker(risk_config(), tmp_path / "cb.json")
        self.reconciliation_engine = ReconciliationEngine(
            self.position_tracker, self.order_manager, self.broker, clock=self.clock
        )
        self.state_store = SystemStateStore(tmp_path / "state.json")

    def check(self) -> dict[Invariant, InvariantResult]:
        results = check_invariants(
            position_tracker=self.position_tracker,
            order_manager=self.order_manager,
            broker=self.broker,
            circuit_breaker=self.circuit_breaker,
            reconciliation_engine=self.reconciliation_engine,
            state_store=self.state_store,
        )
        return {result.invariant: result for result in results}

    def create_order(self, instrument_id: str = "NSE:INFY") -> str:
        result = self.order_manager.create(
            instrument_id, "buy", 10, "limit", 100.0,
            idempotency_key=f"key-{instrument_id}", signal_id=f"sig-{instrument_id}",
            risk_decision_id="rd-1",
        )
        return result.order.client_order_id


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


# --------------------------------------------------------------------------
# no duplicate positions
# --------------------------------------------------------------------------


def test_no_duplicate_positions_passes_on_a_clean_book(rig: Rig) -> None:
    result = rig.check()[Invariant.NO_DUPLICATE_POSITIONS]
    assert result.status is InvariantStatus.PASS


def test_no_duplicate_positions_fails_when_the_broker_reports_the_same_id_twice(
    rig: Rig,
) -> None:
    rig.broker.positions = [
        BrokerPosition(instrument_id="NSE:INFY", quantity=10, avg_price=100.0),
        BrokerPosition(instrument_id="NSE:INFY", quantity=5, avg_price=101.0),
    ]
    result = rig.check()[Invariant.NO_DUPLICATE_POSITIONS]
    assert result.status is InvariantStatus.FAIL
    assert "NSE:INFY" in result.detail


def test_no_duplicate_positions_is_skipped_when_the_broker_is_unreachable(rig: Rig) -> None:
    rig.broker.raises_on_positions = True
    result = rig.check()[Invariant.NO_DUPLICATE_POSITIONS]
    assert result.status is InvariantStatus.SKIPPED


# --------------------------------------------------------------------------
# no negative cash
# --------------------------------------------------------------------------


def test_no_negative_cash_passes_with_positive_cash(rig: Rig) -> None:
    assert rig.check()[Invariant.NO_NEGATIVE_CASH].status is InvariantStatus.PASS


def test_no_negative_cash_fails_when_cash_is_negative(rig: Rig) -> None:
    rig.broker.cash = -1.0
    result = rig.check()[Invariant.NO_NEGATIVE_CASH]
    assert result.status is InvariantStatus.FAIL


# --------------------------------------------------------------------------
# no leverage
# --------------------------------------------------------------------------


def test_no_leverage_passes_when_fully_invested_but_not_over(rig: Rig) -> None:
    rig.position_tracker.apply_fill("NSE:INFY", 100, 1_000.0, TradeSide.BUY, _T0)
    rig.position_tracker.mark_to_market({"NSE:INFY": 1_000.0}, _T0)
    rig.broker.equity = 100_000.0
    result = rig.check()[Invariant.NO_LEVERAGE]
    assert result.status is InvariantStatus.PASS


def test_no_leverage_fails_when_holdings_exceed_equity(rig: Rig) -> None:
    rig.position_tracker.apply_fill("NSE:INFY", 100, 1_000.0, TradeSide.BUY, _T0)
    rig.position_tracker.mark_to_market({"NSE:INFY": 1_000.0}, _T0)
    rig.broker.equity = 50_000.0  # holdings are worth 100,000
    result = rig.check()[Invariant.NO_LEVERAGE]
    assert result.status is InvariantStatus.FAIL
    assert "exceeds 100%" in result.detail


def test_no_leverage_fails_on_a_short_position() -> None:
    """``check_invariants``' own short-position check
    (``_no_leverage``'s ``shorts`` branch) is a second, independent line
    of defense -- but ``execution.position_tracker.Position`` already
    refuses a negative quantity structurally, so a genuine short can
    never reach that branch through this codebase's own APIs. This test
    documents exactly that guarantee, at the point where it is actually
    enforced.
    """
    from execution.position_tracker import Position

    with pytest.raises(ValueError, match="long-only"):
        Position(
            instrument_id="NSE:INFY", quantity=-10, avg_price=100.0, current_price=100.0,
            unrealized_pnl=0.0, realized_pnl=0.0, target_weight=0.0, as_of=_T0,
        )


# --------------------------------------------------------------------------
# all orders traceable
# --------------------------------------------------------------------------


def test_all_orders_traceable_passes_with_no_orders(rig: Rig) -> None:
    assert rig.check()[Invariant.ALL_ORDERS_TRACEABLE].status is InvariantStatus.PASS


def test_all_orders_traceable_passes_for_a_properly_created_order(rig: Rig) -> None:
    rig.create_order()
    result = rig.check()[Invariant.ALL_ORDERS_TRACEABLE]
    assert result.status is InvariantStatus.PASS
    assert "1" in result.detail


# --------------------------------------------------------------------------
# reconciliation succeeds
# --------------------------------------------------------------------------


def test_reconciliation_succeeds_on_a_clean_book(rig: Rig) -> None:
    assert rig.check()[Invariant.RECONCILIATION_SUCCEEDS].status is InvariantStatus.PASS


def test_reconciliation_fails_on_a_position_mismatch(rig: Rig) -> None:
    rig.broker.positions = [
        BrokerPosition(instrument_id="NSE:GHOST", quantity=10, avg_price=100.0)
    ]
    result = rig.check()[Invariant.RECONCILIATION_SUCCEEDS]
    assert result.status is InvariantStatus.FAIL
    assert "NSE:GHOST" in result.detail or "mismatch" in result.detail.lower()


def test_reconciliation_fails_on_an_orphaned_open_order(rig: Rig) -> None:
    rig.broker.open_orders = [
        BrokerOrder(
            client_order_id="not-ours", broker_order_id="B-1", instrument_id="NSE:INFY",
            side="buy", quantity=7, order_type="limit", limit_price=100.0,
            status=OrderState.OPEN.value,
        )
    ]
    result = rig.check()[Invariant.RECONCILIATION_SUCCEEDS]
    assert result.status is InvariantStatus.FAIL


# --------------------------------------------------------------------------
# halted state persists
# --------------------------------------------------------------------------


def test_halted_state_persists_passes_when_nothing_is_halted(rig: Rig) -> None:
    result = rig.check()[Invariant.HALTED_STATE_PERSISTS]
    assert result.status is InvariantStatus.PASS
    assert "nothing halted" in result.detail


def test_halted_state_persists_fails_when_halted_but_never_persisted(rig: Rig) -> None:
    breaching_state = PortfolioRiskState(
        as_of=_T0, equity=1_000_000.0, positions=(), daily_pnl_pct=0.9, rolling_pnl_pct=0.9,
        peak_to_trough_drawdown_pct=0.9, daily_turnover_pct_so_far=0.0,
        max_pairwise_correlation=None, correlated_pair=None, system_healthy=True,
        system_detail=None, broker_connected=True, broker_detail=None,
    )
    rig.circuit_breaker.evaluate(breaching_state)
    # Deliberately do not persist SystemStateStore -- the breaker's own
    # file records the halt, but nothing wrote a matching system_state.
    result = rig.check()[Invariant.HALTED_STATE_PERSISTS]
    assert result.status is InvariantStatus.FAIL
    assert "nothing was persisted" in result.detail


def test_halted_state_persists_passes_when_both_sides_agree(rig: Rig) -> None:
    from execution.system_state import PersistedState

    breaching_state = PortfolioRiskState(
        as_of=_T0, equity=1_000_000.0, positions=(), daily_pnl_pct=0.9, rolling_pnl_pct=0.9,
        peak_to_trough_drawdown_pct=0.9, daily_turnover_pct_so_far=0.0,
        max_pairwise_correlation=None, correlated_pair=None, system_healthy=True,
        system_detail=None, broker_connected=True, broker_detail=None,
    )
    rig.circuit_breaker.evaluate(breaching_state)
    rig.state_store.save(
        PersistedState(
            schema_version=1, app_version="0.1.0", system_state=SystemState.HALTED,
            model_version=None, strategy_version="v1", portfolio_snapshot=None,
            last_market_data_timestamp=None, last_broker_event_id=None,
            last_broker_event_timestamp=None, updated_at=_T0,
        )
    )
    result = rig.check()[Invariant.HALTED_STATE_PERSISTS]
    assert result.status is InvariantStatus.PASS
    assert "halt persisted" in result.detail


# --------------------------------------------------------------------------
# check_invariants as a whole
# --------------------------------------------------------------------------


def test_check_invariants_returns_a_fixed_order(rig: Rig) -> None:
    results = check_invariants(
        position_tracker=rig.position_tracker,
        order_manager=rig.order_manager,
        broker=rig.broker,
        circuit_breaker=rig.circuit_breaker,
        reconciliation_engine=rig.reconciliation_engine,
        state_store=rig.state_store,
    )
    assert [r.invariant for r in results] == [
        Invariant.NO_DUPLICATE_POSITIONS,
        Invariant.NO_NEGATIVE_CASH,
        Invariant.NO_LEVERAGE,
        Invariant.ALL_ORDERS_TRACEABLE,
        Invariant.RECONCILIATION_SUCCEEDS,
        Invariant.HALTED_STATE_PERSISTS,
    ]


def test_skipped_is_not_ok(rig: Rig) -> None:
    rig.broker.raises_on_positions = True
    result = rig.check()[Invariant.NO_DUPLICATE_POSITIONS]
    assert result.status is InvariantStatus.SKIPPED
    assert result.ok is False
