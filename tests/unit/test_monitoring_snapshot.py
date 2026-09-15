"""Unit tests for ``monitoring/snapshot.py`` (Phase 20): every field on
the dashboard comes from the module that actually owns it, and the
collector degrades gracefully when a source has nothing to say.
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
from core.regime.allocation import AllocationRegime, AllocationTarget
from core.regime.hmm_engine import RegimeLabel, RegimeState
from core.regime.model_registry import ModelRegistry
from data.models import SessionType
from execution.order_manager import OrderManager, OrderState
from execution.position_tracker import PositionTracker
from execution.reconciliation import ReconciliationEngine
from monitoring.health import ComponentHealth, HealthChecker
from monitoring.snapshot import SnapshotCollector
from risk.circuit_breaker import CircuitBreaker, CircuitState
from risk.portfolio_risk_state import PortfolioRiskState
from risk.risk_state_builder import EquityHistory
from tests.unit._wf_support import INDEX_SYMBOL, VIX_SYMBOL, Environment

_EQUITY = 10_000_000.0


class _ClockBox:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + dt.timedelta(**kwargs)


class _StubBroker(Broker):
    def __init__(self) -> None:
        self.healthy = True
        self.detail = "ok"
        self.equity = _EQUITY
        self.cash = 4_000_000.0
        self.positions: list[BrokerPosition] = []

    def capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        raise NotImplementedError

    def get_account(self) -> BrokerAccount:
        return BrokerAccount(
            account_id="stub",
            equity=self.equity,
            cash=self.cash,
            buying_power=self.cash,
            as_of=dt.datetime(2015, 1, 1, tzinfo=dt.UTC),
        )

    def get_positions(self) -> list[BrokerPosition]:
        return list(self.positions)

    def get_open_orders(self) -> list[BrokerOrder]:
        return []

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
        return HealthStatus(
            healthy=self.healthy,
            detail=self.detail,
            checked_at=dt.datetime(2015, 1, 1, tzinfo=dt.UTC),
            session_active=self.healthy,
        )


@pytest.fixture(scope="module")
def env() -> Environment:
    return Environment(n_days=120)


@pytest.fixture
def clock(env: Environment) -> _ClockBox:
    return _ClockBox(dt.datetime.combine(env.dates[-1], dt.time(11, 0), tzinfo=dt.UTC))


@pytest.fixture
def broker() -> _StubBroker:
    return _StubBroker()


class Wiring:
    """The collector plus the live components a test needs to mutate."""

    def __init__(
        self,
        env: Environment,
        broker: _StubBroker,
        clock: _ClockBox,
        tmp_path: Path,
        **overrides: object,
    ) -> None:
        self.env = env
        self.broker = broker
        self.clock = clock
        self.position_tracker = PositionTracker()
        self.order_manager = OrderManager(clock=clock)
        self.circuit_breaker = CircuitBreaker(env.risk_cfg, tmp_path / "cb.json")
        self.reconciliation_engine = ReconciliationEngine(
            self.position_tracker, self.order_manager, broker, clock=clock
        )
        self.registry = ModelRegistry(tmp_path / "models")
        self.equity_history = EquityHistory()
        self.regime_state: RegimeState | None = None
        self.allocation_target: AllocationTarget | None = None
        self.cash_flow = 0.0
        self.state = "running"

        health_checker = HealthChecker(
            broker,
            env.market_data,
            env.universe_provider,
            self.registry,
            env.calendar,
            env.instrument_ids,
            clock=clock,
        )
        kwargs: dict[str, object] = dict(
            broker=broker,
            position_tracker=self.position_tracker,
            order_manager=self.order_manager,
            circuit_breaker=self.circuit_breaker,
            reconciliation_engine=self.reconciliation_engine,
            health_checker=health_checker,
            calendar=env.calendar,
            market_data=env.market_data,
            model_registry=self.registry,
            equity_history=self.equity_history,
            index_symbol=INDEX_SYMBOL,
            vix_symbol=VIX_SYMBOL,
            state_supplier=lambda: self.state,
            regime_supplier=lambda: self.regime_state,
            allocation_supplier=lambda: self.allocation_target,
            cash_flow_supplier=lambda: self.cash_flow,
            started_at=clock.now,
            clock=clock,
        )
        kwargs.update(overrides)
        self.collector = SnapshotCollector(**kwargs)  # type: ignore[arg-type]

    def create_order(
        self, instrument_id: str, quantity: int = 10, price: float = 100.0
    ) -> str:
        result = self.order_manager.create(
            instrument_id,
            "buy",
            quantity,
            "limit",
            price,
            idempotency_key=f"key-{instrument_id}-{len(self.order_manager.all_orders())}",
            signal_id=f"sig-{instrument_id}-{len(self.order_manager.all_orders())}",
            risk_decision_id="rd-1",
        )
        return result.order.client_order_id


@pytest.fixture
def wiring(env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path) -> Wiring:
    return Wiring(env, broker, clock, tmp_path)


# --------------------------------------------------------------------------
# SYSTEM
# --------------------------------------------------------------------------


def test_system_reports_the_supplied_lifecycle_state(wiring: Wiring) -> None:
    wiring.state = "degraded"
    assert wiring.collector.collect().system.status == "degraded"


def test_uptime_grows_with_the_clock(wiring: Wiring) -> None:
    assert wiring.collector.collect().system.uptime == dt.timedelta(0)
    wiring.clock.advance(hours=2, minutes=30)
    assert wiring.collector.collect().system.uptime == dt.timedelta(hours=2, minutes=30)


def test_system_reports_the_trading_session(wiring: Wiring) -> None:
    system = wiring.collector.collect().system
    assert system.session_date == wiring.clock.now.date()
    assert system.session_type is SessionType.REGULAR
    assert "-" in system.session_detail


def test_a_date_outside_calendar_coverage_is_reported_not_raised(
    env: Environment, broker: _StubBroker, tmp_path: Path
) -> None:
    far_future = _ClockBox(dt.datetime(2099, 1, 5, 11, 0, tzinfo=dt.UTC))
    wiring = Wiring(env, broker, far_future, tmp_path)
    system = wiring.collector.collect().system
    assert system.session_type is SessionType.CLOSED
    assert "does not cover" in system.session_detail


def test_system_reports_feed_and_broker_health(wiring: Wiring) -> None:
    system = wiring.collector.collect().system
    assert system.data_feed is ComponentHealth.HEALTHY
    assert system.broker_connected is True

    wiring.broker.healthy = False
    wiring.broker.detail = "connection refused"
    system = wiring.collector.collect().system
    assert system.broker_connected is False
    assert system.broker_detail == "connection refused"


def test_model_version_is_none_until_one_is_approved(wiring: Wiring) -> None:
    assert wiring.collector.collect().system.model_version is None


# --------------------------------------------------------------------------
# PORTFOLIO
# --------------------------------------------------------------------------


def test_portfolio_reports_equity_and_cash_from_the_broker(wiring: Wiring) -> None:
    portfolio = wiring.collector.collect().portfolio
    assert portfolio.equity == _EQUITY
    assert portfolio.cash == 4_000_000.0


def test_exposure_and_position_count_come_from_the_position_tracker(wiring: Wiring) -> None:
    now = wiring.clock.now
    wiring.position_tracker.apply_fill("NSE:S00", 1_000, 2_000.0, TradeSide.BUY, now)
    wiring.position_tracker.apply_fill("NSE:S01", 500, 1_000.0, TradeSide.BUY, now)

    portfolio = wiring.collector.collect().portfolio
    assert portfolio.position_count == 2
    # (1000 * 2000 + 500 * 1000) / 10,000,000
    assert portfolio.gross_exposure_pct == pytest.approx(0.25)


def test_daily_pnl_is_signed_unlike_the_risk_engine_s_loss_only_convention(
    wiring: Wiring,
) -> None:
    wiring.equity_history.append(9_000_000.0)
    wiring.equity_history.append(_EQUITY)
    portfolio = wiring.collector.collect().portfolio
    assert portfolio.daily_pnl == pytest.approx(1_000_000.0)
    assert portfolio.daily_pnl_pct == pytest.approx(1_000_000.0 / 9_000_000.0)


def test_daily_pnl_is_zero_without_a_prior_equity_point(wiring: Wiring) -> None:
    portfolio = wiring.collector.collect().portfolio
    assert portfolio.daily_pnl == 0.0
    assert portfolio.daily_pnl_pct == 0.0


def test_drawdown_comes_from_the_equity_history(wiring: Wiring) -> None:
    wiring.equity_history.append(12_500_000.0)
    wiring.equity_history.append(_EQUITY)
    assert wiring.collector.collect().portfolio.drawdown_pct == pytest.approx(0.20)


def test_cumulative_fill_cash_flow_is_taken_from_its_supplier(wiring: Wiring) -> None:
    wiring.cash_flow = -1_234.5
    assert wiring.collector.collect().portfolio.cumulative_fill_cash_flow == -1_234.5


# --------------------------------------------------------------------------
# REGIME
# --------------------------------------------------------------------------


def _regime_state(as_of: dt.date) -> RegimeState:
    return RegimeState(
        as_of=as_of,
        state_id=1,
        label=RegimeLabel.CALM,
        probabilities=(0.12, 0.88),
        confidence=0.88,
        expected_volatility=0.14,
        expected_return=0.08,
        persistence=0.94,
    )


def test_regime_panel_is_empty_before_the_first_regime_call(wiring: Wiring) -> None:
    regime = wiring.collector.collect().regime
    assert regime.regime is None
    assert regime.label is None
    assert regime.confidence is None


def test_regime_panel_reports_the_tier_and_the_label_separately(wiring: Wiring) -> None:
    as_of = wiring.clock.now.date()
    wiring.regime_state = _regime_state(as_of)
    wiring.allocation_target = AllocationTarget(
        as_of=as_of,
        regime=AllocationRegime.NORMAL_RISK,
        target_gross_exposure=0.45,
        min_gross_exposure=0.30,
        max_gross_exposure=0.60,
        allow_new_positions=True,
        confidence=0.88,
        expected_volatility=0.14,
        reason="confirmed",
    )
    regime = wiring.collector.collect().regime
    assert regime.regime == AllocationRegime.NORMAL_RISK.value
    assert regime.label == RegimeLabel.CALM.value
    assert regime.probability == pytest.approx(0.88)
    assert regime.confidence == pytest.approx(0.88)
    assert regime.persistence == pytest.approx(0.94)
    assert regime.as_of == as_of


def test_index_levels_and_changes_come_from_market_data(wiring: Wiring) -> None:
    regime = wiring.collector.collect().regime
    assert regime.india_vix is not None
    assert regime.india_vix_change is not None
    assert regime.nifty_close is not None
    assert regime.nifty_change_pct is not None


def test_a_missing_index_series_reports_none_rather_than_raising(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    wiring = Wiring(env, broker, clock, tmp_path, vix_symbol="NOT-INGESTED")
    regime = wiring.collector.collect().regime
    assert regime.india_vix is None
    assert regime.india_vix_change is None
    assert regime.nifty_close is not None


# --------------------------------------------------------------------------
# EXECUTION
# --------------------------------------------------------------------------


def test_execution_counts_orders_by_state(wiring: Wiring) -> None:
    created_only = wiring.create_order("NSE:S00")
    submitted = wiring.create_order("NSE:S01")
    wiring.order_manager.transition(submitted, OrderState.SUBMITTED)
    rejected = wiring.create_order("NSE:S02")
    wiring.order_manager.transition(rejected, OrderState.SUBMITTED)
    wiring.order_manager.transition(rejected, OrderState.REJECTED, reject_reason="margin")
    unknown = wiring.create_order("NSE:S03")
    wiring.order_manager.transition(unknown, OrderState.SUBMITTED)
    wiring.order_manager.transition(unknown, OrderState.UNKNOWN)

    execution = wiring.collector.collect().execution
    assert execution.orders_submitted == 3, "the CREATED order has not been submitted"
    assert execution.rejected_orders == 1
    assert execution.orders_in_unknown_state == 1
    assert execution.open_orders == 2, "submitted and unknown are both still live"
    assert created_only  # referenced for clarity


def test_execution_counts_fills(wiring: Wiring) -> None:
    filled = wiring.create_order("NSE:S00", quantity=10)
    wiring.order_manager.transition(filled, OrderState.SUBMITTED)
    wiring.order_manager.transition(
        filled, OrderState.FILLED, filled_quantity=10, avg_fill_price=100.0
    )
    assert wiring.collector.collect().execution.fills == 1


def test_pending_reconciliation_sums_unknown_orders_and_position_mismatches(
    wiring: Wiring,
) -> None:
    unknown = wiring.create_order("NSE:S00")
    wiring.order_manager.transition(unknown, OrderState.SUBMITTED)
    wiring.order_manager.transition(unknown, OrderState.UNKNOWN)
    wiring.broker.positions = [
        BrokerPosition(instrument_id="NSE:GHOST", quantity=10, avg_price=100.0)
    ]

    execution = wiring.collector.collect().execution
    assert len(execution.reconciliation_mismatches) == 1
    assert execution.pending_reconciliation == 2


def test_a_clean_system_has_nothing_pending(wiring: Wiring) -> None:
    assert wiring.collector.collect().execution.pending_reconciliation == 0


# --------------------------------------------------------------------------
# RISK
# --------------------------------------------------------------------------


def test_risk_mode_comes_from_the_circuit_breaker(wiring: Wiring) -> None:
    assert wiring.collector.collect().risk.risk_mode is CircuitState.NORMAL

    wiring.circuit_breaker.evaluate(
        PortfolioRiskState(
            as_of=wiring.clock.now,
            equity=_EQUITY,
            positions=(),
            daily_pnl_pct=0.90,
            rolling_pnl_pct=0.90,
            peak_to_trough_drawdown_pct=0.90,
            daily_turnover_pct_so_far=0.0,
            max_pairwise_correlation=None,
            correlated_pair=None,
            system_healthy=True,
            system_detail=None,
            broker_connected=True,
            broker_detail=None,
        )
    )
    risk = wiring.collector.collect().risk
    assert risk.risk_mode is CircuitState.HALTED
    assert risk.requires_manual_recovery is True
    assert risk.circuit_reason


def test_concentration_is_the_largest_single_holding(wiring: Wiring) -> None:
    now = wiring.clock.now
    wiring.position_tracker.apply_fill("NSE:S00", 100, 1_000.0, TradeSide.BUY, now)
    wiring.position_tracker.apply_fill("NSE:S01", 1_000, 2_000.0, TradeSide.BUY, now)

    risk = wiring.collector.collect().risk
    assert risk.concentration_instrument == "NSE:S01"
    assert risk.max_concentration_pct == pytest.approx(0.20)


def test_concentration_is_empty_with_no_positions(wiring: Wiring) -> None:
    risk = wiring.collector.collect().risk
    assert risk.max_concentration_pct == 0.0
    assert risk.concentration_instrument is None


def test_turnover_counts_todays_orders_against_equity(wiring: Wiring) -> None:
    wiring.create_order("NSE:S00", quantity=1_000, price=1_000.0)
    assert wiring.collector.collect().risk.daily_turnover_pct == pytest.approx(0.10)


def test_turnover_ignores_orders_from_a_previous_day(wiring: Wiring) -> None:
    wiring.create_order("NSE:S00", quantity=1_000, price=1_000.0)
    wiring.clock.advance(days=1)
    assert wiring.collector.collect().risk.daily_turnover_pct == 0.0


# --------------------------------------------------------------------------
# The snapshot as a whole
# --------------------------------------------------------------------------


def test_collect_stamps_the_snapshot_with_the_current_time(wiring: Wiring) -> None:
    assert wiring.collector.collect().as_of == wiring.clock.now


def test_collect_returns_a_complete_snapshot(wiring: Wiring) -> None:
    snapshot = wiring.collector.collect()
    assert snapshot.system is not None
    assert snapshot.portfolio is not None
    assert snapshot.regime is not None
    assert snapshot.execution is not None
    assert snapshot.risk is not None
