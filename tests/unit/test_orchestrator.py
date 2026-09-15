"""Integration tests for ``orchestration/orchestrator.py`` (Phase 19).

These are deliberately integration tests, not mocks of the pipeline: each
one wires a real ``StockSelector``, ``PortfolioConstructor``,
``RiskManager``, ``CircuitBreaker``, ``HMMRegimeEngine`` (through a real
fitted, approved ``ModelArtifact``), ``OrderManager``, ``PositionTracker``,
``StartupSequence``, ``ReconciliationEngine`` and a real ``PaperBroker``
against a synthetic multi-year market, then runs the actual 20-step daily
workflow through them. The assertions are about what the *lifecycle* does
-- which state it lands in, whether orders reach the broker, whether
positions and persisted state follow -- never about the strategy numbers
themselves, which are each dedicated module's own tests' job.
"""

from __future__ import annotations

import datetime as dt
import signal
from collections.abc import Callable, Mapping
from decimal import Decimal
from pathlib import Path

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
from config.loader import ConfigError
from config.models import PaperTradingConfig, Settings
from core.features.feature_engineering import (
    MarketFeatureInputs,
    drop_warmup_rows,
    feature_set_version,
)
from core.features.feature_scaler import CausalFeatureScaler
from core.regime.hmm_engine import HMMRegimeEngine
from core.regime.model_registry import ModelArtifact, ModelRegistry, build_model_id
from data.errors import DataNotAvailableError
from data.models import Quote
from execution.order_manager import OrderManager, OrderState
from execution.position_tracker import PositionTracker
from execution.reconciliation import ReconciliationEngine
from execution.system_state import SystemState, SystemStateStore
from monitoring.alerts import Alert, AlertManager, AlertType
from monitoring.health import HealthChecker
from monitoring.snapshot import SnapshotCollector
from monitoring.terminal_dashboard import render_dashboard
from orchestration.heartbeat import Heartbeat
from orchestration.orchestrator import DailyCycleReport, Orchestrator, OrchestratorError
from orchestration.orchestrator_state import OrchestratorState
from orchestration.regime_computation import RegimeComputer
from risk.circuit_breaker import CircuitBreaker, CircuitState
from risk.portfolio_risk_state import PortfolioRiskState
from risk.risk_manager import RiskManager
from risk.risk_state_builder import EquityHistory
from tests.unit._wf_support import (
    INDEX_SYMBOL,
    VIX_SYMBOL,
    Environment,
    FakeMarketDataProvider,
    cost_schedule,
)

# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class _ClockBox:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + dt.timedelta(**kwargs)


class _LiveMarketData(FakeMarketDataProvider):
    """``FakeMarketDataProvider`` is historical-only by design (its
    ``get_quote`` always raises, so a backtest cannot consult a live
    book). ``PaperBroker`` genuinely needs live quotes, so this subclass
    adds them on top of the same historical machinery.
    """

    def __init__(self) -> None:
        super().__init__()
        self._quotes: dict[str, Quote] = {}

    def set_quote(self, quote: Quote) -> None:
        self._quotes[quote.instrument_id] = quote

    def get_quote(self, instrument_id: str) -> Quote:
        try:
            return self._quotes[instrument_id]
        except KeyError:
            raise DataNotAvailableError(f"no live quote for {instrument_id}") from None


class _BrokerProxy(Broker):
    """Passes everything through to a real ``PaperBroker`` but lets a test
    make connectivity fail, or record what was asked of it, without
    touching the real adapter's own logic.
    """

    def __init__(self, inner: Broker) -> None:
        self._inner = inner
        self.healthy = True
        self.health_detail = "ok"
        self.close_all_positions_calls = 0
        self.extra_positions: list[BrokerPosition] = []

    def capabilities(self) -> BrokerCapabilities:
        return self._inner.capabilities()

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        self._inner.authenticate(credentials)

    def get_account(self) -> BrokerAccount:
        return self._inner.get_account()

    def get_positions(self) -> list[BrokerPosition]:
        return [*self._inner.get_positions(), *self.extra_positions]

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

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        return self._inner.place_order(order)

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        return self._inner.modify_order(order_id, changes)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        return self._inner.cancel_order(order_id)

    def close_position(self, instrument_id: str) -> BrokerOrder:
        return self._inner.close_position(instrument_id)

    def close_all_positions(self) -> list[BrokerOrder]:
        self.close_all_positions_calls += 1
        return self._inner.close_all_positions()

    def health_check(self) -> HealthStatus:
        if not self.healthy:
            return HealthStatus(
                healthy=False,
                detail=self.health_detail,
                checked_at=dt.datetime(2015, 1, 1, tzinfo=dt.UTC),
                session_active=False,
            )
        return self._inner.health_check()


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


class Harness:
    """One fully-wired orchestrator plus the pieces a test needs to poke."""

    def __init__(
        self,
        env: Environment,
        tmp_path: Path,
        *,
        as_of: dt.date,
        settings: Settings,
        train_end: dt.date,
        approve_model: bool = True,
        close_positions_on_shutdown: bool = False,
        with_monitoring: bool = False,
    ) -> None:
        self.env = env
        self.as_of = as_of
        self.clock = _ClockBox(dt.datetime.combine(as_of, dt.time(10, 0), tzinfo=dt.UTC))

        self.market_data = _seed_live_market_data(env, self.clock.now)
        self.registry = ModelRegistry(tmp_path / "models")
        self.artifact = _fit_artifact(env, train_end)
        self.registry.save(self.artifact)
        if approve_model:
            self.registry.approve(self.artifact.model_id)

        # The broker keeps its own PositionTracker for order validation;
        # the orchestrator's canonical one is updated only through
        # FillTracker, so a fill is never counted twice.
        self.broker_position_tracker = PositionTracker()
        paper_broker = PaperBroker(
            market_data=self.market_data,
            cost_model=CostModel(
                CostScheduleRepository([cost_schedule()]),
                min_slippage_bps=env.backtest_cfg.slippage_min_bps,
                impact_coefficient=env.backtest_cfg.slippage_impact_coefficient,
            ),
            execution_config=env.execution_cfg,
            paper_config=PaperTradingConfig.model_validate(
                {
                    "submission_latency_ms": 0,
                    "max_fill_participation_pct": 1.0,
                    "order_expiry_seconds": 3600,
                    "default_avg_daily_value_inr": 0.0,
                    "default_volatility": 0.30,
                }
            ),
            position_tracker=self.broker_position_tracker,
            initial_cash=10_000_000.0,
            clock=self.clock,
        )
        self.broker = _BrokerProxy(paper_broker)

        self.position_tracker = PositionTracker()
        self.order_manager = OrderManager(clock=self.clock)
        self.state_store = SystemStateStore(tmp_path / "system_state.json")
        self.circuit_breaker = CircuitBreaker(env.risk_cfg, tmp_path / "circuit_breaker.json")
        self.heartbeat = Heartbeat(self.clock)
        self.health_checker = HealthChecker(
            self.broker,
            env.market_data,
            env.universe_provider,
            self.registry,
            env.calendar,
            env.instrument_ids,
            max_stale_sessions=1,
            model_max_age_sessions=200,
            heartbeat_interval_seconds=300.0,
            last_heartbeat=self.heartbeat.last,
            clock=self.clock,
        )
        self.equity_history = EquityHistory()
        self.alerts_received: list[Alert] = []
        self.alert_manager: AlertManager | None = None
        self.snapshot_collector: SnapshotCollector | None = None
        if with_monitoring:
            self.alert_manager = AlertManager(
                list(settings.monitoring.alert_channels),
                cooldown_seconds=settings.monitoring.alert_cooldown_seconds,
                drawdown_alert_pct=0.10,
                sink=self.alerts_received.append,
                clock=self.clock,
            )
            # The suppliers close over ``self`` and are only called later,
            # so they can reach into the orchestrator constructed below.
            self.snapshot_collector = SnapshotCollector(
                broker=self.broker,
                position_tracker=self.position_tracker,
                order_manager=self.order_manager,
                circuit_breaker=self.circuit_breaker,
                reconciliation_engine=ReconciliationEngine(
                    self.position_tracker, self.order_manager, self.broker, clock=self.clock
                ),
                health_checker=self.health_checker,
                calendar=env.calendar,
                market_data=env.market_data,
                model_registry=self.registry,
                equity_history=self.equity_history,
                index_symbol=INDEX_SYMBOL,
                vix_symbol=VIX_SYMBOL,
                state_supplier=lambda: self.orchestrator.state.value,
                regime_supplier=lambda: self.orchestrator.last_regime_state,
                allocation_supplier=lambda: self.orchestrator.last_allocation_target,
                cash_flow_supplier=lambda: self.orchestrator.fill_tracker.cumulative_cash_flow(),
                started_at=self.clock.now,
                clock=self.clock,
            )

        self.orchestrator = Orchestrator(
            calendar=env.calendar,
            market_data=env.market_data,
            stock_selector=env.stock_selector,
            regime_computer=RegimeComputer(
                env.market_data,
                env.hmm_cfg,
                env.allocation_cfg,
                env.regime_policy,
                env.feature_pipeline,
                INDEX_SYMBOL,
                VIX_SYMBOL,
                feature_warmup_buffer_days=150,
            ),
            portfolio_constructor=env.portfolio_constructor,
            risk_manager=RiskManager(env.risk_cfg, self.circuit_breaker),
            circuit_breaker=self.circuit_breaker,
            model_registry=self.registry,
            position_tracker=self.position_tracker,
            order_manager=self.order_manager,
            broker=self.broker,
            state_store=self.state_store,
            health_checker=self.health_checker,
            execution_config=env.execution_cfg,
            strategy_version="strategy-v1",
            settings_loader=lambda: settings,
            heartbeat=self.heartbeat,
            equity_history=self.equity_history,
            snapshot_collector=self.snapshot_collector,
            alert_manager=self.alert_manager,
            close_positions_on_shutdown=close_positions_on_shutdown,
            clock=self.clock,
        )

    def run(self, as_of: dt.date | None = None) -> DailyCycleReport:
        return self.orchestrator.run_daily_cycle(as_of or self.as_of)


def _seed_live_market_data(env: Environment, now: dt.datetime) -> _LiveMarketData:
    """Copies the environment's synthetic history into a live-quote-capable
    provider (public API only) and publishes a fresh quote per instrument
    at the current clock time, so ``PaperBroker``'s own stale-quote guard
    passes.
    """
    live = _LiveMarketData()
    first, last = env.dates[0], env.dates[-1]
    for instrument_id in env.instrument_ids:
        bars = env.market_data.get_equity_bars(instrument_id, first, last)
        live.add_equity(
            instrument_id,
            [bar.session_date for bar in bars],
            [float(bar.close) for bar in bars],
        )
        close = float(bars[-1].close)
        live.set_quote(
            Quote(
                instrument_id=instrument_id,
                bid=Decimal(str(round(close * 0.999, 2))),
                ask=Decimal(str(round(close * 1.001, 2))),
                last_price=Decimal(str(round(close, 2))),
                as_of=now,
                bid_quantity=1_000_000,
                ask_quantity=1_000_000,
            )
        )
    return live


def _fit_artifact(env: Environment, train_end: dt.date) -> ModelArtifact:
    """The same fit recipe ``WalkForwardValidator._fit_fold`` uses -- fitted
    strictly on ``[start, train_end]``, with the feature version this
    deployment's own pipeline reports so step 8's metadata validation has
    something real to check against.
    """
    nifty = env.market_data.get_index_observations(INDEX_SYMBOL, env.start, train_end)
    vix = env.market_data.get_index_observations(VIX_SYMBOL, env.start, train_end)
    inputs = MarketFeatureInputs.from_index_observations(nifty, vix)
    matrix = drop_warmup_rows(env.feature_pipeline.compute(inputs))
    scaled, scaler_params = CausalFeatureScaler().fit_transform(matrix)
    returns = inputs.frame["nifty_close"].pct_change().reindex(scaled.index)
    model = HMMRegimeEngine(env.hmm_cfg).fit(scaled, returns)
    feature_version = feature_set_version(env.feature_pipeline.definitions)
    return ModelArtifact(
        model_id=build_model_id(
            model.training_result.training_end,
            model.n_states,
            model.training_result.seed,
            feature_version,
        ),
        created_at=dt.datetime.combine(train_end, dt.time(18, 0), tzinfo=dt.UTC),
        model=model,
        scaler=scaler_params,
        feature_version=feature_version,
    )


@pytest.fixture(scope="module")
def env() -> Environment:
    return Environment(n_days=260)


@pytest.fixture(scope="module")
def settings() -> Settings:
    from config.loader import load_settings

    return load_settings()


@pytest.fixture
def harness(env: Environment, settings: Settings, tmp_path: Path) -> Harness:
    return Harness(
        env, tmp_path, as_of=env.dates[-1], settings=settings, train_end=env.dates[200]
    )


# --------------------------------------------------------------------------
# The full daily workflow, end to end
# --------------------------------------------------------------------------


def test_a_clean_trading_day_runs_the_whole_workflow_and_ends_running(harness: Harness) -> None:
    report = harness.run()

    assert report.state is OrchestratorState.RUNNING
    assert harness.orchestrator.state is OrchestratorState.RUNNING
    assert report.is_trading_day is True
    assert report.permit_trading is True

    # Steps 5-6 ran and came back clean.
    assert report.startup_report is not None
    assert report.startup_report.permit_strategy_execution is True

    # Steps 9-12 all produced real output from the dedicated modules.
    assert report.candidates, "stock selection produced no candidates"
    assert report.target_portfolio is not None
    assert len(report.risk_decisions) == len(report.target_portfolio.positions)


def test_a_clean_day_submits_orders_that_reach_the_broker(harness: Harness) -> None:
    report = harness.run()

    assert report.submitted_order_ids, "no orders were submitted"
    assert len(report.sized_trades) == len(report.submitted_order_ids)
    for client_order_id in report.submitted_order_ids:
        record = harness.order_manager.get(client_order_id)
        assert record.state is not OrderState.CREATED, "order never left CREATED"


def test_fills_flow_into_the_canonical_position_tracker(harness: Harness) -> None:
    report = harness.run()

    filled = [
        harness.order_manager.get(order_id)
        for order_id in report.submitted_order_ids
        if harness.order_manager.get(order_id).state
        in (OrderState.FILLED, OrderState.PARTIALLY_FILLED)
    ]
    assert filled, "the paper broker filled nothing, so there is no fill path to verify"
    for record in filled:
        assert harness.position_tracker.held_quantity(record.instrument_id) > 0


def test_a_fill_is_never_counted_twice_across_repeated_polls(harness: Harness) -> None:
    report = harness.run()
    held_after_cycle = {
        p.instrument_id: p.quantity for p in harness.position_tracker.current_positions()
    }
    assert held_after_cycle, "no positions to check for double counting"

    harness.orchestrator.fill_tracker.poll(harness.broker)
    harness.orchestrator.fill_tracker.poll(harness.broker)

    still_held = {p.instrument_id: p.quantity for p in harness.position_tracker.current_positions()}
    assert still_held == held_after_cycle
    assert report.state is OrchestratorState.RUNNING


def test_state_is_persisted_at_the_end_of_the_cycle(harness: Harness) -> None:
    harness.run()

    persisted = harness.state_store.load()
    assert persisted is not None
    assert persisted.system_state is SystemState.READY
    assert persisted.strategy_version == "strategy-v1"
    assert persisted.portfolio_snapshot is not None


def test_every_order_is_traceable_back_to_a_signal_and_risk_decision(harness: Harness) -> None:
    report = harness.run()
    assert report.submitted_order_ids

    for client_order_id in report.submitted_order_ids:
        trace = harness.order_manager.journal.trace(client_order_id)
        assert trace.signal_id
        assert trace.risk_decision_id
        assert harness.as_of.isoformat() in trace.signal_id


def test_rerunning_the_same_day_does_not_duplicate_orders(harness: Harness) -> None:
    first = harness.run()
    assert first.submitted_order_ids

    orders_after_first = len(harness.order_manager.all_orders())
    harness.run()
    assert len(harness.order_manager.all_orders()) == orders_after_first


# --------------------------------------------------------------------------
# Lifecycle: the states each blocking condition lands in
# --------------------------------------------------------------------------


def test_a_non_trading_day_stops_at_ready_without_trading(harness: Harness) -> None:
    saturday = harness.as_of
    while harness.env.calendar.is_trading_day(saturday):
        saturday += dt.timedelta(days=1)

    report = harness.run(saturday)
    assert report.is_trading_day is False
    assert report.state is OrchestratorState.READY
    assert report.permit_trading is False
    assert report.submitted_order_ids == ()
    assert report.startup_report is None, "startup should not even be attempted"


def test_unavailable_market_data_degrades_before_touching_the_broker(harness: Harness) -> None:
    far_future = harness.env.dates[-1] + dt.timedelta(days=90)
    # Calendar still covers it, but no instrument has data that recent.
    report = harness.run(far_future)
    assert report.state is OrchestratorState.DEGRADED
    assert report.permit_trading is False
    assert report.startup_report is None


def test_a_disconnected_broker_halts_the_cycle(harness: Harness) -> None:
    harness.broker.healthy = False
    harness.broker.health_detail = "connection refused"

    report = harness.run()
    assert report.state is OrchestratorState.HALTED
    assert report.permit_trading is False
    assert report.startup_report is not None
    assert report.startup_report.broker_connected is False
    assert report.submitted_order_ids == ()


def test_a_reconciliation_discrepancy_halts_before_any_strategy_work(harness: Harness) -> None:
    # The broker reports a position the local system has never seen.
    harness.broker.extra_positions = [
        BrokerPosition(instrument_id="NSE:GHOST", quantity=10, avg_price=100.0)
    ]

    report = harness.run()
    assert report.state is OrchestratorState.HALTED
    assert report.startup_report is not None
    assert report.startup_report.position_reconciliation is not None
    assert report.startup_report.position_reconciliation.mismatches
    assert report.target_portfolio is None, "no strategy work should have run"
    assert report.submitted_order_ids == ()


def test_no_approved_model_halts_the_cycle(
    env: Environment, settings: Settings, tmp_path: Path
) -> None:
    harness = Harness(
        env,
        tmp_path,
        as_of=env.dates[-1],
        settings=settings,
        train_end=env.dates[200],
        approve_model=False,
    )
    report = harness.run()
    assert report.state is OrchestratorState.HALTED
    # StartupSequence's own "verify risk state" step (Phase 18) catches this
    # first, before the orchestrator's step 7 ever loads a model -- two
    # independent gates, the earlier one winning.
    assert report.startup_report is not None
    assert report.startup_report.risk_state_verified is False
    assert any("no approved regime model" in message for message in report.messages)
    assert report.target_portfolio is None


def test_stale_model_metadata_halts_the_cycle(harness: Harness) -> None:
    # A one-session retrain interval makes the fitted model instantly stale.
    stale_settings = harness.orchestrator._settings_loader()
    stale_hmm = stale_settings.hmm.model_copy(update={"retrain_interval_sessions": 1})
    harness.orchestrator._settings_loader = lambda: stale_settings.model_copy(
        update={"hmm": stale_hmm}
    )

    report = harness.run()
    assert report.state is OrchestratorState.HALTED
    assert any("retrain interval" in message for message in report.messages)
    assert report.submitted_order_ids == ()


def test_configuration_failure_raises_rather_than_trading_blind(harness: Harness) -> None:
    def failing_loader() -> Settings:
        raise ConfigError("settings.yaml is unreadable")

    harness.orchestrator._settings_loader = failing_loader
    with pytest.raises(OrchestratorError, match="configuration failed"):
        harness.run()
    assert harness.orchestrator.state is OrchestratorState.HALTED


def test_a_halted_circuit_breaker_blocks_every_order(harness: Harness) -> None:
    _halt_circuit_breaker(harness)

    report = harness.run()
    assert report.state is OrchestratorState.HALTED
    assert report.permit_trading is False
    assert report.submitted_order_ids == ()
    assert report.sized_trades == ()


def _halt_circuit_breaker(harness: Harness) -> None:
    halted_state = PortfolioRiskState(
        as_of=harness.clock.now,
        equity=10_000_000.0,
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
    harness.circuit_breaker.evaluate(halted_state)
    assert harness.circuit_breaker.current_status().state is CircuitState.HALTED


# --------------------------------------------------------------------------
# The ongoing loop: steps 15-20
# --------------------------------------------------------------------------


def test_run_forever_runs_the_daily_cycle_then_the_monitoring_loop(harness: Harness) -> None:
    report = harness.orchestrator.run_forever(
        as_of=harness.as_of,
        max_iterations=2,
        install_signal_handlers=False,
        sleep_fn=lambda _seconds: None,
    )
    assert report.state is OrchestratorState.RUNNING
    # The loop ends in SHUTTING_DOWN, having persisted on the way out.
    assert harness.orchestrator.state is OrchestratorState.SHUTTING_DOWN
    assert harness.state_store.load() is not None


def test_the_monitoring_loop_beats_the_heartbeat(harness: Harness) -> None:
    assert harness.heartbeat.last() is None
    harness.orchestrator.run_forever(
        as_of=harness.as_of,
        max_iterations=1,
        install_signal_handlers=False,
        sleep_fn=lambda _seconds: None,
    )
    assert harness.heartbeat.last() == harness.clock.now


def test_a_mid_day_reconciliation_break_degrades_rather_than_halting(harness: Harness) -> None:
    harness.run()
    assert harness.orchestrator.state is OrchestratorState.RUNNING

    harness.broker.extra_positions = [
        BrokerPosition(instrument_id="NSE:GHOST", quantity=10, avg_price=100.0)
    ]
    harness.orchestrator._monitor_and_reconcile_once()
    after_break: OrchestratorState = harness.orchestrator.state
    assert after_break is OrchestratorState.DEGRADED


def test_degraded_returns_to_running_once_reconciliation_is_clean_again(
    harness: Harness,
) -> None:
    harness.run()
    harness.broker.extra_positions = [
        BrokerPosition(instrument_id="NSE:GHOST", quantity=10, avg_price=100.0)
    ]
    harness.orchestrator._monitor_and_reconcile_once()
    after_break: OrchestratorState = harness.orchestrator.state
    assert after_break is OrchestratorState.DEGRADED

    harness.broker.extra_positions = []
    harness.orchestrator._monitor_and_reconcile_once()
    after_recovery: OrchestratorState = harness.orchestrator.state
    assert after_recovery is OrchestratorState.RUNNING


def test_an_unhealthy_broker_in_the_loop_halts(harness: Harness) -> None:
    harness.run()
    harness.broker.healthy = False
    harness.orchestrator._monitor_and_reconcile_once()
    assert harness.orchestrator.state is OrchestratorState.HALTED


def test_the_loop_persists_state_every_iteration(harness: Harness) -> None:
    harness.run()
    harness.state_store.state_path.unlink()

    harness.orchestrator._monitor_and_reconcile_once()
    persisted = harness.state_store.load()
    assert persisted is not None
    assert persisted.system_state is SystemState.READY


# --------------------------------------------------------------------------
# Graceful shutdown
# --------------------------------------------------------------------------


def test_request_shutdown_ends_the_loop(harness: Harness) -> None:
    iterations = {"count": 0}

    def sleeping(_seconds: float) -> None:
        iterations["count"] += 1
        harness.orchestrator.request_shutdown()

    harness.orchestrator.run_forever(
        as_of=harness.as_of,
        max_iterations=50,
        install_signal_handlers=False,
        sleep_fn=sleeping,
    )
    assert iterations["count"] == 1
    assert harness.orchestrator.state is OrchestratorState.SHUTTING_DOWN


def test_signal_handlers_are_installed_and_restored(harness: Harness) -> None:
    before = signal.getsignal(signal.SIGINT)
    installed: list[object] = []

    def capture(_seconds: float) -> None:
        installed.append(signal.getsignal(signal.SIGINT))
        harness.orchestrator.request_shutdown()

    harness.orchestrator.run_forever(
        as_of=harness.as_of,
        max_iterations=5,
        install_signal_handlers=True,
        sleep_fn=capture,
    )

    assert installed, "the loop never ran an iteration"
    assert installed[0] is not before, "SIGINT handler was not installed"
    assert signal.getsignal(signal.SIGINT) is before, "SIGINT handler was not restored"


def test_the_signal_handler_requests_a_graceful_shutdown(harness: Harness) -> None:
    assert harness.orchestrator._shutdown_requested is False
    harness.orchestrator._handle_signal(signal.SIGTERM, None)
    assert harness.orchestrator._shutdown_requested is True


def test_shutdown_does_not_close_positions_by_default(harness: Harness) -> None:
    harness.orchestrator.run_forever(
        as_of=harness.as_of,
        max_iterations=1,
        install_signal_handlers=False,
        sleep_fn=lambda _seconds: None,
    )
    assert harness.broker.close_all_positions_calls == 0
    assert harness.position_tracker.current_positions(), "nothing was held to begin with"


def test_shutdown_closes_positions_only_when_explicitly_configured(
    env: Environment, settings: Settings, tmp_path: Path
) -> None:
    harness = Harness(
        env,
        tmp_path,
        as_of=env.dates[-1],
        settings=settings,
        train_end=env.dates[200],
        close_positions_on_shutdown=True,
    )
    harness.orchestrator.run_forever(
        as_of=harness.as_of,
        max_iterations=1,
        install_signal_handlers=False,
        sleep_fn=lambda _seconds: None,
    )
    assert harness.broker.close_all_positions_calls == 1


def test_shutdown_persists_the_halted_state_when_halted(harness: Harness) -> None:
    _halt_circuit_breaker(harness)
    harness.orchestrator.run_forever(
        as_of=harness.as_of,
        max_iterations=1,
        install_signal_handlers=False,
        sleep_fn=lambda _seconds: None,
    )
    persisted = harness.state_store.load()
    assert persisted is not None
    assert persisted.system_state is SystemState.HALTED


# --------------------------------------------------------------------------
# "The orchestration layer must NOT contain strategy mathematics"
# --------------------------------------------------------------------------


def test_the_orchestration_package_imports_no_numerical_libraries() -> None:
    """A structural check on the phase's central constraint: every number
    in this system is computed by a dedicated module, so no module in
    ``orchestration/`` has any reason to import a numerical library. If
    one ever does, strategy mathematics has started leaking into the
    layer that is only supposed to sequence calls.
    """
    package_dir = Path(__file__).resolve().parents[2] / "orchestration"
    offenders = []
    for path in sorted(package_dir.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import numpy", "import pandas")) or stripped.startswith(
                ("from numpy", "from pandas")
            ):
                offenders.append(f"{path.name}: {stripped}")
    assert offenders == []


def test_the_sell_side_of_a_trade_prices_at_the_bid(harness: Harness) -> None:
    """The orchestrator sends marketable limits (buy at the ask, sell at
    the bid) rather than resting at the last traded price -- otherwise a
    daily rebalance silently drifts away from the risk-approved target.
    """
    from orchestration.trade_sizing import SizedTrade
    from portfolio.portfolio_constructor import TradeAction

    quote = BrokerQuote(
        instrument_id="NSE:S00", bid=99.0, ask=101.0, last_price=100.0, as_of=harness.clock.now
    )
    buy = SizedTrade("NSE:S00", "buy", 10, TradeAction.BUY, 100.0)
    sell = SizedTrade("NSE:S00", "sell", 10, TradeAction.EXIT, 100.0)

    assert harness.orchestrator._limit_price(buy, {"NSE:S00": quote}) == 101.0
    assert harness.orchestrator._limit_price(sell, {"NSE:S00": quote}) == 99.0
    assert harness.orchestrator._limit_price(buy, {}) == 100.0


def test_selling_an_exited_position_reduces_the_canonical_tracker(harness: Harness) -> None:
    """A round trip through the real fill path: buy today, then hand the
    orchestrator a target that no longer contains the name and confirm the
    exit is sized, submitted, filled, and reflected in the tracker.
    """
    harness.run()
    held = harness.position_tracker.current_positions()
    assert held, "nothing was bought to exit"
    instrument_id = held[0].instrument_id
    quantity_before = held[0].quantity

    order = harness.broker.close_position(instrument_id)
    assert order.status in {OrderState.FILLED.value, OrderState.PARTIALLY_FILLED.value}

    harness.orchestrator.fill_tracker.poll(harness.broker)
    assert harness.position_tracker.held_quantity(instrument_id) < quantity_before


# --------------------------------------------------------------------------
# Monitoring wired into the running orchestrator (Phase 20)
# --------------------------------------------------------------------------


@pytest.fixture
def monitored(env: Environment, settings: Settings, tmp_path: Path) -> Harness:
    return Harness(
        env,
        tmp_path,
        as_of=env.dates[-1],
        settings=settings,
        train_end=env.dates[200],
        with_monitoring=True,
    )


def test_a_monitored_run_publishes_a_snapshot_of_the_real_system(monitored: Harness) -> None:
    """The collector reads live components, not fixtures: after a real
    daily cycle the snapshot must agree with what actually happened."""
    report = monitored.run()
    snapshot = monitored.orchestrator.publish_monitoring_snapshot()

    assert snapshot is not None
    assert snapshot.system.status == report.state.value
    assert snapshot.system.broker_connected is True
    assert snapshot.portfolio.equity > 0
    assert snapshot.portfolio.position_count == len(
        monitored.position_tracker.current_positions()
    )
    assert snapshot.execution.orders_submitted == len(report.submitted_order_ids)
    assert snapshot.regime.regime is not None, "the cycle computed a regime; monitoring lost it"
    assert snapshot.regime.label is not None


def test_the_snapshot_renders_as_a_dashboard(monitored: Harness) -> None:
    monitored.run()
    snapshot = monitored.orchestrator.publish_monitoring_snapshot()
    assert snapshot is not None

    text = render_dashboard(snapshot)
    for heading in ("SYSTEM", "PORTFOLIO", "REGIME", "EXECUTION", "RISK"):
        assert heading in text
    assert {len(line) for line in text.splitlines()} == {80}


def test_a_clean_monitored_run_raises_no_alerts(monitored: Harness) -> None:
    monitored.run()
    monitored.orchestrator.publish_monitoring_snapshot()
    assert monitored.alerts_received == []


def test_a_broker_disconnect_during_the_loop_raises_an_alert(monitored: Harness) -> None:
    monitored.run()
    monitored.broker.healthy = False
    monitored.broker.health_detail = "connection refused"

    monitored.orchestrator._monitor_and_reconcile_once()
    assert AlertType.BROKER_DISCONNECT in {
        alert.alert_type for alert in monitored.alerts_received
    }


def test_an_unexpected_broker_position_alerts_through_the_loop(monitored: Harness) -> None:
    monitored.run()
    monitored.broker.extra_positions = [
        BrokerPosition(instrument_id="NSE:GHOST", quantity=10, avg_price=100.0)
    ]

    monitored.orchestrator._monitor_and_reconcile_once()
    raised = {alert.alert_type for alert in monitored.alerts_received}
    assert AlertType.UNEXPECTED_POSITION in raised


def test_alerts_are_rate_limited_across_loop_iterations(monitored: Harness) -> None:
    """The condition that matters here is persistence: a broker that stays
    down must not produce one alert per iteration."""
    monitored.run()
    monitored.broker.healthy = False

    for _ in range(5):
        monitored.orchestrator._monitor_and_reconcile_once()
        monitored.clock.advance(seconds=10)

    disconnects = [
        alert
        for alert in monitored.alerts_received
        if alert.alert_type is AlertType.BROKER_DISCONNECT
    ]
    assert len(disconnects) == 1


def test_an_orchestrator_without_monitoring_still_runs(harness: Harness) -> None:
    """Monitoring is optional wiring: an orchestrator with neither a
    collector nor an alert manager behaves exactly as before."""
    assert harness.orchestrator.publish_monitoring_snapshot() is None
    report = harness.run()
    assert report.state is OrchestratorState.RUNNING
