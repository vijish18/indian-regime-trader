"""Wires the full system for the end-to-end paper-trading validation
(Phase 21), using this repository's own ``config/settings.yaml`` --
production configuration, unmodified -- against a synthetic vendor drop
(``validation.synthetic_market``) ingested through the real
``data.ingestion.DataIngestionPipeline``.

**No live credentials, structurally, not by convention.** This module
never imports ``broker.zerodha`` and never reads ``BROKER_API_KEY``/
``BROKER_API_SECRET``; the only broker it ever constructs is
``broker.adapters.paper_broker.PaperBroker``. ``broker.factory.build_broker``
is not used here only because it hardcodes a wall-clock time source and
this harness needs a controllable one for a deterministic, reproducible
run -- ``settings.execution.mode`` is asserted to be ``"paper"`` before
anything is built, as a second, independent guard against ever
constructing a live adapter by accident.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from backtest.cost_schedule import CostScheduleRepository
from backtest.costs import CostModel
from broker.adapters.paper_broker import PaperBroker
from config.loader import load_settings
from config.models import Settings
from core.features.feature_engineering import (
    FeaturePipeline,
    MarketFeatureInputs,
    build_default_feature_definitions,
    drop_warmup_rows,
    feature_set_version,
)
from core.features.feature_scaler import CausalFeatureScaler
from core.regime.hmm_engine import HMMRegimeEngine
from core.regime.model_registry import ModelArtifact, ModelRegistry, build_model_id
from core.regime.regime_policy import RegimePolicy
from data.calendar import NSETradingCalendar
from data.corporate_actions import InMemoryCorporateActionProvider
from data.ingestion import DataIngestionPipeline
from data.instrument_master import InMemoryInstrumentRepository
from data.market_data import LocalMarketDataProvider
from data.membership import InMemoryIndexMembershipProvider
from data.storage import LocalDataStore, StorageFormat
from execution.order_manager import OrderManager
from execution.position_tracker import PositionTracker
from execution.reconciliation import ReconciliationEngine
from execution.startup import StartupSequence
from execution.system_state import SystemStateStore
from monitoring.alerts import AlertManager
from monitoring.health import HealthChecker
from monitoring.snapshot import SnapshotCollector
from orchestration.heartbeat import Heartbeat
from orchestration.orchestrator import Orchestrator
from orchestration.regime_computation import RegimeComputer
from portfolio.portfolio_constructor import PortfolioConstructor
from risk.circuit_breaker import CircuitBreaker
from risk.risk_manager import RiskManager
from risk.risk_state_builder import EquityHistory
from universe.stock_selector import StockSelector
from universe.universe import UniverseProvider
from validation.failures import DisconnectableFeed, FaultInjectingBroker
from validation.synthetic_market import (
    INDEX_SYMBOL,
    VIX_SYMBOL,
    SyntheticMarket,
    write_synthetic_market,
)

STRATEGY_VERSION = "validation-phase-21"


class _Clock:
    """A settable clock every wired-up component shares, so the whole
    session (and every timestamp the generated report carries) advances
    deterministically rather than by real wall-clock time.
    """

    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + dt.timedelta(**kwargs)

    def set(self, moment: dt.datetime) -> None:
        self.now = moment


@dataclass
class ValidationEnvironment:
    """Every wired component the validation scenario drives or inspects.
    ``rebuild_after_crash`` is the one mutator -- everything else is
    assembled once by :func:`build_validation_environment`.
    """

    settings: Settings
    root: Path
    market: SyntheticMarket
    calendar: NSETradingCalendar
    store: LocalDataStore
    market_data: LocalMarketDataProvider
    feed: DisconnectableFeed
    universe_provider: UniverseProvider
    feature_pipeline: FeaturePipeline
    stock_selector: StockSelector
    portfolio_constructor: PortfolioConstructor
    regime_policy: RegimePolicy
    model_registry: ModelRegistry
    model_artifact: ModelArtifact
    train_end: dt.date
    clock: _Clock
    broker: FaultInjectingBroker
    broker_position_tracker: PositionTracker
    """``PaperBroker``'s own tracker, used only for its internal order
    validation (reject a sell beyond held quantity, etc). Never shared
    with the canonical tracker below -- see ``orchestration/fill_tracker.py``
    for why: a paper fill applied to both would be counted twice."""

    circuit_breaker: CircuitBreaker
    state_store: SystemStateStore
    equity_history: EquityHistory

    position_tracker: PositionTracker
    order_manager: OrderManager
    heartbeat: Heartbeat
    health_checker: HealthChecker
    reconciliation_engine: ReconciliationEngine
    alert_manager: AlertManager
    snapshot_collector: SnapshotCollector
    orchestrator: Orchestrator

    def rebuild_after_crash(self) -> None:
        """Simulates a process crash and restart: every in-memory Python
        object is discarded and rebuilt from scratch, *except* the broker
        (a real process would still be a separate service with its own
        state) and everything persisted to disk (the circuit-breaker file,
        ``SystemStateStore``, the model registry). This is the literal
        meaning of "application crash" for this architecture -- Phase 18's
        entire restart-recovery design exists to make exactly this safe.
        """
        self.position_tracker = PositionTracker()
        self.order_manager = OrderManager(clock=self.clock)
        self.heartbeat = Heartbeat(self.clock)
        self.equity_history = EquityHistory()
        self.reconciliation_engine = ReconciliationEngine(
            self.position_tracker, self.order_manager, self.broker, clock=self.clock
        )
        self.health_checker = HealthChecker(
            self.broker,
            self.market_data,
            self.universe_provider,
            self.model_registry,
            self.calendar,
            self.market.instrument_ids,
            max_stale_sessions=1,
            model_max_age_sessions=self.settings.hmm.retrain_interval_sessions,
            heartbeat_interval_seconds=self.settings.monitoring.heartbeat_interval_seconds,
            last_heartbeat=self.heartbeat.last,
            clock=self.clock,
        )
        self.alert_manager = AlertManager(
            list(self.settings.monitoring.alert_channels),
            cooldown_seconds=self.settings.monitoring.alert_cooldown_seconds,
            drawdown_alert_pct=self.settings.risk.peak_to_trough_drawdown_halt_pct * 0.5,
            clock=self.clock,
        )
        self.snapshot_collector = SnapshotCollector(
            broker=self.broker,
            position_tracker=self.position_tracker,
            order_manager=self.order_manager,
            circuit_breaker=self.circuit_breaker,
            reconciliation_engine=self.reconciliation_engine,
            health_checker=self.health_checker,
            calendar=self.calendar,
            market_data=self.market_data,
            model_registry=self.model_registry,
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
            calendar=self.calendar,
            market_data=self.market_data,
            stock_selector=self.stock_selector,
            regime_computer=RegimeComputer(
                self.market_data,
                self.settings.hmm,
                self.settings.allocation,
                self.regime_policy,
                self.feature_pipeline,
                INDEX_SYMBOL,
                VIX_SYMBOL,
                feature_warmup_buffer_days=400,
            ),
            portfolio_constructor=self.portfolio_constructor,
            risk_manager=RiskManager(self.settings.risk, self.circuit_breaker),
            circuit_breaker=self.circuit_breaker,
            model_registry=self.model_registry,
            position_tracker=self.position_tracker,
            order_manager=self.order_manager,
            broker=self.broker,
            state_store=self.state_store,
            health_checker=self.health_checker,
            execution_config=self.settings.execution,
            strategy_version=STRATEGY_VERSION,
            settings_loader=lambda: self.settings,
            heartbeat=self.heartbeat,
            equity_history=self.equity_history,
            snapshot_collector=self.snapshot_collector,
            alert_manager=self.alert_manager,
            close_positions_on_shutdown=False,
            clock=self.clock,
        )

    def startup_sequence(self) -> StartupSequence:
        """A standalone ``StartupSequence`` sharing this environment's live
        components -- used by the scenario to test restart recovery in
        isolation from a full daily cycle (e.g. right after a simulated
        database-file corruption, before anything else has been decided
        for the day).
        """
        return self.orchestrator.startup_sequence


_EARLIEST_COST_SCHEDULE = dt.date(2019, 1, 1)
"""``config/cost_schedules.yaml``'s earliest dated entry -- an order dated
before this has no rates to price it with
(``backtest.cost_schedule.MissingCostScheduleError``). The synthetic
market's default start is chosen with a safety margin past it."""


def build_validation_environment(
    root: Path,
    *,
    sessions: int = 1100,
    train_end_index: int = 900,
    start: dt.date = dt.date(2019, 6, 1),
    seed: int = 5,
) -> ValidationEnvironment:
    if start < _EARLIEST_COST_SCHEDULE:
        raise ValueError(
            f"start={start} predates the earliest entry in config/cost_schedules.yaml "
            f"({_EARLIEST_COST_SCHEDULE}); paper orders on those dates would have no "
            "cost schedule to price against"
        )
    settings = load_settings()
    if settings.execution.mode != "paper":
        raise RuntimeError(
            f"validation refuses to run with execution.mode={settings.execution.mode!r}; "
            "only 'paper' is permitted here"
        )

    market = write_synthetic_market(root / "vendor", start=start, sessions=sessions, seed=seed)
    calendar = NSETradingCalendar(
        holidays={}, covered_years=frozenset(range(start.year - 1, market.last_session.year + 2))
    )
    store = LocalDataStore(
        root / "data" / "raw",
        root / "data" / "normalized",
        root / "data" / "reference",
        StorageFormat(settings.data.storage_format),
    )
    _ingest(store, calendar, market)

    # No corporate actions exist in the synthetic drop, but stock selection
    # requests PriceBasis.ADJUSTED unconditionally -- without a provider
    # (even an empty one), LocalMarketDataProvider fails closed on every
    # adjusted-price request, which would silently exclude every candidate
    # rather than genuinely rank them. An empty provider means "no
    # adjustments happened," which is the truth here.
    market_data = LocalMarketDataProvider(store, InMemoryCorporateActionProvider([]))
    clock = _Clock(dt.datetime.combine(market.last_session, dt.time(16, 0), tzinfo=dt.UTC))
    feed = DisconnectableFeed(market_data, market.last_session, clock=clock)

    universe_provider = UniverseProvider(
        INDEX_SYMBOL,
        InMemoryIndexMembershipProvider.from_store(store),
        InMemoryInstrumentRepository.from_store(store),
    )
    feature_pipeline = FeaturePipeline(build_default_feature_definitions(settings.features))
    stock_selector = StockSelector(
        settings.selection, settings.universe, universe_provider, market_data
    )
    portfolio_constructor = PortfolioConstructor(
        settings.portfolio, settings.selection, settings.execution, market_data
    )
    regime_policy = RegimePolicy(settings.regime_policy)

    train_end = market.sessions[train_end_index]
    model_registry = ModelRegistry(root / "models")
    artifact = _fit_and_approve_model(
        market_data, settings, feature_pipeline, model_registry, market.first_session, train_end
    )

    cost_model = CostModel(
        CostScheduleRepository.from_file(Path(settings.costs.schedule_file)),
        min_slippage_bps=settings.backtest.slippage_min_bps,
        impact_coefficient=settings.backtest.slippage_impact_coefficient,
    )
    broker_position_tracker = PositionTracker()
    paper_broker = PaperBroker(
        market_data=feed,
        cost_model=cost_model,
        execution_config=settings.execution,
        paper_config=settings.paper_trading,
        position_tracker=broker_position_tracker,
        initial_cash=50_000_000.0,
        clock=clock,
    )
    broker = FaultInjectingBroker(paper_broker)

    circuit_breaker = CircuitBreaker(settings.risk, root / "circuit_breaker.json")
    state_store = SystemStateStore(root / "system_state.json")
    equity_history = EquityHistory()
    position_tracker = PositionTracker()
    order_manager = OrderManager(clock=clock)
    heartbeat = Heartbeat(clock)
    reconciliation_engine = ReconciliationEngine(
        position_tracker, order_manager, broker, clock=clock
    )
    health_checker = HealthChecker(
        broker,
        market_data,
        universe_provider,
        model_registry,
        calendar,
        market.instrument_ids,
        max_stale_sessions=1,
        model_max_age_sessions=settings.hmm.retrain_interval_sessions,
        heartbeat_interval_seconds=settings.monitoring.heartbeat_interval_seconds,
        last_heartbeat=heartbeat.last,
        clock=clock,
    )
    alert_manager = AlertManager(
        list(settings.monitoring.alert_channels),
        cooldown_seconds=settings.monitoring.alert_cooldown_seconds,
        drawdown_alert_pct=settings.risk.peak_to_trough_drawdown_halt_pct * 0.5,
        clock=clock,
    )

    env = ValidationEnvironment(
        settings=settings,
        root=root,
        market=market,
        calendar=calendar,
        store=store,
        market_data=market_data,
        feed=feed,
        universe_provider=universe_provider,
        feature_pipeline=feature_pipeline,
        stock_selector=stock_selector,
        portfolio_constructor=portfolio_constructor,
        regime_policy=regime_policy,
        model_registry=model_registry,
        model_artifact=artifact,
        train_end=train_end,
        clock=clock,
        broker=broker,
        broker_position_tracker=broker_position_tracker,
        circuit_breaker=circuit_breaker,
        state_store=state_store,
        equity_history=equity_history,
        position_tracker=position_tracker,
        order_manager=order_manager,
        heartbeat=heartbeat,
        health_checker=health_checker,
        reconciliation_engine=reconciliation_engine,
        alert_manager=alert_manager,
        snapshot_collector=None,  # type: ignore[arg-type]  # built just below
        orchestrator=None,  # type: ignore[arg-type]  # built by rebuild_after_crash
    )
    env.rebuild_after_crash()  # not a crash yet -- just the shared way to (re)build these
    return env


def _ingest(store: LocalDataStore, calendar: NSETradingCalendar, market: SyntheticMarket) -> None:
    pipeline = DataIngestionPipeline(store, calendar)
    results = [
        *(pipeline.ingest_equity_bars(path) for path in market.bar_files),
        pipeline.ingest_index_observations(market.index_file, index_symbol=INDEX_SYMBOL),
        pipeline.ingest_index_observations(market.vix_file, index_symbol=VIX_SYMBOL),
        pipeline.ingest_instruments(market.instruments_file),
        pipeline.ingest_index_membership(market.membership_file),
    ]
    rejected = [result for result in results if not result.accepted]
    if rejected:
        raise RuntimeError(
            f"synthetic vendor data failed ingestion validation: "
            f"{[r.summary() for r in rejected]}"
        )


def _fit_and_approve_model(
    market_data: LocalMarketDataProvider,
    settings: Settings,
    feature_pipeline: FeaturePipeline,
    registry: ModelRegistry,
    history_start: dt.date,
    train_end: dt.date,
) -> ModelArtifact:
    """The same fit recipe ``backtest.walk_forward.WalkForwardValidator._fit_fold``
    and Phase 19/20's own test harnesses use: fit strictly on
    ``[history_start, train_end]``, nothing after it.
    """
    nifty = market_data.get_index_observations(INDEX_SYMBOL, history_start, train_end)
    vix = market_data.get_index_observations(VIX_SYMBOL, history_start, train_end)
    inputs = MarketFeatureInputs.from_index_observations(nifty, vix)
    matrix = drop_warmup_rows(feature_pipeline.compute(inputs))
    scaled, scaler_params = CausalFeatureScaler().fit_transform(matrix)
    returns = inputs.frame["nifty_close"].pct_change().reindex(scaled.index)

    model = HMMRegimeEngine(settings.hmm).fit(scaled, returns)
    feature_version = feature_set_version(feature_pipeline.definitions)
    artifact = ModelArtifact(
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
        notes="Phase 21 end-to-end validation -- fitted on synthetic data, never for real use",
    )
    registry.save(artifact)
    registry.approve(artifact.model_id)
    return artifact
