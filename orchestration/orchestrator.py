"""The application lifecycle and daily workflow (Phase 19).

``Orchestrator`` sequences the 20 steps of a trading day, from loading
configuration through to periodic monitoring/reconciliation, but computes
nothing itself: every decision (is today a trading day, what's the regime,
which stocks rank highest, what should the portfolio look like, is a trade
approved, how many shares does that mean) is delegated to the dedicated
module that already owns it --

    1.  load configuration               -> config.loader.load_settings
    2.  validate configuration            -> (pydantic validation, part of load_settings)
    3.  verify market calendar            -> data.interfaces.TradingCalendar
    4.  verify data availability          -> monitoring.health.HealthChecker
    5.  verify broker connectivity        -> execution.startup.StartupSequence
    6.  reconcile portfolio               -> execution.startup.StartupSequence
    7.  load the correct model            -> core.regime.model_registry.ModelRegistry
    8.  validate model metadata           -> orchestration.model_validation
    9.  compute market regime             -> orchestration.regime_computation.RegimeComputer
    10. compute stock rankings            -> universe.stock_selector.StockSelector
    11. construct target portfolio        -> portfolio.portfolio_constructor.PortfolioConstructor
    12. run risk engine                   -> risk.risk_manager.RiskManager
    13. calculate required trades         -> portfolio_constructor.required_trades,
                                              orchestration.trade_sizing
    14. submit permitted orders           -> execution.order_manager.OrderManager
    15. track fills                       -> orchestration.fill_tracker.FillTracker
    16. update portfolio                  -> execution.position_tracker.PositionTracker
    17. update stops/risk rules           -> risk.circuit_breaker.CircuitBreaker (see below)
    18. persist state                     -> execution.startup.StartupSequence.persist_heartbeat
    19. monitor health                    -> monitoring.health.HealthChecker
    20. reconcile periodically            -> execution.reconciliation.ReconciliationEngine

Steps 1-6 (through "reconcile portfolio") are steps 5-6 of
``execution.startup.StartupSequence``'s own 13-step sequence in
everything but name -- rather than re-implement config/broker/reconciliation
verification a second time, ``Orchestrator`` constructs and runs a
``StartupSequence`` internally and adds only what it does not already
cover: the market-calendar and data-availability checks.

**Step 17 ("update stops/risk rules where applicable") is honest about
what exists.** No trailing-stop or resting protective-stop concept exists
anywhere in this codebase (docs/SPECIFICATION.md section 1.2 explicitly
demotes live stop orders to a last-resort control, and
``risk/position_sizer.py``'s future stop-distance formula, Phase 7c, is a
*sizing* input, not a resting order). This step is the defined seam a
future stop-loss module would plug into; today it re-marks positions to
the broker's latest reported prices and re-evaluates the circuit breaker,
so the risk rule that *does* exist (drawdown-based halting) reflects the
fills just observed rather than this morning's picture.
"""

from __future__ import annotations

import datetime as dt
import logging
import signal
import time
from collections.abc import Callable

from broker.base import Broker, BrokerQuote
from config.loader import ConfigError, load_settings
from config.models import ExecutionConfig, Settings
from core.features.feature_engineering import feature_set_version
from core.regime.allocation import AllocationRegime
from core.regime.model_registry import ModelRegistry, NoApprovedModelError
from data.errors import DataNotAvailableError
from data.interfaces import MarketDataProvider, TradingCalendar
from execution.order_manager import DuplicateSignalError, OrderManager, OrderState
from execution.order_reconciler import OrderReconciler
from execution.position_tracker import Position, PositionTracker
from execution.reconciliation import ReconciliationEngine, ReconciliationStatus
from execution.startup import StartupReport, StartupSequence
from execution.system_state import SystemState, SystemStateStore
from monitoring.health import ComponentHealth, HealthChecker
from orchestration.fill_tracker import FillTracker
from orchestration.heartbeat import Heartbeat
from orchestration.model_validation import validate_model_metadata
from orchestration.orchestrator_state import OrchestratorState
from orchestration.portfolio_snapshot import current_target_portfolio
from orchestration.regime_computation import RegimeComputationError, RegimeComputer
from orchestration.trade_sizing import SizedTrade, size_trades
from portfolio.portfolio_constructor import (
    PortfolioConstructionError,
    PortfolioConstructor,
    RequiredTrade,
    TargetPortfolio,
    TradeAction,
    required_trades,
)
from risk.circuit_breaker import CircuitBreaker, CircuitState
from risk.risk_manager import RiskDecision, RiskManager
from risk.risk_state_builder import EquityHistory, build_risk_state
from universe.factor_calculator import InsufficientFactorHistoryError
from universe.stock_selector import StockScore, StockSelector

logger = logging.getLogger(__name__)


class OrchestratorError(RuntimeError):
    """A daily-cycle step failed in a way that makes it impossible to even
    attempt the workflow -- configuration, calendar coverage. Everything
    else (no data, no approved model, insufficient regime history, risk
    rejection, broker/reconciliation trouble) is reported through
    ``DailyCycleReport`` instead, with ``permit_trading=False``, never
    raised.
    """


class DailyCycleReport:
    """A snapshot of one ``run_daily_cycle`` call's outcome. Plain class
    (not frozen) so fields can be filled in incrementally as the workflow
    progresses and returned early from whichever step first blocks it.
    """

    def __init__(
        self,
        as_of: dt.date,
        state: OrchestratorState,
        *,
        is_trading_day: bool = True,
        permit_trading: bool = False,
        startup_report: StartupReport | None = None,
        candidates: tuple[StockScore, ...] = (),
        target_portfolio: TargetPortfolio | None = None,
        risk_decisions: tuple[RiskDecision, ...] = (),
        sized_trades: tuple[SizedTrade, ...] = (),
        submitted_order_ids: tuple[str, ...] = (),
        skipped_trades: tuple[str, ...] = (),
        messages: tuple[str, ...] = (),
    ) -> None:
        self.as_of = as_of
        self.state = state
        self.is_trading_day = is_trading_day
        self.permit_trading = permit_trading
        self.startup_report = startup_report
        self.candidates = candidates
        self.target_portfolio = target_portfolio
        self.risk_decisions = risk_decisions
        self.sized_trades = sized_trades
        self.submitted_order_ids = submitted_order_ids
        self.skipped_trades = skipped_trades
        self.messages = messages


class Orchestrator:
    def __init__(
        self,
        *,
        calendar: TradingCalendar,
        market_data: MarketDataProvider,
        stock_selector: StockSelector,
        regime_computer: RegimeComputer,
        portfolio_constructor: PortfolioConstructor,
        risk_manager: RiskManager,
        circuit_breaker: CircuitBreaker,
        model_registry: ModelRegistry,
        position_tracker: PositionTracker,
        order_manager: OrderManager,
        broker: Broker,
        state_store: SystemStateStore,
        health_checker: HealthChecker,
        execution_config: ExecutionConfig,
        strategy_version: str,
        settings_loader: Callable[[], Settings] = load_settings,
        order_reconciler: OrderReconciler | None = None,
        equity_history: EquityHistory | None = None,
        heartbeat: Heartbeat | None = None,
        close_positions_on_shutdown: bool = False,
        min_order_value_inr: float = 0.0,
        clock: Callable[[], dt.datetime] | None = None,
    ) -> None:
        self.calendar = calendar
        self.market_data = market_data
        self.stock_selector = stock_selector
        self.regime_computer = regime_computer
        self.portfolio_constructor = portfolio_constructor
        self.risk_manager = risk_manager
        self.circuit_breaker = circuit_breaker
        self.model_registry = model_registry
        self.position_tracker = position_tracker
        self.order_manager = order_manager
        self.broker = broker
        self.state_store = state_store
        self.health_checker = health_checker
        self.execution_config = execution_config
        self.strategy_version = strategy_version
        self._settings_loader = settings_loader
        self.order_reconciler = order_reconciler or OrderReconciler(clock=clock)
        self.equity_history = equity_history or EquityHistory()
        self.heartbeat = heartbeat or Heartbeat(clock)
        self.close_positions_on_shutdown = close_positions_on_shutdown
        self.min_order_value_inr = min_order_value_inr
        self._clock: Callable[[], dt.datetime] = clock or (lambda: dt.datetime.now(dt.UTC))

        self.startup_sequence = StartupSequence(
            state_store=state_store,
            position_tracker=position_tracker,
            order_manager=order_manager,
            broker=broker,
            circuit_breaker=circuit_breaker,
            model_registry=model_registry,
            strategy_version=strategy_version,
            order_reconciler=self.order_reconciler,
            clock=self._clock,
            settings_loader=settings_loader,
        )
        self.reconciliation_engine = ReconciliationEngine(
            position_tracker, order_manager, broker, order_reconciler=self.order_reconciler,
            clock=self._clock,
        )
        self.fill_tracker = FillTracker(position_tracker)

        self.state = OrchestratorState.STARTING
        self._shutdown_requested = False
        self._previous_signal_handlers: dict[int, object] = {}

    # -- lifecycle / signals --------------------------------------------

    def request_shutdown(self) -> None:
        self._shutdown_requested = True

    def _handle_signal(self, signum: int, frame: object) -> None:
        logger.warning(
            "received signal %s; requesting graceful shutdown",
            signum,
            extra={"extra_fields": {"event": "orchestrator_signal_received", "signum": signum}},
        )
        self.request_shutdown()

    def install_signal_handlers(self) -> None:
        self._previous_signal_handlers[signal.SIGINT] = signal.signal(
            signal.SIGINT, self._handle_signal
        )
        if hasattr(signal, "SIGTERM"):
            self._previous_signal_handlers[signal.SIGTERM] = signal.signal(
                signal.SIGTERM, self._handle_signal
            )

    def restore_signal_handlers(self) -> None:
        for sig, handler in self._previous_signal_handlers.items():
            signal.signal(sig, handler)  # type: ignore[arg-type]
        self._previous_signal_handlers.clear()

    def run_forever(
        self,
        *,
        as_of: dt.date | None = None,
        poll_interval_seconds: float = 60.0,
        max_iterations: int | None = None,
        install_signal_handlers: bool = True,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> DailyCycleReport:
        """Runs one daily cycle, then loops steps 15-20 (fill tracking,
        portfolio update, stop/risk-rule refresh, persistence, health
        monitoring, periodic reconciliation) until shutdown is requested
        (SIGINT/SIGTERM, or ``request_shutdown()``) or ``max_iterations``
        is reached. Positions are never closed on shutdown unless
        ``close_positions_on_shutdown`` was set at construction.
        """
        if install_signal_handlers:
            self.install_signal_handlers()
        try:
            report = self.run_daily_cycle(as_of)
            iterations = 0
            while not self._shutdown_requested:
                if max_iterations is not None and iterations >= max_iterations:
                    break
                self._monitor_and_reconcile_once()
                iterations += 1
                if self._shutdown_requested:
                    break
                if max_iterations is not None and iterations >= max_iterations:
                    break
                sleep_fn(poll_interval_seconds)
        finally:
            self._shutdown(install_signal_handlers)
        return report

    def _shutdown(self, restore_handlers: bool) -> None:
        was_halted = self.state is OrchestratorState.HALTED
        self.state = OrchestratorState.SHUTTING_DOWN
        logger.warning(
            "shutting down; draining and persisting final state",
            extra={"extra_fields": {"event": "orchestrator_shutting_down"}},
        )
        try:
            self.startup_sequence.persist_heartbeat(
                SystemState.HALTED if was_halted else SystemState.READY
            )
        except Exception:
            logger.exception("failed to persist final state during shutdown")

        if self.close_positions_on_shutdown:
            logger.warning(
                "close_positions_on_shutdown is enabled; closing all open positions",
                extra={"extra_fields": {"event": "orchestrator_closing_positions_on_shutdown"}},
            )
            try:
                self.broker.close_all_positions()
            except Exception:
                logger.exception("failed to close positions during shutdown")

        if restore_handlers:
            self.restore_signal_handlers()

    # -- the daily cycle: steps 1-14, plus one immediate fill sweep -----

    def run_daily_cycle(self, as_of: dt.date | None = None) -> DailyCycleReport:
        self.state = OrchestratorState.STARTING
        messages: list[str] = []

        # Steps 1-2: load + validate configuration. One call does both --
        # config.loader.load_settings runs the JSON-schema check and every
        # pydantic invariant before returning.
        try:
            settings = self._settings_loader()
        except ConfigError as exc:
            self.state = OrchestratorState.HALTED
            raise OrchestratorError(f"configuration failed to load/validate: {exc}") from exc
        messages.append("configuration loaded and validated")

        as_of = as_of or self._clock().date()
        self.state = OrchestratorState.HEALTH_CHECK

        # Step 3: verify market calendar.
        is_trading_day = self.calendar.is_trading_day(as_of)
        if not is_trading_day:
            messages.append(f"{as_of} is not a trading day; nothing to do")
            self.state = OrchestratorState.READY
            return DailyCycleReport(
                as_of, self.state, is_trading_day=False, permit_trading=False,
                messages=tuple(messages),
            )

        # Step 4: verify data availability.
        freshness = self.health_checker.check_market_data_freshness(as_of)
        messages.append(f"market data freshness: {freshness.status.value} ({freshness.detail})")
        if freshness.status is ComponentHealth.UNHEALTHY:
            self.state = OrchestratorState.DEGRADED
            return DailyCycleReport(as_of, self.state, messages=tuple(messages))

        # Steps 5-6: verify broker connectivity + reconcile portfolio,
        # delegated entirely to StartupSequence (Phase 18), which also
        # rebuilds portfolio state and verifies risk/circuit-breaker state
        # once reconciliation is clean.
        self.state = OrchestratorState.RECONCILING
        startup_report = self.startup_sequence.run()
        messages.extend(startup_report.messages)
        if not startup_report.permit_strategy_execution:
            self.state = OrchestratorState.HALTED
            return DailyCycleReport(
                as_of, self.state, startup_report=startup_report, messages=tuple(messages)
            )
        self.state = OrchestratorState.READY

        return self._run_strategy_pipeline(as_of, settings, startup_report, messages)

    def _run_strategy_pipeline(
        self,
        as_of: dt.date,
        settings: Settings,
        startup_report: StartupReport,
        messages: list[str],
    ) -> DailyCycleReport:
        self.state = OrchestratorState.RUNNING

        # Step 7: load the correct model.
        try:
            artifact = self.model_registry.load_current_approved()
        except NoApprovedModelError as exc:
            messages.append(f"no approved model: {exc}")
            self.state = OrchestratorState.HALTED
            return DailyCycleReport(
                as_of, self.state, startup_report=startup_report, messages=tuple(messages)
            )

        # Step 8: validate model metadata.
        feature_definitions = self.regime_computer.feature_pipeline.definitions
        problems = validate_model_metadata(
            artifact,
            [d.name for d in feature_definitions],
            feature_set_version(feature_definitions),
            as_of=as_of,
            calendar=self.calendar,
            max_age_sessions=settings.hmm.retrain_interval_sessions,
        )
        if problems:
            messages.extend(problems)
            self.state = OrchestratorState.HALTED
            return DailyCycleReport(
                as_of, self.state, startup_report=startup_report, messages=tuple(messages)
            )

        # Step 9: compute market regime.
        try:
            regime_target, regime_state = self.regime_computer.compute_today(artifact, as_of)
        except RegimeComputationError as exc:
            messages.append(str(exc))
            self.state = OrchestratorState.DEGRADED
            return DailyCycleReport(
                as_of, self.state, startup_report=startup_report, messages=tuple(messages)
            )
        messages.append(
            f"regime: {regime_target.regime.value} "
            f"(exposure={regime_target.target_gross_exposure:.2%}, "
            f"confidence={regime_state.confidence:.2f}, reason={regime_target.reason})"
        )

        # Step 10: compute stock rankings.
        try:
            candidates = self.stock_selector.select(as_of)
        except (DataNotAvailableError, InsufficientFactorHistoryError) as exc:
            messages.append(f"stock selection failed: {exc}")
            self.state = OrchestratorState.DEGRADED
            return DailyCycleReport(
                as_of, self.state, startup_report=startup_report, messages=tuple(messages)
            )

        equity = self.broker.get_account().equity
        self.equity_history.append(equity)
        current_positions = self.position_tracker.current_positions()
        current_portfolio = (
            current_target_portfolio(current_positions, equity, as_of, regime_target.regime)
            if current_positions
            else None
        )

        # Step 11: construct target portfolio.
        try:
            target_portfolio = self.portfolio_constructor.construct(
                candidates, regime_target, as_of, equity, current_portfolio=current_portfolio
            )
        except (ValueError, PortfolioConstructionError) as exc:
            messages.append(f"portfolio construction failed: {exc}")
            self.state = OrchestratorState.DEGRADED
            return DailyCycleReport(
                as_of, self.state, startup_report=startup_report, candidates=tuple(candidates),
                messages=tuple(messages),
            )

        # Step 12: run risk engine.
        risk_state = build_risk_state(
            target_portfolio,
            equity,
            self._clock(),
            self.market_data,
            self.equity_history,
            assumed_spread_bps=self.execution_config.order_price_guard_bps,
        )
        decisions = self.risk_manager.evaluate(
            target_portfolio, risk_state, current=current_portfolio
        )
        decisions_by_instrument = {decision.instrument_id: decision for decision in decisions}
        circuit_status = self.circuit_breaker.current_status()
        circuit_halted = circuit_status.state is CircuitState.HALTED
        if circuit_halted:
            messages.append(f"circuit breaker HALTED: {circuit_status.reason}")
            self.state = OrchestratorState.HALTED

        # Step 13: calculate required trades.
        weight_trades = required_trades(target_portfolio, current_portfolio)
        quotes = self._fetch_quotes(weight_trades)
        sized_trades, skip_reasons = self._size_trades(
            weight_trades, decisions_by_instrument, current_positions, equity, quotes,
            circuit_halted=circuit_halted,
        )
        messages.extend(skip_reasons)

        # Step 14: submit permitted orders.
        submitted_order_ids = self._submit_orders(
            as_of, sized_trades, decisions_by_instrument, quotes, messages
        )

        # An immediate fill sweep for whatever just filled synchronously
        # (paper trading) so today's own report already reflects it --
        # the ongoing loop (steps 15/16) keeps doing this for anything
        # that fills later.
        self.fill_tracker.poll(self.broker)

        if self.state is not OrchestratorState.HALTED:
            self.state = OrchestratorState.RUNNING
        self.startup_sequence.persist_heartbeat(
            SystemState.HALTED if self.state is OrchestratorState.HALTED else SystemState.READY
        )

        return DailyCycleReport(
            as_of,
            self.state,
            permit_trading=not circuit_halted,
            startup_report=startup_report,
            candidates=tuple(candidates),
            target_portfolio=target_portfolio,
            risk_decisions=tuple(decisions),
            sized_trades=tuple(sized_trades),
            submitted_order_ids=tuple(submitted_order_ids),
            skipped_trades=tuple(skip_reasons),
            messages=tuple(messages),
        )

    def _fetch_quotes(self, weight_trades: list[RequiredTrade]) -> dict[str, BrokerQuote]:
        instrument_ids = sorted(
            {trade.instrument_id for trade in weight_trades if trade.action is not TradeAction.HOLD}
        )
        if not instrument_ids:
            return {}
        try:
            return {quote.instrument_id: quote for quote in self.broker.get_quotes(instrument_ids)}
        except Exception as exc:  # noqa: BLE001 - sizing degrades to "no price", not a crash
            logger.warning("failed to fetch quotes for sizing: %s", exc)
            return {}

    def _size_trades(
        self,
        weight_trades: list[RequiredTrade],
        decisions_by_instrument: dict[str, RiskDecision],
        current_positions: list[Position],
        equity: float,
        quotes: dict[str, BrokerQuote],
        *,
        circuit_halted: bool,
    ) -> tuple[list[SizedTrade], list[str]]:
        # Quantities are sized off the last traded price; the *limit* price
        # each order is finally sent at is side-aware (see _limit_price).
        prices: dict[str, float] = {p.instrument_id: p.current_price for p in current_positions}
        prices.update({instrument_id: q.last_price for instrument_id, q in quotes.items()})
        current_quantities = {p.instrument_id: p.quantity for p in current_positions}
        return size_trades(
            weight_trades,
            decisions_by_instrument,
            prices,
            current_quantities,
            equity,
            circuit_halted=circuit_halted,
            min_order_value_inr=self.min_order_value_inr,
        )

    @staticmethod
    def _limit_price(trade: SizedTrade, quotes: dict[str, BrokerQuote]) -> float:
        """A marketable limit: buy at the ask, sell at the bid. A limit
        resting at the last traded price would frequently never fill,
        which for a daily-rebalanced system means silently drifting away
        from the risk-approved target portfolio. How far *through* the
        touch an order may be priced is not this layer's call -- the
        broker's own ``execution.order_price_guard_bps`` check rejects
        anything outside the configured band.
        """
        quote = quotes.get(trade.instrument_id)
        if quote is None:
            return trade.price
        touch = quote.ask if trade.side == "buy" else quote.bid
        return touch if touch > 0 else trade.price

    def _submit_orders(
        self,
        as_of: dt.date,
        sized_trades: list[SizedTrade],
        decisions_by_instrument: dict[str, RiskDecision],
        quotes: dict[str, BrokerQuote],
        messages: list[str],
    ) -> list[str]:
        submitted: list[str] = []
        order_type = self.execution_config.order_type
        for trade in sized_trades:
            idempotency_key = f"{as_of.isoformat()}:{trade.instrument_id}:{trade.action.value}"
            signal_id = f"{as_of.isoformat()}:{trade.instrument_id}"
            decision = decisions_by_instrument.get(trade.instrument_id)
            decision_tag = decision.circuit_state.value if decision else "exit"
            risk_decision_id = f"{idempotency_key}:{decision_tag}"
            limit_price = self._limit_price(trade, quotes) if order_type == "limit" else None
            try:
                result = self.order_manager.create(
                    trade.instrument_id,
                    trade.side,
                    trade.quantity,
                    order_type,
                    limit_price,
                    idempotency_key,
                    signal_id=signal_id,
                    risk_decision_id=risk_decision_id,
                )
            except DuplicateSignalError as exc:
                messages.append(f"{trade.instrument_id}: {exc}")
                continue
            if not result.was_duplicate or result.order.state is OrderState.CREATED:
                self.order_manager.submit(result.order.client_order_id, self.broker)
            submitted.append(result.order.client_order_id)
        return submitted

    # -- the ongoing loop: steps 15-20 -----------------------------------

    def _monitor_and_reconcile_once(self) -> None:
        # Steps 15-16: track fills, update portfolio.
        new_fills = self.fill_tracker.poll(self.broker)
        if new_fills:
            logger.info(
                "applied %d new fill(s) to the portfolio",
                len(new_fills),
                extra={"extra_fields": {"event": "orchestrator_fills_applied"}},
            )

        # Step 17: update stops/risk rules where applicable.
        self._update_stops_and_risk_rules()

        # Step 19: monitor health.
        broker_health = self.health_checker.check_broker()
        heartbeat_health = self.health_checker.check_heartbeat()
        self.heartbeat.beat()
        checks = (broker_health, heartbeat_health)
        unhealthy = any(check.status is ComponentHealth.UNHEALTHY for check in checks)
        degraded = any(check.status is ComponentHealth.DEGRADED for check in checks)

        # Step 20: reconcile periodically.
        position_report = self.reconciliation_engine.reconcile_positions()
        order_report = self.reconciliation_engine.reconcile_open_orders()
        quarantined = (
            position_report.status is ReconciliationStatus.QUARANTINED
            or order_report.status is ReconciliationStatus.QUARANTINED
        )

        circuit_halted = self.circuit_breaker.current_status().state is CircuitState.HALTED
        if circuit_halted or unhealthy:
            self.state = OrchestratorState.HALTED
        elif quarantined or degraded:
            self.state = OrchestratorState.DEGRADED
        elif self.state is OrchestratorState.DEGRADED:
            self.state = OrchestratorState.RUNNING

        # Step 18: persist state.
        self.startup_sequence.persist_heartbeat(
            SystemState.HALTED if self.state is OrchestratorState.HALTED else SystemState.READY
        )

    def _update_stops_and_risk_rules(self) -> None:
        broker_prices = {
            position.instrument_id: position.avg_price
            for position in self.broker.get_positions()
            if position.avg_price > 0
        }
        if broker_prices:
            self.position_tracker.mark_to_market(broker_prices, self._clock())

        positions = self.position_tracker.current_positions()
        if not positions:
            return
        equity = self.broker.get_account().equity
        current_portfolio = current_target_portfolio(
            positions, equity, self._clock().date(), AllocationRegime.UNCERTAIN
        )
        risk_state = build_risk_state(
            current_portfolio, equity, self._clock(), self.market_data, self.equity_history
        )
        self.circuit_breaker.evaluate(risk_state)
