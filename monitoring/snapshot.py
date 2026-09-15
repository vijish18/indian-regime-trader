"""One point-in-time picture of the whole running system (Phase 20).

Both consumers in this package read from here and nowhere else: the
terminal dashboard renders a ``MonitoringSnapshot``, and the alert rules
evaluate one. That split is deliberate -- gathering the numbers is done
once, by ``SnapshotCollector``; the dashboard then purely formats, and the
rules purely compare. Neither of them queries a broker, a tracker or a
calendar of its own, so what an operator sees on screen and what triggers
an alert can never be two different readings of the same moment.

``SnapshotCollector`` computes nothing a dedicated module already owns: it
reads ``HealthChecker`` for feed/broker health, ``PositionTracker`` for
holdings, ``CircuitBreaker`` for the risk mode, ``ReconciliationEngine``
for mismatches, and so on. The arithmetic it does do is the arithmetic of
*presentation* -- turning a held quantity and a price into an exposure
percentage -- not of strategy.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from broker.base import Broker
from core.regime.allocation import AllocationTarget
from core.regime.hmm_engine import RegimeState
from core.regime.model_registry import ModelRegistry
from data.errors import CalendarCoverageError
from data.interfaces import MarketDataProvider, TradingCalendar
from data.models import SessionType
from execution.order_manager import TERMINAL_STATES, OrderManager, OrderState
from execution.position_tracker import Position, PositionTracker
from execution.reconciliation import ReconciliationEngine, ReconciliationMismatch
from monitoring.health import ComponentHealth, HealthChecker
from risk.circuit_breaker import CircuitBreaker, CircuitState
from risk.risk_state_builder import EquityHistory


@dataclass(frozen=True)
class SystemPanel:
    status: str
    """The application lifecycle state, as a plain string. Deliberately not
    ``orchestration.orchestrator_state.OrchestratorState``: ``monitoring``
    must not import ``orchestration`` (see docs/ARCHITECTURE.md's
    dependency rules), so the caller supplies its own state as text."""

    uptime: dt.timedelta
    session_date: dt.date
    session_type: SessionType
    session_detail: str
    data_feed: ComponentHealth
    data_feed_detail: str
    broker_connected: bool
    broker_detail: str
    model_version: str | None


@dataclass(frozen=True)
class PortfolioPanel:
    equity: float
    cash: float
    gross_exposure_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    """Signed, unlike ``EquityHistory.daily_pnl_pct``: a dashboard shows a
    gain as a gain, while the risk engine only ever cares about losses."""

    drawdown_pct: float
    position_count: int
    cumulative_fill_cash_flow: float
    """Net cash the fills this system has observed should have moved, sells
    positive and buys negative, gross of costs. Only meaningful as a
    difference between two snapshots -- see
    ``monitoring.alert_rules`` on the unexpected-cash check."""


@dataclass(frozen=True)
class RegimePanel:
    regime: str | None
    """The allocation tier actually driving exposure
    (``core.regime.allocation.AllocationRegime``)."""

    label: str | None
    """The HMM's descriptive label (``core.regime.hmm_engine.RegimeLabel``).
    Shown beside the tier, never instead of it: labels are assigned by
    ranking measured statistics and are reporting-only, never a decision
    input -- so a dashboard that showed only the label would imply the
    system acts on something it does not."""

    probability: float | None
    confidence: float | None
    persistence: float | None
    india_vix: float | None
    india_vix_change: float | None
    nifty_close: float | None
    nifty_change_pct: float | None
    as_of: dt.date | None


@dataclass(frozen=True)
class ExecutionPanel:
    orders_submitted: int
    fills: int
    rejected_orders: int
    open_orders: int
    orders_in_unknown_state: int
    reconciliation_mismatches: tuple[ReconciliationMismatch, ...]

    @property
    def pending_reconciliation(self) -> int:
        """Everything still waiting on reconciliation: orders whose true
        state is unknown, plus instruments whose quantity local state and
        the broker disagree about."""
        return self.orders_in_unknown_state + len(self.reconciliation_mismatches)


@dataclass(frozen=True)
class RiskPanel:
    risk_mode: CircuitState
    circuit_reason: str | None
    circuit_triggered_by: str | None
    requires_manual_recovery: bool
    max_concentration_pct: float
    concentration_instrument: str | None
    daily_turnover_pct: float


@dataclass(frozen=True)
class MonitoringSnapshot:
    as_of: dt.datetime
    system: SystemPanel
    portfolio: PortfolioPanel
    regime: RegimePanel
    execution: ExecutionPanel
    risk: RiskPanel


class SnapshotCollector:
    def __init__(
        self,
        *,
        broker: Broker,
        position_tracker: PositionTracker,
        order_manager: OrderManager,
        circuit_breaker: CircuitBreaker,
        reconciliation_engine: ReconciliationEngine,
        health_checker: HealthChecker,
        calendar: TradingCalendar,
        market_data: MarketDataProvider,
        model_registry: ModelRegistry,
        equity_history: EquityHistory,
        index_symbol: str,
        vix_symbol: str,
        state_supplier: Callable[[], str],
        regime_supplier: Callable[[], RegimeState | None],
        allocation_supplier: Callable[[], AllocationTarget | None] | None = None,
        cash_flow_supplier: Callable[[], float] | None = None,
        started_at: dt.datetime | None = None,
        clock: Callable[[], dt.datetime] | None = None,
    ) -> None:
        self.broker = broker
        self.position_tracker = position_tracker
        self.order_manager = order_manager
        self.circuit_breaker = circuit_breaker
        self.reconciliation_engine = reconciliation_engine
        self.health_checker = health_checker
        self.calendar = calendar
        self.market_data = market_data
        self.model_registry = model_registry
        self.equity_history = equity_history
        self.index_symbol = index_symbol
        self.vix_symbol = vix_symbol
        self._clock: Callable[[], dt.datetime] = clock or (lambda: dt.datetime.now(dt.UTC))
        self.started_at = started_at or self._clock()
        # Suppliers for the things only the running application knows.
        self._state_supplier = state_supplier
        self._regime_supplier = regime_supplier
        self._allocation_supplier: Callable[[], AllocationTarget | None] = (
            allocation_supplier or (lambda: None)
        )
        self._cash_flow_supplier: Callable[[], float] = cash_flow_supplier or (lambda: 0.0)

    def collect(self) -> MonitoringSnapshot:
        now = self._clock()
        as_of_date = now.date()
        return MonitoringSnapshot(
            as_of=now,
            system=self._system(now, as_of_date),
            portfolio=self._portfolio(),
            regime=self._regime(as_of_date),
            execution=self._execution(),
            risk=self._risk(),
        )

    # -- sections -------------------------------------------------------

    def _system(self, now: dt.datetime, as_of: dt.date) -> SystemPanel:
        try:
            session = self.calendar.session(as_of)
            session_type = session.session_type
            session_detail = (
                f"{session.regular_open:%H:%M}-{session.regular_close:%H:%M}"
                if session_type is not SessionType.CLOSED
                else "market closed"
            )
        except CalendarCoverageError as exc:
            session_type = SessionType.CLOSED
            session_detail = f"calendar does not cover {as_of}: {exc}"

        feed = self.health_checker.check_market_data_freshness(as_of)
        broker_health = self.health_checker.check_broker()
        return SystemPanel(
            status=self._state_supplier(),
            uptime=now - self.started_at,
            session_date=as_of,
            session_type=session_type,
            session_detail=session_detail,
            data_feed=feed.status,
            data_feed_detail=feed.detail,
            broker_connected=broker_health.status is ComponentHealth.HEALTHY,
            broker_detail=broker_health.detail,
            model_version=self.model_registry.approved_model_id(),
        )

    def _portfolio(self) -> PortfolioPanel:
        account = self.broker.get_account()
        positions = self.position_tracker.current_positions()
        holdings_value = sum(p.quantity * p.current_price for p in positions)
        equity = account.equity
        previous_equity = (
            self.equity_history.values[-2] if len(self.equity_history.values) >= 2 else None
        )
        daily_pnl = equity - previous_equity if previous_equity is not None else 0.0
        daily_pnl_pct = (
            daily_pnl / previous_equity if previous_equity else 0.0
        )
        return PortfolioPanel(
            equity=equity,
            cash=account.cash,
            gross_exposure_pct=holdings_value / equity if equity > 0 else 0.0,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            drawdown_pct=self.equity_history.peak_to_trough_drawdown_pct(),
            position_count=len(positions),
            cumulative_fill_cash_flow=self._cash_flow_supplier(),
        )

    def _regime(self, as_of: dt.date) -> RegimePanel:
        state = self._regime_supplier()
        allocation = self._allocation_supplier()
        vix_level, vix_change = self._index_level_and_change(self.vix_symbol, as_of)
        nifty_level, nifty_change = self._index_level_and_change(self.index_symbol, as_of)
        return RegimePanel(
            regime=allocation.regime.value if allocation else None,
            label=state.label.value if state else None,
            probability=(max(state.probabilities) if state and state.probabilities else None),
            confidence=state.confidence if state else None,
            persistence=state.persistence if state else None,
            india_vix=vix_level,
            india_vix_change=vix_change,
            nifty_close=nifty_level,
            nifty_change_pct=(
                nifty_change / (nifty_level - nifty_change)
                if nifty_level is not None
                and nifty_change is not None
                and (nifty_level - nifty_change) != 0
                else None
            ),
            as_of=state.as_of if state else None,
        )

    def _index_level_and_change(
        self, symbol: str, as_of: dt.date
    ) -> tuple[float | None, float | None]:
        """The latest close for ``symbol`` and its change from the previous
        observation. Returns ``(None, None)`` when the series is absent --
        a missing index feed is reported by the data-feed health check, not
        by crashing the dashboard.
        """
        observations = self.market_data.get_index_observations(
            symbol, as_of - dt.timedelta(days=30), as_of
        )
        if not observations:
            return None, None
        latest = float(observations[-1].close)
        if len(observations) < 2:
            return latest, None
        return latest, latest - float(observations[-2].close)

    def _execution(self) -> ExecutionPanel:
        orders = self.order_manager.all_orders()
        submitted = [order for order in orders if order.state is not OrderState.CREATED]
        fills = [order for order in orders if order.filled_quantity > 0]
        rejected = [order for order in orders if order.state is OrderState.REJECTED]
        unknown = [order for order in orders if order.state is OrderState.UNKNOWN]
        open_orders = [
            order
            for order in orders
            if order.state not in TERMINAL_STATES and order.state is not OrderState.CREATED
        ]
        position_report = self.reconciliation_engine.reconcile_positions()
        return ExecutionPanel(
            orders_submitted=len(submitted),
            fills=len(fills),
            rejected_orders=len(rejected),
            open_orders=len(open_orders),
            orders_in_unknown_state=len(unknown),
            reconciliation_mismatches=position_report.mismatches,
        )

    def _risk(self) -> RiskPanel:
        status = self.circuit_breaker.current_status()
        positions = self.position_tracker.current_positions()
        equity = self.broker.get_account().equity
        concentration, instrument = _largest_weight(positions, equity)
        return RiskPanel(
            risk_mode=status.state,
            circuit_reason=status.reason,
            circuit_triggered_by=status.triggered_by,
            requires_manual_recovery=status.requires_manual_recovery,
            max_concentration_pct=concentration,
            concentration_instrument=instrument,
            daily_turnover_pct=self._daily_turnover_pct(equity),
        )

    def _daily_turnover_pct(self, equity: float) -> float:
        """Traded value as a fraction of equity, from the orders this
        system actually created today -- not a re-derivation of the risk
        engine's own turnover limit, just what the order book shows.
        """
        if equity <= 0:
            return 0.0
        today = self._clock().date()
        traded = 0.0
        for order in self.order_manager.all_orders():
            if order.created_at.date() != today:
                continue
            price = order.avg_fill_price or order.limit_price
            if price is None:
                continue
            quantity = order.filled_quantity or order.quantity
            traded += quantity * price
        return traded / equity


def _largest_weight(positions: Sequence[Position], equity: float) -> tuple[float, str | None]:
    """The single most concentrated holding, as a fraction of equity."""
    if equity <= 0:
        return 0.0, None
    largest = 0.0
    instrument: str | None = None
    for position in positions:
        weight = (position.quantity * position.current_price) / equity
        if weight > largest:
            largest = weight
            instrument = position.instrument_id
    return largest, instrument
