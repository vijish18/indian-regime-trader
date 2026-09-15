"""Restart recovery and broker reconciliation (Phase 18): the startup
sequence that must run, in order, before this system is ever allowed to
permit strategy execution after a (re)start.

    1. load configuration
    2. verify application version / persisted-state schema compatibility
    3. verify database (the state store -- see execution.system_state)
    4. verify broker connectivity
    5. retrieve broker positions
    6. retrieve open orders
    7. retrieve recent fills
    8. compare broker state with local state
    9. resolve discrepancies (order-level ambiguity only -- see below)
    10. rebuild portfolio state (only once reconciliation is clean)
    11. verify risk state (an approved regime model exists, if one is required)
    12. verify circuit-breaker state
    13. only then permit strategy execution

**"Resolve discrepancies" (step 9) does not mean "make them go away."**
Order-level ambiguity (an ``UNKNOWN`` order) genuinely can be resolved
safely -- ``execution.order_reconciler.OrderReconciler`` already does
this by querying the broker, which is always authoritative for its own
order state. A *position* discrepancy (the broker reports a different
quantity than local state believes) cannot be resolved the same way:
there is no query that tells this system *why* the numbers differ, only
*that* they do. Per this phase's explicit instruction, a position
discrepancy is never auto-resolved -- the system moves to
``SystemState.RECONCILIATION_REQUIRED`` and refuses to permit strategy
execution until :meth:`StartupSequence.acknowledge_and_recover` is
called explicitly, by name, with an operator and a reason -- the same
"explicit, separately logged, never automatic" pattern
``risk.circuit_breaker.CircuitBreaker.manual_reset`` already established
for exactly this kind of decision.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

from broker.base import Broker
from config.loader import ConfigError, load_settings
from config.models import Settings
from core.regime.model_registry import ModelRegistry
from execution.order_manager import OrderManager
from execution.order_reconciler import OrderReconciler
from execution.position_tracker import PositionTracker
from execution.reconciliation import (
    ReconciliationEngine,
    ReconciliationReport,
    ReconciliationStatus,
)
from execution.system_state import (
    APP_VERSION,
    STATE_SCHEMA_VERSION,
    PersistedState,
    PortfolioPositionSnapshot,
    PortfolioSnapshot,
    SystemState,
    SystemStateStore,
    SystemStateStoreError,
)
from risk.circuit_breaker import CircuitBreaker, CircuitState

logger = logging.getLogger(__name__)


class StartupError(RuntimeError):
    """A startup step failed in a way that blocks the whole sequence --
    configuration invalid, persisted state schema incompatible, the
    state store inaccessible. Distinct from a reconciliation discrepancy,
    which never raises: it produces a :class:`StartupReport` with
    ``system_state=RECONCILIATION_REQUIRED`` so a caller can inspect and
    act on it, rather than only catching an exception.
    """


@dataclass(frozen=True, slots=True)
class StartupReport:
    as_of: dt.datetime
    system_state: SystemState
    permit_strategy_execution: bool
    app_version: str
    previous_app_version: str | None
    schema_version_ok: bool
    database_verified: bool
    broker_connected: bool
    broker_health_detail: str
    position_reconciliation: ReconciliationReport | None
    order_reconciliation: ReconciliationReport | None
    model_version: str | None
    risk_state_verified: bool
    circuit_breaker_state: CircuitState | None
    messages: tuple[str, ...]

    def is_clean(self) -> bool:
        return self.system_state is SystemState.READY


def _empty_reconciliation(now: dt.datetime) -> ReconciliationReport:
    return ReconciliationReport(as_of=now, status=ReconciliationStatus.CLEAN, mismatches=())


class StartupSequence:
    """Runs the 13-step restart-recovery sequence once, via :meth:`run`.
    Construct one per (re)start -- it holds no state of its own beyond
    its dependencies; every fact about the current attempt lives in the
    :class:`StartupReport` it returns.
    """

    def __init__(
        self,
        state_store: SystemStateStore,
        position_tracker: PositionTracker,
        order_manager: OrderManager,
        broker: Broker,
        circuit_breaker: CircuitBreaker,
        model_registry: ModelRegistry | None = None,
        strategy_version: str = "unknown",
        order_reconciler: OrderReconciler | None = None,
        clock: Callable[[], dt.datetime] | None = None,
        settings_loader: Callable[[], Settings] = load_settings,
    ) -> None:
        self.state_store = state_store
        self.position_tracker = position_tracker
        self.order_manager = order_manager
        self.broker = broker
        self.circuit_breaker = circuit_breaker
        self.model_registry = model_registry
        self.strategy_version = strategy_version
        self.order_reconciler = order_reconciler or OrderReconciler(clock=clock)
        self._clock: Callable[[], dt.datetime] = clock or (lambda: dt.datetime.now(dt.UTC))
        self._settings_loader = settings_loader

    def run(self) -> StartupReport:
        now = self._clock()
        messages: list[str] = []

        # Step 1: load configuration.
        try:
            self._settings_loader()
        except ConfigError as exc:
            raise StartupError(f"configuration failed to load: {exc}") from exc
        messages.append("configuration loaded")

        # Step 2: verify application version / persisted-state schema.
        try:
            previous = self.state_store.load()
        except SystemStateStoreError as exc:
            raise StartupError(f"state store verification failed: {exc}") from exc
        if previous is not None and previous.schema_version != STATE_SCHEMA_VERSION:
            raise StartupError(
                f"persisted state schema_version={previous.schema_version} does not match "
                f"this code's STATE_SCHEMA_VERSION={STATE_SCHEMA_VERSION} -- refusing to "
                "interpret possibly-incompatible persisted data"
            )
        previous_app_version = previous.app_version if previous is not None else None
        if previous_app_version is not None and previous_app_version != APP_VERSION:
            messages.append(
                f"application version changed since last run: "
                f"{previous_app_version} -> {APP_VERSION}"
            )
        messages.append(f"application version verified: {APP_VERSION}")

        # Step 3: verify database (the state store).
        try:
            self.state_store.verify_accessible()
        except SystemStateStoreError as exc:
            raise StartupError(f"state store verification failed: {exc}") from exc
        messages.append("state store verified")

        # Step 4: verify broker connectivity.
        health = self.broker.health_check()
        if not health.healthy:
            messages.append(f"broker connectivity check failed: {health.detail}")
            return self._finish(
                now,
                messages,
                system_state=SystemState.RECONCILIATION_REQUIRED,
                broker_connected=False,
                broker_health_detail=health.detail,
                position_report=None,
                order_report=None,
                model_version=None,
                risk_state_verified=False,
                previous=previous,
            )
        messages.append("broker connectivity verified")

        # Steps 5-7: retrieve broker positions, open orders, and recent
        # fills. Fills are deduplicated by trade_id -- a redelivered
        # broker event must never be processed twice.
        try:
            self.broker.get_positions()
            self.broker.get_open_orders()
            fills = self.broker.get_trades()
        except Exception as exc:
            messages.append(f"failed to retrieve broker state: {exc}")
            return self._finish(
                now,
                messages,
                system_state=SystemState.RECONCILIATION_REQUIRED,
                broker_connected=True,
                broker_health_detail=health.detail,
                position_report=None,
                order_report=None,
                model_version=None,
                risk_state_verified=False,
                previous=previous,
            )
        deduplicated_fills = {fill.trade_id: fill for fill in fills}
        if len(deduplicated_fills) != len(fills):
            messages.append(
                f"duplicate broker fill events detected and collapsed: "
                f"{len(fills)} received, {len(deduplicated_fills)} distinct trade_ids"
            )
        last_fill = max(deduplicated_fills.values(), key=lambda f: f.as_of, default=None)

        # Step 8: compare broker state with local state.
        reconciler = ReconciliationEngine(
            self.position_tracker,
            self.order_manager,
            self.broker,
            order_reconciler=self.order_reconciler,
            clock=self._clock,
        )
        position_report = reconciler.reconcile_positions()

        # Step 9: resolve discrepancies -- order-level ambiguity only;
        # see this module's own docstring for why a position mismatch is
        # never resolved here.
        order_report = reconciler.reconcile_open_orders()

        discrepancy_found = (
            position_report.status is ReconciliationStatus.QUARANTINED
            or order_report.status is ReconciliationStatus.QUARANTINED
        )
        if discrepancy_found:
            messages.append(
                f"reconciliation found {len(position_report.mismatches)} position "
                f"mismatch(es) and {len(order_report.mismatches)} unrecognized broker "
                "order(s) -- refusing to permit strategy execution"
            )
            return self._finish(
                now,
                messages,
                system_state=SystemState.RECONCILIATION_REQUIRED,
                broker_connected=True,
                broker_health_detail=health.detail,
                position_report=position_report,
                order_report=order_report,
                model_version=None,
                risk_state_verified=False,
                previous=previous,
            )
        messages.append("broker and local state reconcile cleanly")

        # Step 10: rebuild portfolio state -- only reached once clean.
        self._mark_to_market_from_broker()
        messages.append("portfolio state rebuilt from broker positions")

        # Step 11: verify risk state (an approved regime model exists,
        # if this deployment requires one).
        model_version: str | None = None
        risk_state_verified = True
        if self.model_registry is not None:
            model_version = self.model_registry.approved_model_id()
            if model_version is None:
                risk_state_verified = False
                messages.append(
                    "no approved regime model -- refusing to permit strategy execution"
                )

        # Step 12: verify circuit-breaker state.
        circuit_status = self.circuit_breaker.current_status()
        circuit_ok = circuit_status.state is not CircuitState.HALTED
        if not circuit_ok:
            messages.append(f"circuit breaker is HALTED: {circuit_status.reason}")

        # Step 13: only then permit strategy execution.
        permit = risk_state_verified and circuit_ok
        if permit:
            final_state = SystemState.READY
        elif not circuit_ok:
            final_state = SystemState.HALTED
        else:
            final_state = SystemState.RECONCILIATION_REQUIRED

        report = self._finish(
            now,
            messages,
            system_state=final_state,
            broker_connected=True,
            broker_health_detail=health.detail,
            position_report=position_report,
            order_report=order_report,
            model_version=model_version,
            risk_state_verified=risk_state_verified,
            previous=previous,
        )

        self._persist(
            now,
            system_state=final_state,
            model_version=model_version,
            last_broker_event_id=last_fill.trade_id if last_fill else None,
            last_broker_event_timestamp=last_fill.as_of if last_fill else None,
            carry_forward=previous,
        )
        return report

    def acknowledge_and_recover(self, operator: str, reason: str) -> StartupReport:
        """The safe manual recovery mechanism: an explicit, logged,
        human-invoked re-run of the sequence. Never automatic -- there is
        no code path anywhere in this module that calls this on a
        caller's behalf. If the broker now reconciles cleanly, this
        moves the system out of ``RECONCILIATION_REQUIRED``; if it still
        does not, the system stays in ``RECONCILIATION_REQUIRED`` and the
        report says why.
        """
        if not operator or not reason:
            raise StartupError("acknowledge_and_recover requires a non-empty operator and reason")
        logger.critical(
            "manual recovery acknowledged: re-running startup sequence "
            "(operator=%s, reason=%s)",
            operator,
            reason,
            extra={
                "extra_fields": {
                    "event": "startup_manual_recovery",
                    "operator": operator,
                    "reason": reason,
                }
            },
        )
        return self.run()

    # -- internals -----------------------------------------------------

    def _mark_to_market_from_broker(self) -> None:
        prices = {
            position.instrument_id: position.avg_price
            for position in self.broker.get_positions()
            if position.avg_price > 0
        }
        if prices:
            self.position_tracker.mark_to_market(prices, self._clock())

    def _finish(
        self,
        now: dt.datetime,
        messages: list[str],
        *,
        system_state: SystemState,
        broker_connected: bool,
        broker_health_detail: str,
        position_report: ReconciliationReport | None,
        order_report: ReconciliationReport | None,
        model_version: str | None,
        risk_state_verified: bool,
        previous: PersistedState | None,
    ) -> StartupReport:
        circuit_state: CircuitState | None = None
        try:
            circuit_state = self.circuit_breaker.current_status().state
        except Exception:  # noqa: BLE001 - reporting is best-effort here
            pass
        return StartupReport(
            as_of=now,
            system_state=system_state,
            permit_strategy_execution=system_state is SystemState.READY,
            app_version=APP_VERSION,
            previous_app_version=previous.app_version if previous else None,
            schema_version_ok=True,
            database_verified=True,
            broker_connected=broker_connected,
            broker_health_detail=broker_health_detail,
            position_reconciliation=position_report,
            order_reconciliation=order_report,
            model_version=model_version,
            risk_state_verified=risk_state_verified,
            circuit_breaker_state=circuit_state,
            messages=tuple(messages),
        )

    def _persist(
        self,
        now: dt.datetime,
        *,
        system_state: SystemState,
        model_version: str | None,
        last_broker_event_id: str | None,
        last_broker_event_timestamp: dt.datetime | None,
        carry_forward: PersistedState | None,
    ) -> None:
        positions = tuple(
            PortfolioPositionSnapshot(
                instrument_id=p.instrument_id, quantity=p.quantity, avg_price=p.avg_price
            )
            for p in self.position_tracker.current_positions()
        )
        snapshot = PortfolioSnapshot(
            as_of=now,
            cash=self.broker.get_account().cash,
            positions=positions,
        )
        state = PersistedState(
            schema_version=STATE_SCHEMA_VERSION,
            app_version=APP_VERSION,
            system_state=system_state,
            model_version=model_version,
            strategy_version=self.strategy_version,
            portfolio_snapshot=snapshot,
            last_market_data_timestamp=(
                carry_forward.last_market_data_timestamp if carry_forward else None
            ),
            last_broker_event_id=(
                last_broker_event_id
                or (carry_forward.last_broker_event_id if carry_forward else None)
            ),
            last_broker_event_timestamp=(
                last_broker_event_timestamp
                or (carry_forward.last_broker_event_timestamp if carry_forward else None)
            ),
            updated_at=now,
        )
        self.state_store.save(state)

    def persist_heartbeat(self, system_state: SystemState) -> None:
        """Persists the current picture -- state, model/strategy version,
        portfolio snapshot, last processed timestamps -- without
        re-running verification, for a caller (``orchestration.orchestrator.Orchestrator``'s
        own ongoing monitoring loop) that already knows it is past startup
        and just wants Phase 19 step 18 ("persist state") done repeatedly
        with whatever ``system_state`` it has independently determined.
        Carries forward everything else from the last persisted state.
        """
        previous = self.state_store.load()
        self._persist(
            self._clock(),
            system_state=system_state,
            model_version=previous.model_version if previous else None,
            last_broker_event_id=None,
            last_broker_event_timestamp=None,
            carry_forward=previous,
        )

    def checkpoint_market_data(self, timestamp: dt.datetime) -> None:
        """Updates only the persisted "last processed market-data
        timestamp", preserving every other field -- called by a live
        trading loop as it processes data, independent of the startup
        sequence itself, using the same store.
        """
        current = self.state_store.load()
        if current is None:
            raise StartupError(
                "cannot checkpoint market data before the startup sequence has run once"
            )
        self.state_store.save(replace(current, last_market_data_timestamp=timestamp))
