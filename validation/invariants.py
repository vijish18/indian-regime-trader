"""The properties a paper session must never violate (Phase 21).

These are deliberately checkable *at any moment*, not only at the end: the
end-to-end validation runs them after every stage, so a violation is
attributed to the stage that caused it rather than discovered at teardown
with no way to tell what broke it.

Each check reads live state -- the position tracker, the broker, the order
manager, the circuit breaker, the reconciliation engine -- and never
mutates anything. A check that cannot be evaluated (because the state it
needs is unavailable) reports ``SKIPPED`` rather than passing, because a
check that quietly passes when it did not run is worse than no check.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from broker.base import Broker
from execution.order_manager import OrderManager
from execution.position_tracker import PositionTracker
from execution.reconciliation import ReconciliationEngine, ReconciliationStatus
from execution.system_state import SystemState, SystemStateStore, SystemStateStoreError
from risk.circuit_breaker import CircuitBreaker, CircuitState

_LEVERAGE_TOLERANCE = 1e-6
"""Gross exposure is compared against 1.0 with a small tolerance so
floating-point noise in a fully-invested book is not reported as
leverage."""


class Invariant(StrEnum):
    NO_DUPLICATE_POSITIONS = "no duplicate positions"
    NO_NEGATIVE_CASH = "no negative cash"
    NO_LEVERAGE = "no leverage"
    ALL_ORDERS_TRACEABLE = "all orders traceable"
    RECONCILIATION_SUCCEEDS = "reconciliation succeeds"
    HALTED_STATE_PERSISTS = "halted state persists"
    RESTART_IS_SAFE = "restart is safe"


class InvariantStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class InvariantResult:
    invariant: Invariant
    status: InvariantStatus
    detail: str

    @property
    def ok(self) -> bool:
        """``SKIPPED`` is not ``ok``: a validation run that could not
        evaluate a property has not established it."""
        return self.status is InvariantStatus.PASS


def check_invariants(
    *,
    position_tracker: PositionTracker,
    order_manager: OrderManager,
    broker: Broker,
    circuit_breaker: CircuitBreaker,
    reconciliation_engine: ReconciliationEngine,
    state_store: SystemStateStore,
) -> list[InvariantResult]:
    """Every point-in-time invariant, in a fixed order so two runs'
    reports line up row for row.

    ``Invariant.RESTART_IS_SAFE`` is not among them: unlike the other six,
    "is a restart safe" is not a property of a single moment -- it is a
    property of a *sequence* (crash, then a correct refusal to trade
    until reconciled, then recovery). ``validation.scenario`` proves it
    directly, by asserting exactly that sequence happened, rather than by
    asking this function to guess at it from one snapshot.
    """
    return [
        _no_duplicate_positions(position_tracker, broker),
        _no_negative_cash(broker),
        _no_leverage(position_tracker, broker),
        _all_orders_traceable(order_manager),
        _reconciliation_succeeds(reconciliation_engine),
        _halted_state_persists(circuit_breaker, state_store),
    ]


def _no_duplicate_positions(
    position_tracker: PositionTracker, broker: Broker
) -> InvariantResult:
    """Neither side may hold the same instrument twice. Locally this is
    structural (the tracker is keyed by instrument), so the check that
    earns its place is the broker's own list -- a duplicate there means
    two positions were opened for one instrument.
    """
    local = [position.instrument_id for position in position_tracker.current_positions()]
    try:
        remote = [position.instrument_id for position in broker.get_positions()]
    except Exception as exc:  # noqa: BLE001 - an unreachable broker cannot be checked
        return InvariantResult(
            Invariant.NO_DUPLICATE_POSITIONS,
            InvariantStatus.SKIPPED,
            f"broker positions unavailable: {exc}",
        )

    local_dupes = _duplicates(local)
    remote_dupes = _duplicates(remote)
    if local_dupes or remote_dupes:
        return InvariantResult(
            Invariant.NO_DUPLICATE_POSITIONS,
            InvariantStatus.FAIL,
            f"duplicate instrument ids -- local: {local_dupes}, broker: {remote_dupes}",
        )
    return InvariantResult(
        Invariant.NO_DUPLICATE_POSITIONS,
        InvariantStatus.PASS,
        f"{len(local)} local and {len(remote)} broker position(s), all distinct",
    )


def _no_negative_cash(broker: Broker) -> InvariantResult:
    try:
        account = broker.get_account()
    except Exception as exc:  # noqa: BLE001
        return InvariantResult(
            Invariant.NO_NEGATIVE_CASH,
            InvariantStatus.SKIPPED,
            f"broker account unavailable: {exc}",
        )
    if account.cash < 0:
        return InvariantResult(
            Invariant.NO_NEGATIVE_CASH,
            InvariantStatus.FAIL,
            f"cash is {account.cash:,.2f}; this system never borrows",
        )
    return InvariantResult(
        Invariant.NO_NEGATIVE_CASH, InvariantStatus.PASS, f"cash {account.cash:,.2f}"
    )


def _no_leverage(position_tracker: PositionTracker, broker: Broker) -> InvariantResult:
    """Two separate ways leverage could appear: holdings worth more than
    equity, or a short position. V1 is long-only and unlevered, so both
    are failures.
    """
    positions = position_tracker.current_positions()
    shorts = [p.instrument_id for p in positions if p.quantity < 0]
    if shorts:
        return InvariantResult(
            Invariant.NO_LEVERAGE,
            InvariantStatus.FAIL,
            f"short position(s) in {shorts}; this system is long-only",
        )
    try:
        equity = broker.get_account().equity
    except Exception as exc:  # noqa: BLE001
        return InvariantResult(
            Invariant.NO_LEVERAGE, InvariantStatus.SKIPPED, f"equity unavailable: {exc}"
        )
    if equity <= 0:
        return InvariantResult(
            Invariant.NO_LEVERAGE, InvariantStatus.FAIL, f"equity is {equity:,.2f}"
        )
    holdings_value = sum(p.quantity * p.current_price for p in positions)
    gross_exposure = holdings_value / equity
    if gross_exposure > 1.0 + _LEVERAGE_TOLERANCE:
        return InvariantResult(
            Invariant.NO_LEVERAGE,
            InvariantStatus.FAIL,
            f"gross exposure {gross_exposure:.4%} exceeds 100% of equity",
        )
    return InvariantResult(
        Invariant.NO_LEVERAGE,
        InvariantStatus.PASS,
        f"gross exposure {gross_exposure:.2%}, no short positions",
    )


def _all_orders_traceable(order_manager: OrderManager) -> InvariantResult:
    """Every order must carry its signal and risk-decision lineage and be
    reconstructable from the journal. ``OrderManager.create`` enforces the
    first structurally; this verifies the chain actually resolves, which
    is the claim the audit trail makes.
    """
    orders = order_manager.all_orders()
    if not orders:
        return InvariantResult(
            Invariant.ALL_ORDERS_TRACEABLE, InvariantStatus.PASS, "no orders to trace"
        )
    untraceable: list[str] = []
    for order in orders:
        if not order.signal_id or not order.risk_decision_id:
            untraceable.append(f"{order.client_order_id} (missing lineage ids)")
            continue
        try:
            trace = order_manager.journal.trace(order.client_order_id)
        except Exception as exc:  # noqa: BLE001
            untraceable.append(f"{order.client_order_id} (journal: {exc})")
            continue
        if trace.signal_id != order.signal_id:
            untraceable.append(
                f"{order.client_order_id} (journal signal {trace.signal_id!r} != "
                f"order signal {order.signal_id!r})"
            )
    if untraceable:
        return InvariantResult(
            Invariant.ALL_ORDERS_TRACEABLE,
            InvariantStatus.FAIL,
            f"{len(untraceable)} of {len(orders)} order(s) not traceable: {untraceable}",
        )
    return InvariantResult(
        Invariant.ALL_ORDERS_TRACEABLE,
        InvariantStatus.PASS,
        f"all {len(orders)} order(s) trace to a signal and a risk decision",
    )


def _reconciliation_succeeds(engine: ReconciliationEngine) -> InvariantResult:
    try:
        positions = engine.reconcile_positions()
        orders = engine.reconcile_open_orders()
    except Exception as exc:  # noqa: BLE001
        return InvariantResult(
            Invariant.RECONCILIATION_SUCCEEDS,
            InvariantStatus.SKIPPED,
            f"reconciliation could not run: {exc}",
        )
    if (
        positions.status is ReconciliationStatus.CLEAN
        and orders.status is ReconciliationStatus.CLEAN
    ):
        return InvariantResult(
            Invariant.RECONCILIATION_SUCCEEDS,
            InvariantStatus.PASS,
            "local and broker state agree on positions and open orders",
        )
    details = [m.detail for m in (*positions.mismatches, *orders.mismatches)]
    return InvariantResult(
        Invariant.RECONCILIATION_SUCCEEDS,
        InvariantStatus.FAIL,
        f"{len(details)} mismatch(es): {details}",
    )


def _halted_state_persists(
    circuit_breaker: CircuitBreaker, state_store: SystemStateStore
) -> InvariantResult:
    """A halt must survive on disk. This is only meaningful once something
    has halted, so it reports ``PASS`` with an explicit note when nothing
    has -- there is no halt to lose.
    """
    try:
        status = circuit_breaker.current_status()
    except Exception as exc:  # noqa: BLE001
        return InvariantResult(
            Invariant.HALTED_STATE_PERSISTS,
            InvariantStatus.SKIPPED,
            f"circuit-breaker state unreadable: {exc}",
        )
    if status.state is not CircuitState.HALTED:
        return InvariantResult(
            Invariant.HALTED_STATE_PERSISTS,
            InvariantStatus.PASS,
            f"nothing halted (circuit breaker is {status.state.value})",
        )

    # The breaker itself re-reads its own file on every call, so a HALTED
    # status here is already proof it persisted. What remains to check is
    # that the system state written alongside it agrees -- a halt the
    # breaker knows about but the persisted system state does not would
    # come back from a restart as READY.
    try:
        persisted = state_store.load()
    except SystemStateStoreError as exc:
        return InvariantResult(
            Invariant.HALTED_STATE_PERSISTS,
            InvariantStatus.FAIL,
            f"circuit breaker is HALTED but persisted state is unreadable: {exc}",
        )
    if persisted is None:
        return InvariantResult(
            Invariant.HALTED_STATE_PERSISTS,
            InvariantStatus.FAIL,
            "circuit breaker is HALTED but nothing was persisted",
        )
    if persisted.system_state is not SystemState.HALTED:
        return InvariantResult(
            Invariant.HALTED_STATE_PERSISTS,
            InvariantStatus.FAIL,
            f"circuit breaker is HALTED but persisted system state is "
            f"{persisted.system_state.value}",
        )
    return InvariantResult(
        Invariant.HALTED_STATE_PERSISTS,
        InvariantStatus.PASS,
        f"halt persisted (reason: {status.reason})",
    )


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicated: list[str] = []
    for value in values:
        if value in seen and value not in duplicated:
            duplicated.append(value)
        seen.add(value)
    return duplicated
