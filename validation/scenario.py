"""The end-to-end paper-trading session (Phase 21).

One coherent narrative, not twenty independent snippets: ingest a
synthetic vendor drop, run a real trading day through
``orchestration.orchestrator.Orchestrator``, watch it monitor itself,
shut it down, crash it, restart it, and reconcile -- with each of the
eight required failure-injection scenarios triggered at the point in that
narrative where the real failure would actually occur, and
``validation.invariants.check_invariants`` run after every stage so a
violation is attributed to the stage that caused it.

Every stage function takes the shared :class:`~validation.harness.ValidationEnvironment`
and returns a :class:`StageResult`; nothing here asserts (a stage that
fails is recorded, not raised) so one bad stage never prevents the report
from describing every other one.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from data.errors import DataNotAvailableError
from execution.order_manager import OrderState
from execution.system_state import SystemState
from monitoring.terminal_dashboard import render_dashboard
from orchestration.orchestrator_state import OrchestratorState
from validation.harness import ValidationEnvironment
from validation.invariants import InvariantResult, InvariantStatus, check_invariants

_OPERATOR = "validation-harness"


class StageStatus:
    PASS = "pass"
    FAIL = "fail"


@dataclass
class StageResult:
    name: str
    category: str
    """``"stage"`` for the required narrative, ``"failure_injection"`` for
    one of the eight required fault scenarios."""

    status: str
    detail: str
    invariants: tuple[InvariantResult, ...] = ()
    """A snapshot of every invariant *immediately after this stage*, for
    audit -- not itself the pass/fail signal (see ``ok``). Several stages
    deliberately, correctly put the system into a state where one of
    these reads FAIL for a while: a crash must leave reconciliation
    failing until the recovery stage explicitly resolves it, and that is
    the crash-handling working, not a defect. Each stage's own ``status``
    already encodes the specific, context-aware assertion that stage
    exists to make (see the stage functions themselves); the *session's*
    overall health is judged once, at the very end, by
    ``SessionReport.final_invariants`` -- after every recovery the
    narrative performs has had the chance to run.
    """

    @property
    def ok(self) -> bool:
        return self.status == StageStatus.PASS


@dataclass
class SessionReport:
    started_at: dt.datetime
    """Real wall-clock time the run started -- deliberately *not*
    ``env.clock``, which is a simulated market clock the narrative moves
    backwards and forwards across trading dates as each stage needs (a
    failure-injection stage tests an earlier live date than the daily
    cycle that ran before it, for instance), so treating it as a
    monotonic session timer would produce a nonsensical duration."""

    finished_at: dt.datetime
    market_sessions: int
    live_session_dates: tuple[dt.date, ...]
    stages: list[StageResult] = field(default_factory=list)
    final_invariants: tuple[InvariantResult, ...] = ()
    """Every invariant, checked once more after the full narrative
    (including recovery from every injected failure) has completed --
    the authoritative "did the session end in a genuinely consistent
    state" signal, as opposed to the per-stage snapshots above."""

    @property
    def final_invariants_ok(self) -> bool:
        return all(
            result.ok or result.status is InvariantStatus.SKIPPED
            for result in self.final_invariants
        )

    @property
    def all_ok(self) -> bool:
        return all(stage.ok for stage in self.stages) and self.final_invariants_ok

    @property
    def failed_stages(self) -> list[StageResult]:
        return [stage for stage in self.stages if not stage.ok]


def _check(env: ValidationEnvironment) -> tuple[InvariantResult, ...]:
    return tuple(
        check_invariants(
            position_tracker=env.position_tracker,
            order_manager=env.order_manager,
            broker=env.broker,
            circuit_breaker=env.circuit_breaker,
            reconciliation_engine=env.reconciliation_engine,
            state_store=env.state_store,
        )
    )


def _record(
    stages: list[StageResult],
    name: str,
    category: str,
    ok: bool,
    detail: str,
    env: ValidationEnvironment,
) -> StageResult:
    result = StageResult(
        name=name,
        category=category,
        status=StageStatus.PASS if ok else StageStatus.FAIL,
        detail=detail,
        invariants=_check(env),
    )
    stages.append(result)
    return result


def run_end_to_end_validation(env: ValidationEnvironment) -> SessionReport:
    stages: list[StageResult] = []
    started_at = dt.datetime.now(dt.UTC)
    live_start = env.market.sessions.index(env.train_end) + 5
    live_dates = env.market.sessions[live_start : live_start + 5]

    _stage_ingestion(env, stages)
    _stage_features_and_hmm(env, stages)
    _stage_daily_cycle(env, stages, live_dates[0])
    _stage_accounting(env, stages)

    _failure_broker_timeout(env, stages, live_dates[1])
    _failure_rejected_order(env, stages)
    _failure_partial_fill(env, stages, live_dates[1])
    _failure_duplicate_event(env, stages)

    _failure_lost_websocket(env, stages, live_dates[2])
    _failure_delayed_market_data(env, stages, live_dates[2])
    _stage_monitoring(env, stages)

    _stage_shutdown(env, stages)
    _failure_application_crash(env, stages)
    _failure_database_restart(env, stages)
    _stage_reconciliation(env, stages)

    _stage_halted_state_persists(env, stages)

    return SessionReport(
        started_at=started_at,
        finished_at=dt.datetime.now(dt.UTC),
        market_sessions=len(env.market.sessions),
        live_session_dates=tuple(live_dates),
        stages=stages,
        final_invariants=_check(env),
    )


# --------------------------------------------------------------------------
# The required narrative
# --------------------------------------------------------------------------


def _stage_ingestion(env: ValidationEnvironment, stages: list[StageResult]) -> None:
    """Already ran inside ``build_validation_environment`` (ingestion has
    to happen before anything else can be wired); this stage verifies the
    result is actually readable back through the same
    ``MarketDataProvider`` interface the rest of the system uses, rather
    than just trusting the ingest call didn't raise.
    """
    ok = True
    details = []
    for instrument_id in env.market.instrument_ids:
        available = env.market_data.available_range(instrument_id)
        if available is None:
            ok = False
            details.append(f"{instrument_id}: no bars readable")
    index_range_ok = bool(
        env.market_data.get_index_observations(
            "NIFTY50", env.market.first_session, env.market.last_session
        )
    )
    if not index_range_ok:
        ok = False
        details.append("NIFTY50 series unreadable")
    detail = (
        f"{len(env.market.instrument_ids)} instrument(s), "
        f"{len(env.market.sessions)} session(s) ingested and read back"
        if ok
        else "; ".join(details)
    )
    _record(stages, "market-data ingestion", "stage", ok, detail, env)


def _stage_features_and_hmm(env: ValidationEnvironment, stages: list[StageResult]) -> None:
    """Feature calculation and the HMM, verified as
    ``RegimeComputer`` -- the same live-inference path the orchestrator
    itself uses -- actually does at the trained model's own training-end
    date, before any daily cycle runs."""
    try:
        target, state = env.orchestrator.regime_computer.compute_today(
            env.model_artifact, env.train_end
        )
        detail = (
            f"regime {target.regime.value} (label {state.label.value}, "
            f"confidence {state.confidence:.2f}) as of {env.train_end}"
        )
        ok = True
    except Exception as exc:  # noqa: BLE001 - a stage failure is recorded, not raised
        detail = f"feature/HMM computation failed: {exc}"
        ok = False
    _record(stages, "feature calculation + HMM", "stage", ok, detail, env)


def _stage_daily_cycle(
    env: ValidationEnvironment, stages: list[StageResult], as_of: dt.date
) -> None:
    """Stock ranking, portfolio construction, risk management, paper
    execution and fills all happen inside one
    ``Orchestrator.run_daily_cycle`` call -- decomposed below into the
    granular stages the brief names, from the one real report it produced,
    rather than re-deriving each step outside the orchestrator (which
    would duplicate exactly the sequencing logic this validation exists to
    prove).
    """
    env.clock.set(dt.datetime.combine(as_of, dt.time(16, 0), tzinfo=dt.UTC))
    env.feed.advance_to(as_of)
    report = env.orchestrator.run_daily_cycle(as_of)

    _record(
        stages,
        "stock ranking",
        "stage",
        len(report.candidates) > 0,
        f"{len(report.candidates)} candidate(s) ranked",
        env,
    )
    _record(
        stages,
        "portfolio construction",
        "stage",
        report.target_portfolio is not None,
        f"{len(report.target_portfolio.positions)} target position(s), "
        f"cash_weight={report.target_portfolio.cash_weight:.2%}"
        if report.target_portfolio is not None
        else "no target portfolio produced",
        env,
    )
    _record(
        stages,
        "risk management",
        "stage",
        report.permit_trading,
        f"{sum(1 for d in report.risk_decisions if d.approved)}/{len(report.risk_decisions)} "
        f"position(s) risk-approved, circuit breaker "
        f"{env.circuit_breaker.current_status().state.value}",
        env,
    )
    _record(
        stages,
        "paper execution",
        "stage",
        len(report.submitted_order_ids) > 0,
        f"{len(report.submitted_order_ids)} order(s) submitted to the paper broker",
        env,
    )
    filled = [
        env.order_manager.get(oid)
        for oid in report.submitted_order_ids
        if env.order_manager.get(oid).state in (OrderState.FILLED, OrderState.PARTIALLY_FILLED)
    ]
    _record(
        stages,
        "fills",
        "stage",
        len(filled) > 0,
        f"{len(filled)}/{len(report.submitted_order_ids)} order(s) filled or partially filled",
        env,
    )


def _stage_accounting(env: ValidationEnvironment, stages: list[StageResult]) -> None:
    """Portfolio accounting: cash actually moved by the notional of the
    fills the canonical tracker observed, and every held position's
    average price is positive and finite -- the arithmetic
    ``execution.position_tracker.PositionTracker`` does on every fill,
    checked here against the broker's own account figure rather than
    trusted blindly.
    """
    positions = env.position_tracker.current_positions()
    cash = env.broker.get_account().cash
    holdings_value = sum(p.quantity * p.avg_price for p in positions)
    accounted_for = holdings_value + cash
    starting_cash = 50_000_000.0
    drift = abs(accounted_for - starting_cash) / starting_cash
    # Costs (brokerage, STT, GST, slippage) are real and expected to widen
    # this gap slightly; a large drift would mean money appeared or
    # vanished, which a cost model never does.
    ok = drift < 0.02 and all(p.avg_price > 0 for p in positions)
    detail = (
        f"holdings {holdings_value:,.2f} + cash {cash:,.2f} = {accounted_for:,.2f} "
        f"vs starting {starting_cash:,.2f} (drift {drift:.4%}, expected from trading costs)"
    )
    _record(stages, "portfolio accounting", "stage", ok, detail, env)


def _stage_monitoring(env: ValidationEnvironment, stages: list[StageResult]) -> None:
    snapshot = env.orchestrator.publish_monitoring_snapshot()
    ok = snapshot is not None
    detail = "no snapshot produced"
    if snapshot is not None:
        text = render_dashboard(snapshot)
        widths = {len(line) for line in text.splitlines()}
        alerts_evaluated = True
        try:
            env.alert_manager.evaluate(snapshot)
        except Exception as exc:  # noqa: BLE001
            alerts_evaluated = False
            detail = f"alert evaluation raised: {exc}"
        ok = len(widths) == 1 and alerts_evaluated
        if ok:
            detail = (
                f"dashboard rendered ({len(text.splitlines())} lines, width {widths.pop()}); "
                f"{len(env.alert_manager.history)} alert(s) delivered so far"
            )
    _record(stages, "monitoring", "stage", ok, detail, env)


def _stage_shutdown(env: ValidationEnvironment, stages: list[StageResult]) -> None:
    pre_shutdown_state = env.orchestrator.state
    env.orchestrator._shutdown(restore_handlers=False)  # noqa: SLF001 - validation, not a caller
    persisted = env.state_store.load()
    close_calls = env.broker.call_counts.get("close_all_positions", 0)
    ok = (
        env.orchestrator.state is OrchestratorState.SHUTTING_DOWN
        and persisted is not None
        and close_calls == 0
    )
    detail = (
        f"pre-shutdown state was {pre_shutdown_state.value}; final state persisted "
        f"({persisted.system_state.value if persisted else 'nothing persisted'}); "
        f"positions closed on shutdown: {close_calls > 0} "
        "(must be False -- close_positions_on_shutdown was not configured)"
    )
    _record(stages, "shutdown", "stage", ok, detail, env)


def _stage_reconciliation(env: ValidationEnvironment, stages: list[StageResult]) -> None:
    """The operator recovery Phase 18/17 actually expect -- neither a
    position mismatch nor an orphaned open order is ever auto-resolved,
    on principle, so an operator resolves each explicitly before calling
    ``acknowledge_and_recover``:

    - a broker position with no local record: the operator confirms the
      broker's own reported truth and brings local state in line with it
      (mirrors ``execution.startup.StartupSequence.acknowledge_and_recover``'s
      own documented recovery path).
    - an open order the fresh process has no record of (the resting
      remainder of the partial-fill order from before the crash): this
      system has no signal/risk lineage for it and Phase 17's own design
      refuses to adopt one blind, so the only safe resolution is to cancel
      it at the broker and let a human re-decide whether to replace it --
      exactly what ``execution.order_reconciler.OrderReconciler``'s own
      docstring prescribes for an orphan.
    """
    from backtest.costs import TradeSide

    for position in env.broker.get_positions():
        locally_unknown = env.position_tracker.held_quantity(position.instrument_id) == 0
        if locally_unknown and position.quantity > 0:
            env.position_tracker.apply_fill(
                position.instrument_id,
                position.quantity,
                position.avg_price,
                TradeSide.BUY,
                env.clock.now,
            )

    pre_check = env.orchestrator.order_reconciler.reconcile_after_reconnect(
        env.order_manager, env.broker
    )
    cancelled = []
    for orphan in pre_check.orphaned_broker_orders:
        # PaperBroker's own order table is keyed by client_order_id (the
        # id whoever placed the order generated) -- broker_order_id is a
        # separate, broker-assigned identifier reported alongside it, not
        # what cancel_order expects here.
        env.broker.cancel_order(orphan.client_order_id)
        cancelled.append(orphan.client_order_id)

    report = env.orchestrator.startup_sequence.acknowledge_and_recover(
        _OPERATOR,
        "confirmed broker's reported positions match the operator's own review; "
        f"cancelled {len(cancelled)} unrecognized open order(s) with no recoverable lineage",
    )
    ok = report.permit_strategy_execution and report.system_state is SystemState.READY
    _record(
        stages,
        "reconciliation",
        "stage",
        ok,
        f"{len(pre_check.orphaned_broker_orders)} orphaned order(s) found and cancelled; "
        f"post-recovery state: {report.system_state.value}, "
        f"permit_strategy_execution={report.permit_strategy_execution}",
        env,
    )


def _stage_halted_state_persists(env: ValidationEnvironment, stages: list[StageResult]) -> None:
    """The "halted state persists" invariant is only meaningful once
    something has actually halted -- this stage manufactures exactly that
    (the same deliberately-breaching ``PortfolioRiskState`` Phase 19's own
    tests use) so the invariant is verified as PASS, not merely reported
    as "nothing to check" for the whole session.
    """
    from risk.portfolio_risk_state import PortfolioRiskState

    breaching_state = PortfolioRiskState(
        as_of=env.clock.now,
        equity=env.broker.get_account().equity,
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
    env.circuit_breaker.evaluate(breaching_state)
    result = _record(
        stages,
        "halted state persists (deliberately triggered)",
        "stage",
        env.circuit_breaker.current_status().state.value == "halted",
        f"circuit breaker: {env.circuit_breaker.current_status().state.value}",
        env,
    )
    # Restore normal operation so the report's final invariant table (the
    # summary a reader checks last) does not end on a manufactured halt --
    # the manual-reset path itself is exactly what a real operator uses.
    env.circuit_breaker.manual_reset(_OPERATOR, "clearing the deliberately-triggered halt")
    result.invariants = _check(env)


# --------------------------------------------------------------------------
# The eight required failure injections
# --------------------------------------------------------------------------


def _failure_lost_websocket(
    env: ValidationEnvironment, stages: list[StageResult], as_of: dt.date
) -> None:
    env.feed.disconnect()
    try:
        # check_market_data_freshness reads historical bar coverage
        # (available_range), which a lost streaming connection does not
        # affect -- the outage shows up in whether a *live quote* can be
        # served, which is exactly what PaperBroker asks for on every
        # order it prices.
        try:
            env.feed.get_quote(env.market.instrument_ids[0])
            quote_blocked = False
        except DataNotAvailableError:
            quote_blocked = True
        broker_health = env.broker.health_check()
        detail = (
            f"a quote request during the outage was correctly blocked: {quote_blocked}; "
            f"broker health check during the outage: {broker_health.detail}"
        )
        ok = quote_blocked
    finally:
        env.feed.reconnect()
    _record(stages, "lost WebSocket", "failure_injection", ok, detail, env)


def _failure_delayed_market_data(
    env: ValidationEnvironment, stages: list[StageResult], as_of: dt.date
) -> None:
    env.feed.delay_by(dt.timedelta(minutes=45))
    try:
        quote = env.feed.get_quote(env.market.instrument_ids[0])
        age = (env.clock.now - quote.as_of).total_seconds()
        stale_guard_triggers = age > env.settings.execution.stale_quote_seconds
        detail = (
            f"quote stamped {age:.0f}s old against a "
            f"{env.settings.execution.stale_quote_seconds}s guard "
            f"-- PaperBroker's own staleness check would reject an order against it: "
            f"{stale_guard_triggers}"
        )
        ok = stale_guard_triggers
    finally:
        env.feed.clear_delay()
    _record(stages, "delayed market data", "failure_injection", ok, detail, env)


def _failure_broker_timeout(
    env: ValidationEnvironment, stages: list[StageResult], as_of: dt.date
) -> None:
    """The literal Phase 17 CRITICAL scenario, run inside the full
    end-to-end session: the broker accepts an order but the response never
    arrives. ``OrderManager.submit`` marks it ``UNKNOWN`` rather than
    guessing; the next reconciliation sweep (run right here, not deferred
    to the restart stage) resolves it by querying the broker -- the order
    is never resubmitted.
    """
    env.clock.set(dt.datetime.combine(as_of, dt.time(10, 0), tzinfo=dt.UTC))
    env.feed.advance_to(as_of)
    instrument_id = env.market.instrument_ids[0]
    quote = env.feed.get_quote(instrument_id)
    current_quantity = env.position_tracker.held_quantity(instrument_id)
    if current_quantity < 1:
        _record(
            stages,
            "broker API timeout",
            "failure_injection",
            False,
            f"nothing held in {instrument_id} to sell; scenario ordering assumption violated",
            env,
        )
        return

    result = env.order_manager.create(
        instrument_id,
        "sell",
        1,
        "limit",
        float(quote.bid),
        idempotency_key=f"timeout-probe:{as_of.isoformat()}",
        signal_id=f"timeout-probe:{as_of.isoformat()}",
        risk_decision_id="validation-injected",
    )
    env.broker.queue_fault("place_order", TimeoutError("simulated broker API timeout"))
    submit_call_count_before = env.broker.call_counts.get("place_order", 0)
    record = env.order_manager.submit(result.order.client_order_id, env.broker)
    marked_unknown = record.state is OrderState.UNKNOWN

    resolved = env.orchestrator.order_reconciler.reconcile_after_reconnect(
        env.order_manager, env.broker
    )
    final = env.order_manager.get(result.order.client_order_id)
    resolved_cleanly = final.state is not OrderState.UNKNOWN
    never_duplicated = (
        env.broker.call_counts.get("place_order", 0) - submit_call_count_before == 1
    )
    ok = marked_unknown and resolved_cleanly and never_duplicated
    detail = (
        f"submit() marked UNKNOWN after the timeout: {marked_unknown}; "
        f"reconciliation resolved it to {final.state.value}: {resolved_cleanly}; "
        f"exactly one place_order call reached the broker (never resubmitted): "
        f"{never_duplicated}; {len(resolved.resolved_unknown)} order(s) resolved this sweep"
    )
    _record(stages, "broker API timeout", "failure_injection", ok, detail, env)


def _failure_rejected_order(env: ValidationEnvironment, stages: list[StageResult]) -> None:
    """An order the broker's own validation refuses outright -- quantity
    far beyond available cash -- exercised through the real
    ``PaperBroker._reject`` path, not a fabricated one.
    """
    instrument_id = env.market.instrument_ids[0]
    quote = env.feed.get_quote(instrument_id)
    absurd_quantity = int(env.broker.get_account().cash / float(quote.ask) * 1000)
    result = env.order_manager.create(
        instrument_id,
        "buy",
        absurd_quantity,
        "limit",
        float(quote.ask),
        idempotency_key="rejection-probe",
        signal_id="rejection-probe",
        risk_decision_id="validation-injected",
    )
    record = env.order_manager.submit(result.order.client_order_id, env.broker)
    ok = record.state is OrderState.REJECTED and bool(record.reject_reason)
    detail = f"order for {absurd_quantity} shares -> {record.state.value} ({record.reject_reason})"
    _record(stages, "rejected order", "failure_injection", ok, detail, env)


def _failure_partial_fill(
    env: ValidationEnvironment, stages: list[StageResult], as_of: dt.date
) -> None:
    """Depth thinner than the order, so ``PaperBroker`` matches only
    ``paper_trading.max_fill_participation_pct`` of it -- real matching
    logic, not a fabricated partial state. Polls fills before returning,
    the same way a live loop's next monitoring iteration would, so later
    stages see a canonical tracker already caught up with this fill
    rather than reading a momentary, expected lag as a defect.
    """
    instrument_id = env.market.instrument_ids[1]
    original_depth = env.feed.depth
    env.feed.depth = 500
    try:
        quote = env.feed.get_quote(instrument_id)
        quantity = max(
            2,
            int(original_depth * 0.01),  # comfortably more than the thinned depth can fill at once
        )
        result = env.order_manager.create(
            instrument_id,
            "buy",
            quantity,
            "limit",
            float(quote.ask),
            idempotency_key="partial-fill-probe",
            signal_id="partial-fill-probe",
            risk_decision_id="validation-injected",
        )
        record = env.order_manager.submit(result.order.client_order_id, env.broker)
        ok = (
            record.state is OrderState.PARTIALLY_FILLED
            and 0 < record.filled_quantity < record.quantity
        )
        detail = f"{record.filled_quantity}/{record.quantity} filled against thinned depth"
    finally:
        env.feed.depth = original_depth
    env.orchestrator.fill_tracker.poll(env.broker)
    _record(stages, "partial fill", "failure_injection", ok, detail, env)


def _failure_duplicate_event(env: ValidationEnvironment, stages: list[StageResult]) -> None:
    """A single, fresh, self-contained fill this stage creates itself
    (not leftover state from an earlier one), redelivered twice in the
    same ``get_trades()`` response -- exactly the shape a redelivered
    broker event has. ``FillTracker`` must apply it once, not twice; this
    is the regression the fix in ``orchestration/fill_tracker.py``
    (found by an earlier run of this very scenario) exists to prevent.
    """
    instrument_id = env.market.instrument_ids[2]
    quote = env.feed.get_quote(instrument_id)
    before = env.position_tracker.held_quantity(instrument_id)

    result = env.order_manager.create(
        instrument_id,
        "buy",
        50,
        "limit",
        float(quote.ask),
        idempotency_key="duplicate-event-probe",
        signal_id="duplicate-event-probe",
        risk_decision_id="validation-injected",
    )
    record = env.order_manager.submit(result.order.client_order_id, env.broker)
    if record.filled_quantity == 0:
        _record(
            stages,
            "duplicate event",
            "failure_injection",
            False,
            f"probe order did not fill at all (state={record.state.value}); "
            "nothing to redeliver",
            env,
        )
        return

    env.broker.duplicate_next_trades = True
    new_fills = env.orchestrator.fill_tracker.poll(env.broker)
    after = env.position_tracker.held_quantity(instrument_id)
    expected = before + record.filled_quantity
    ok = after == expected and len(new_fills) == 1
    detail = (
        f"one fill for {record.filled_quantity} shares, delivered twice in one response; "
        f"{len(new_fills)} fill(s) applied (expected 1); "
        f"{instrument_id} quantity {before} -> {after} (expected {expected})"
    )
    _record(stages, "duplicate event", "failure_injection", ok, detail, env)


def _failure_application_crash(env: ValidationEnvironment, stages: list[StageResult]) -> None:
    """Every in-memory object discarded and rebuilt, sharing only the
    broker (a real crash would leave a separate broker process running)
    and whatever was persisted to disk. The fresh system must not silently
    resume trading: the broker still holds positions the fresh
    ``PositionTracker`` has never heard of, which the very next startup
    sequence must catch, not paper over.

    **A finding worth stating plainly, not just implied by the numbers
    below:** ``execution.execution_journal.ExecutionJournal`` is
    documented (Phase 17) as in-memory only -- "not a persistent store".
    This crash therefore genuinely loses the pre-crash orders' audit
    trail; the "all orders traceable" invariant holds for orders created
    after the crash, not retroactively for the ones before it. That is an
    existing, already-documented scope boundary (durable journal
    persistence is deferred to ``storage/``, still an unimplemented
    stub), not something this validation phase fixes -- but an end-to-end
    run is exactly what should say so out loud rather than leave it
    implicit in a module docstring.
    """
    broker_positions_before = {p.instrument_id: p.quantity for p in env.broker.get_positions()}
    orders_before = len(env.order_manager.all_orders())

    env.rebuild_after_crash()

    orders_after_rebuild = len(env.order_manager.all_orders())
    startup_report = env.orchestrator.startup_sequence.run()
    caught_the_gap = not startup_report.permit_strategy_execution
    no_new_orders = orders_after_rebuild == 0  # a fresh in-memory OrderManager starts empty
    detail = (
        f"broker still holds {broker_positions_before}; the fresh process's own order "
        f"history is empty (orders_before_crash={orders_before}, "
        f"orders_after_rebuild={orders_after_rebuild}); startup correctly refused to permit "
        f"trading until this is resolved: {caught_the_gap} "
        f"(system_state={startup_report.system_state.value}). "
        f"NOTE: the {orders_before} pre-crash order(s)' audit trail is lost with them -- "
        "ExecutionJournal is in-memory only (Phase 17's own documented scope boundary, "
        "durable persistence deferred to storage/)."
    )
    ok = caught_the_gap and no_new_orders
    _record(stages, "application crash", "failure_injection", ok, detail, env)


def _failure_database_restart(env: ValidationEnvironment, stages: list[StageResult]) -> None:
    """The persisted-state file is corrupted mid-session (a database
    restart landing mid-write is the closest real analogue for this
    system's single-file store) -- startup must refuse to guess at
    unreadable state, then recover once the file is repaired.
    """
    state_path = env.state_store.state_path
    original = state_path.read_bytes() if state_path.is_file() else None
    state_path.write_text("{not valid json, mid-write corruption", encoding="utf-8")

    raised = False
    try:
        env.orchestrator.startup_sequence.run()
    except Exception as exc:  # noqa: BLE001 - StartupError specifically, checked below
        raised = "StartupError" in type(exc).__name__

    if original is not None:
        state_path.write_bytes(original)
    else:
        state_path.unlink(missing_ok=True)

    recovered = False
    detail_suffix = "no prior state to restore, repair step skipped"
    if original is not None:
        try:
            report = env.orchestrator.startup_sequence.run()
            recovered = report.schema_version_ok and report.database_verified
            detail_suffix = f"post-repair system_state={report.system_state.value}"
        except Exception as exc:  # noqa: BLE001
            detail_suffix = f"still failing after repair: {exc}"

    ok = raised and (recovered or original is None)
    detail = (
        f"corrupted state file raised a StartupError: {raised}; after restoring the file, "
        f"{detail_suffix}"
    )
    _record(stages, "database restart", "failure_injection", ok, detail, env)
