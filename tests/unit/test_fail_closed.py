"""Fail-closed tests (Phase 23): each of the six conditions in
``orchestration.fail_closed.FailClosedReason`` must stop the daily cycle
without placing an order, name itself on the returned report, and leave
the process alive to be asked about it.

Every condition here is triggered at the boundary the real failure would
arrive at -- a calendar that refuses to answer for an uncovered year, a
state store whose file has been corrupted, a risk engine that raises --
and is then left to run through the orchestrator's *real* handling. These
are the production-safety guarantees the deployment in `docs/DEPLOYMENT.md`
depends on: a supervisor configured to restart the container automatically
turns any uncaught exception into a crash loop, which is why "do not
trade" must mean "refuse and stay up", not "die".
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from config.loader import ConfigError
from config.models import Settings
from data.errors import CalendarCoverageError
from execution.order_manager import OrderState
from orchestration.fail_closed import FailClosedReason
from orchestration.orchestrator_state import OrchestratorState
from tests.unit.test_orchestrator import Harness, env, settings  # noqa: F401 - pytest fixtures

_EVERY_REASON = set(FailClosedReason)


@pytest.fixture
def harness(env: object, settings: Settings, tmp_path: Path) -> Harness:  # noqa: F811
    environment = env  # the module-scoped Environment fixture from test_orchestrator
    return Harness(
        environment,  # type: ignore[arg-type]
        tmp_path,
        as_of=environment.dates[-1],  # type: ignore[attr-defined]
        settings=settings,
        train_end=environment.dates[200],  # type: ignore[attr-defined]
    )


def _assert_refused_to_trade(report: object, reason: FailClosedReason) -> None:
    assert report.fail_closed_reason is reason  # type: ignore[attr-defined]
    assert report.permit_trading is False  # type: ignore[attr-defined]
    assert report.submitted_order_ids == ()  # type: ignore[attr-defined]
    assert any(
        "FAIL CLOSED" in message for message in report.messages  # type: ignore[attr-defined]
    )


# --------------------------------------------------------------------------
# configuration failure -> do not trade
# --------------------------------------------------------------------------


def test_configuration_failure_refuses_to_trade(harness: Harness) -> None:
    def failing_loader() -> Settings:
        raise ConfigError("settings.yaml is unreadable")

    harness.orchestrator._settings_loader = failing_loader
    report = harness.run()
    _assert_refused_to_trade(report, FailClosedReason.CONFIGURATION_FAILURE)
    assert harness.orchestrator.state is OrchestratorState.HALTED


def test_configuration_failure_does_not_raise(harness: Harness) -> None:
    """The whole point of the Phase 23 change: a supervisor that restarts
    the process cannot report a reason for a process that no longer
    exists."""
    def failing_loader() -> Settings:
        raise ConfigError("settings.yaml is unreadable")

    harness.orchestrator._settings_loader = failing_loader
    harness.run()  # must not raise


# --------------------------------------------------------------------------
# market-calendar uncertainty -> do not trade
# --------------------------------------------------------------------------


def test_market_calendar_uncertainty_refuses_to_trade(harness: Harness) -> None:
    class _RefusingCalendar:
        def is_trading_day(self, day: dt.date) -> bool:
            raise CalendarCoverageError(f"{day.year} is not covered by the holiday file")

        def session(self, day: dt.date) -> object:
            raise CalendarCoverageError("not covered")

        def next_trading_day(self, day: dt.date) -> dt.date:
            raise CalendarCoverageError("not covered")

        def previous_trading_day(self, day: dt.date) -> dt.date:
            raise CalendarCoverageError("not covered")

        def sessions_offset(self, day: dt.date, sessions: int) -> dt.date:
            raise CalendarCoverageError("not covered")

    harness.orchestrator.calendar = _RefusingCalendar()  # type: ignore[assignment]
    report = harness.run()
    _assert_refused_to_trade(report, FailClosedReason.MARKET_CALENDAR_UNCERTAINTY)
    assert harness.orchestrator.state is OrchestratorState.HALTED


def test_an_uncovered_calendar_year_is_uncertainty_not_a_holiday(harness: Harness) -> None:
    """A calendar that cannot answer must never be read as "the exchange
    is closed" -- that would silently look like a normal quiet day."""
    class _RefusingCalendar:
        def is_trading_day(self, day: dt.date) -> bool:
            raise CalendarCoverageError("not covered")

    harness.orchestrator.calendar = _RefusingCalendar()  # type: ignore[assignment]
    report = harness.run()
    assert report.fail_closed_reason is FailClosedReason.MARKET_CALENDAR_UNCERTAINTY
    assert report.is_trading_day is not False or report.fail_closed_reason is not None


# --------------------------------------------------------------------------
# stale market data -> do not trade
# --------------------------------------------------------------------------


def test_stale_market_data_refuses_to_trade(harness: Harness) -> None:
    far_future = harness.env.dates[-1] + dt.timedelta(days=120)
    report = harness.run(far_future)
    _assert_refused_to_trade(report, FailClosedReason.STALE_MARKET_DATA)


def test_an_unverifiable_feed_is_treated_as_stale(harness: Harness) -> None:
    """"I could not check the feed" must read as "the feed is stale", not
    as "the feed is fine"."""
    def _raise(as_of: dt.date) -> object:
        raise RuntimeError("freshness check itself failed")

    harness.orchestrator.health_checker.check_market_data_freshness = _raise  # type: ignore[assignment]
    report = harness.run()
    _assert_refused_to_trade(report, FailClosedReason.STALE_MARKET_DATA)


# --------------------------------------------------------------------------
# database failure -> do not trade
# --------------------------------------------------------------------------


def test_database_failure_refuses_to_trade(harness: Harness) -> None:
    """A corrupted state store is Phase 18's own StartupError path; here
    it must surface as a named fail-closed condition rather than an
    exception escaping the cycle."""
    harness.state_store.state_path.parent.mkdir(parents=True, exist_ok=True)
    harness.state_store.state_path.write_text("{not valid json", encoding="utf-8")

    report = harness.run()
    _assert_refused_to_trade(report, FailClosedReason.DATABASE_FAILURE)
    assert harness.orchestrator.state is OrchestratorState.HALTED


# --------------------------------------------------------------------------
# risk engine failure -> do not trade
# --------------------------------------------------------------------------


def test_risk_engine_failure_refuses_to_trade(harness: Harness) -> None:
    def _raise(*args: object, **kwargs: object) -> object:
        raise RuntimeError("risk engine exploded")

    harness.orchestrator.risk_manager.evaluate = _raise  # type: ignore[assignment]
    report = harness.run()
    _assert_refused_to_trade(report, FailClosedReason.RISK_ENGINE_FAILURE)
    assert harness.orchestrator.state is OrchestratorState.HALTED


def test_an_absent_risk_verdict_is_never_read_as_approval(harness: Harness) -> None:
    """The risk layer's veto is non-negotiable, so a missing verdict must
    mean "rejected", not "nothing objected"."""
    def _raise(*args: object, **kwargs: object) -> object:
        raise RuntimeError("risk engine exploded")

    harness.orchestrator.risk_manager.evaluate = _raise  # type: ignore[assignment]
    report = harness.run()
    assert report.submitted_order_ids == ()
    assert report.sized_trades == ()


# --------------------------------------------------------------------------
# unknown broker state -> do not place more orders
# --------------------------------------------------------------------------


def test_an_unresolved_unknown_order_blocks_further_submission(harness: Harness) -> None:
    """The order this system already placed may or may not have filled.
    Until reconciliation settles that, another order on the same book
    risks doubling a position that may already exist.

    This is a defence-in-depth guard, and is tested as one: normally the
    startup step's own ``OrderReconciler`` sweep settles every UNKNOWN
    order before step 14 is reached (that is what
    ``tests/unit/test_order_reconciler.py`` proves). Here the sweep is
    stubbed out to represent reconciliation having run without being able
    to settle this one -- the state the guard exists to refuse to trade
    through, however it is arrived at.
    """
    created = harness.order_manager.create(
        "NSE:S00", "buy", 1, "limit", 100.0,
        idempotency_key="unknown-probe", signal_id="unknown-probe",
        risk_decision_id="validation-injected",
    )
    harness.order_manager.transition(created.order.client_order_id, OrderState.SUBMITTED)
    harness.order_manager.transition(created.order.client_order_id, OrderState.UNKNOWN)

    def _cannot_settle(order_manager: object, broker: object) -> object:
        from execution.order_reconciler import OrderReconciliationReport

        return OrderReconciliationReport(
            as_of=harness.clock.now,
            resolved_unknown=(),
            timed_out=(),
            refreshed_stale=(),
            orphaned_broker_orders=(),
        )

    harness.orchestrator.order_reconciler.reconcile_after_reconnect = _cannot_settle  # type: ignore[assignment]

    report = harness.run()
    _assert_refused_to_trade(report, FailClosedReason.UNKNOWN_BROKER_STATE)


def test_a_broker_position_with_no_local_record_blocks_trading(harness: Harness) -> None:
    from broker.base import BrokerPosition

    harness.broker.extra_positions = [
        BrokerPosition(instrument_id="NSE:GHOST", quantity=10, avg_price=100.0)
    ]
    report = harness.run()
    _assert_refused_to_trade(report, FailClosedReason.UNKNOWN_BROKER_STATE)


# --------------------------------------------------------------------------
# The set as a whole
# --------------------------------------------------------------------------


def test_every_fail_closed_reason_is_covered_by_a_test() -> None:
    """A reason nobody can trigger is a reason nobody has tested. Each
    member above has at least one test that asserts it is reachable; this
    guards against a seventh being added without one."""
    covered = {
        FailClosedReason.CONFIGURATION_FAILURE,
        FailClosedReason.MARKET_CALENDAR_UNCERTAINTY,
        FailClosedReason.STALE_MARKET_DATA,
        FailClosedReason.DATABASE_FAILURE,
        FailClosedReason.RISK_ENGINE_FAILURE,
        FailClosedReason.UNKNOWN_BROKER_STATE,
    }
    assert covered == _EVERY_REASON


def test_a_failing_monitoring_iteration_halts_without_killing_the_loop(
    harness: Harness,
) -> None:
    """The loop is what keeps a halted system observable -- an exception
    inside it must halt and continue, not take the process down."""
    harness.run()

    def _raise(*args: object, **kwargs: object) -> object:
        raise RuntimeError("monitoring exploded")

    harness.orchestrator._monitor_and_reconcile_once = _raise  # type: ignore[assignment]
    harness.orchestrator.run_forever(
        as_of=harness.as_of,
        max_iterations=2,
        install_signal_handlers=False,
        sleep_fn=lambda _seconds: None,
    )
    # Reached the end without propagating; the shutdown path still ran.
    assert harness.orchestrator.state is OrchestratorState.SHUTTING_DOWN
