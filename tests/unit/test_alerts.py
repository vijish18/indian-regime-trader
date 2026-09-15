"""Unit tests for ``monitoring/alerts.py`` (Phase 20): each of the ten
alert conditions fires exactly when it should and not otherwise, and the
rate limiter stops a condition that persists from producing one alert per
monitoring-loop iteration.
"""

from __future__ import annotations

import datetime as dt
import logging

import pytest

from data.models import SessionType
from execution.reconciliation import ReconciliationMismatch
from monitoring.alerts import (
    Alert,
    AlertConfigurationError,
    AlertManager,
    AlertSeverity,
    AlertType,
    RateLimiter,
    evaluate_alerts,
)
from monitoring.health import ComponentHealth
from monitoring.snapshot import (
    ExecutionPanel,
    MonitoringSnapshot,
    PortfolioPanel,
    RegimePanel,
    RiskPanel,
    SystemPanel,
)
from risk.circuit_breaker import CircuitState

_T0 = dt.datetime(2024, 6, 3, 10, 0, tzinfo=dt.UTC)
_DRAWDOWN_ALERT_PCT = 0.10


class _ClockBox:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + dt.timedelta(**kwargs)


def _snapshot(
    *,
    broker_connected: bool = True,
    data_feed: ComponentHealth = ComponentHealth.HEALTHY,
    rejected_orders: int = 0,
    orders_in_unknown_state: int = 0,
    mismatches: tuple[ReconciliationMismatch, ...] = (),
    risk_mode: CircuitState = CircuitState.NORMAL,
    drawdown_pct: float = 0.0,
    cash: float = 1_000_000.0,
    equity: float = 10_000_000.0,
    cumulative_fill_cash_flow: float = 0.0,
) -> MonitoringSnapshot:
    return MonitoringSnapshot(
        as_of=_T0,
        system=SystemPanel(
            status="running",
            uptime=dt.timedelta(hours=1),
            session_date=_T0.date(),
            session_type=SessionType.REGULAR,
            session_detail="09:15-15:30",
            data_feed=data_feed,
            data_feed_detail="feed detail",
            broker_connected=broker_connected,
            broker_detail="broker detail",
            model_version="model-1",
        ),
        portfolio=PortfolioPanel(
            equity=equity,
            cash=cash,
            gross_exposure_pct=0.5,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            drawdown_pct=drawdown_pct,
            position_count=2,
            cumulative_fill_cash_flow=cumulative_fill_cash_flow,
        ),
        regime=RegimePanel(
            regime="normal_risk",
            label="calm",
            probability=0.8,
            confidence=0.7,
            persistence=0.9,
            india_vix=14.0,
            india_vix_change=0.1,
            nifty_close=23000.0,
            nifty_change_pct=0.001,
            as_of=_T0.date(),
        ),
        execution=ExecutionPanel(
            orders_submitted=5,
            fills=4,
            rejected_orders=rejected_orders,
            open_orders=1,
            orders_in_unknown_state=orders_in_unknown_state,
            reconciliation_mismatches=mismatches,
        ),
        risk=RiskPanel(
            risk_mode=risk_mode,
            circuit_reason="daily loss breached" if risk_mode is not CircuitState.NORMAL else None,
            circuit_triggered_by="daily_loss_halt"
            if risk_mode is not CircuitState.NORMAL
            else None,
            requires_manual_recovery=risk_mode is CircuitState.HALTED,
            max_concentration_pct=0.2,
            concentration_instrument="NSE:INFY",
            daily_turnover_pct=0.1,
        ),
    )


def _evaluate(
    current: MonitoringSnapshot, previous: MonitoringSnapshot | None = None
) -> list[Alert]:
    return evaluate_alerts(
        current, previous, drawdown_alert_pct=_DRAWDOWN_ALERT_PCT, cash_tolerance_pct=0.001
    )


def _types(alerts: list[Alert]) -> set[AlertType]:
    return {alert.alert_type for alert in alerts}


# --------------------------------------------------------------------------
# A healthy system raises nothing
# --------------------------------------------------------------------------


def test_a_healthy_snapshot_raises_no_alerts() -> None:
    assert _evaluate(_snapshot()) == []


def test_a_healthy_pair_of_snapshots_raises_no_alerts() -> None:
    assert _evaluate(_snapshot(), _snapshot()) == []


# --------------------------------------------------------------------------
# The ten conditions
# --------------------------------------------------------------------------


def test_broker_disconnect_alerts() -> None:
    alerts = _evaluate(_snapshot(broker_connected=False))
    assert _types(alerts) == {AlertType.BROKER_DISCONNECT}
    assert alerts[0].severity is AlertSeverity.CRITICAL


def test_market_data_disconnect_alerts_when_the_feed_is_unreachable() -> None:
    alerts = _evaluate(_snapshot(data_feed=ComponentHealth.UNHEALTHY))
    assert _types(alerts) == {AlertType.MARKET_DATA_DISCONNECT}
    assert alerts[0].severity is AlertSeverity.CRITICAL


def test_stale_data_alerts_when_the_feed_is_merely_behind() -> None:
    alerts = _evaluate(_snapshot(data_feed=ComponentHealth.DEGRADED))
    assert _types(alerts) == {AlertType.STALE_DATA}
    assert alerts[0].severity is AlertSeverity.WARNING


def test_a_degraded_feed_does_not_also_raise_a_disconnect() -> None:
    """The two are different problems with different responses, so a stale
    feed must not masquerade as an unreachable one."""
    alerts = _evaluate(_snapshot(data_feed=ComponentHealth.DEGRADED))
    assert AlertType.MARKET_DATA_DISCONNECT not in _types(alerts)


def test_order_rejection_alerts_on_first_sight() -> None:
    alerts = _evaluate(_snapshot(rejected_orders=1))
    assert AlertType.ORDER_REJECTION in _types(alerts)


def test_order_rejection_alerts_only_on_an_increase() -> None:
    previous = _snapshot(rejected_orders=2)
    unchanged = _evaluate(_snapshot(rejected_orders=2), previous)
    assert AlertType.ORDER_REJECTION not in _types(unchanged)

    increased = _evaluate(_snapshot(rejected_orders=3), previous)
    assert AlertType.ORDER_REJECTION in _types(increased)


def test_unknown_order_state_alerts() -> None:
    alerts = _evaluate(_snapshot(orders_in_unknown_state=2))
    assert _types(alerts) == {AlertType.UNKNOWN_ORDER_STATE}
    assert alerts[0].severity is AlertSeverity.CRITICAL


def test_reconciliation_mismatch_alerts_per_instrument() -> None:
    mismatches = (
        ReconciliationMismatch("NSE:INFY", 10, 12, "quantity mismatch"),
        ReconciliationMismatch("NSE:TCS", 5, 4, "quantity mismatch"),
    )
    alerts = _evaluate(_snapshot(mismatches=mismatches))
    assert _types(alerts) == {AlertType.RECONCILIATION_MISMATCH}
    assert {alert.subject for alert in alerts} == {"NSE:INFY", "NSE:TCS"}


def test_a_local_position_the_broker_no_longer_reports_is_a_mismatch() -> None:
    mismatches = (ReconciliationMismatch("NSE:INFY", 10, 0, "broker no longer reports"),)
    alerts = _evaluate(_snapshot(mismatches=mismatches))
    assert _types(alerts) == {AlertType.RECONCILIATION_MISMATCH}


def test_risk_halt_alerts() -> None:
    alerts = _evaluate(_snapshot(risk_mode=CircuitState.HALTED))
    assert AlertType.RISK_HALT in _types(alerts)
    assert alerts[0].severity is AlertSeverity.CRITICAL


def test_reduced_risk_is_not_a_halt() -> None:
    alerts = _evaluate(_snapshot(risk_mode=CircuitState.REDUCED_RISK))
    assert AlertType.RISK_HALT not in _types(alerts)


def test_excessive_drawdown_alerts_at_the_threshold() -> None:
    assert AlertType.EXCESSIVE_DRAWDOWN not in _types(_evaluate(_snapshot(drawdown_pct=0.09)))
    assert AlertType.EXCESSIVE_DRAWDOWN in _types(_evaluate(_snapshot(drawdown_pct=0.10)))
    assert AlertType.EXCESSIVE_DRAWDOWN in _types(_evaluate(_snapshot(drawdown_pct=0.25)))


def test_unexpected_position_alerts_when_the_broker_holds_something_unknown() -> None:
    mismatches = (ReconciliationMismatch("NSE:GHOST", 0, 10, "no local record"),)
    alerts = _evaluate(_snapshot(mismatches=mismatches))
    assert _types(alerts) == {AlertType.UNEXPECTED_POSITION}
    assert alerts[0].subject == "NSE:GHOST"


def test_unexpected_cash_balance_alerts_on_an_unexplained_move() -> None:
    previous = _snapshot(cash=1_000_000.0, cumulative_fill_cash_flow=0.0)
    # Cash fell by 500,000 but no fills were observed to explain it.
    current = _snapshot(cash=500_000.0, cumulative_fill_cash_flow=0.0)
    alerts = _evaluate(current, previous)
    assert AlertType.UNEXPECTED_CASH_BALANCE in _types(alerts)


def test_cash_that_matches_the_observed_fills_raises_nothing() -> None:
    previous = _snapshot(cash=1_000_000.0, cumulative_fill_cash_flow=0.0)
    # A 400,000 buy: cash down 400,000, fill cash flow down 400,000.
    current = _snapshot(cash=600_000.0, cumulative_fill_cash_flow=-400_000.0)
    assert AlertType.UNEXPECTED_CASH_BALANCE not in _types(_evaluate(current, previous))


def test_cash_within_tolerance_raises_nothing() -> None:
    """The fill cash flow is gross of brokerage and taxes, so cash always
    drifts slightly below what the fills alone imply."""
    previous = _snapshot(cash=1_000_000.0, cumulative_fill_cash_flow=0.0)
    current = _snapshot(cash=600_000.0 - 2_000.0, cumulative_fill_cash_flow=-400_000.0)
    # 2,000 of costs against 10,000,000 equity is 0.02%, inside the 0.1% tolerance.
    assert AlertType.UNEXPECTED_CASH_BALANCE not in _types(_evaluate(current, previous))


def test_unexpected_cash_needs_a_previous_snapshot() -> None:
    assert AlertType.UNEXPECTED_CASH_BALANCE not in _types(_evaluate(_snapshot(cash=1.0)))


def test_several_conditions_at_once_all_alert() -> None:
    alerts = _evaluate(
        _snapshot(
            broker_connected=False,
            data_feed=ComponentHealth.UNHEALTHY,
            orders_in_unknown_state=1,
            risk_mode=CircuitState.HALTED,
            drawdown_pct=0.30,
            mismatches=(ReconciliationMismatch("NSE:GHOST", 0, 10, "no local record"),),
        )
    )
    assert _types(alerts) == {
        AlertType.BROKER_DISCONNECT,
        AlertType.MARKET_DATA_DISCONNECT,
        AlertType.UNKNOWN_ORDER_STATE,
        AlertType.RISK_HALT,
        AlertType.EXCESSIVE_DRAWDOWN,
        AlertType.UNEXPECTED_POSITION,
    }


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


def test_rate_limiter_allows_the_first_alert() -> None:
    limiter = RateLimiter(300.0, clock=_ClockBox(_T0))
    allowed, suppressed = limiter.allow("broker_disconnect")
    assert allowed is True
    assert suppressed == 0


def test_rate_limiter_suppresses_within_the_cooldown() -> None:
    clock = _ClockBox(_T0)
    limiter = RateLimiter(300.0, clock=clock)
    limiter.allow("broker_disconnect")
    clock.advance(seconds=299)
    allowed, suppressed = limiter.allow("broker_disconnect")
    assert allowed is False
    assert suppressed == 1


def test_rate_limiter_allows_again_after_the_cooldown() -> None:
    clock = _ClockBox(_T0)
    limiter = RateLimiter(300.0, clock=clock)
    limiter.allow("broker_disconnect")
    clock.advance(seconds=300)
    allowed, _ = limiter.allow("broker_disconnect")
    assert allowed is True


def test_rate_limiter_reports_and_resets_the_suppressed_count() -> None:
    clock = _ClockBox(_T0)
    limiter = RateLimiter(300.0, clock=clock)
    limiter.allow("k")
    for _ in range(4):
        limiter.allow("k")
    assert limiter.suppressed_count("k") == 4

    clock.advance(seconds=301)
    allowed, suppressed = limiter.allow("k")
    assert allowed is True
    assert suppressed == 4
    assert limiter.suppressed_count("k") == 0


def test_rate_limiting_is_per_key() -> None:
    limiter = RateLimiter(300.0, clock=_ClockBox(_T0))
    assert limiter.allow("reconciliation_mismatch:NSE:INFY")[0] is True
    assert limiter.allow("reconciliation_mismatch:NSE:TCS")[0] is True


def test_a_zero_cooldown_never_suppresses() -> None:
    limiter = RateLimiter(0.0, clock=_ClockBox(_T0))
    assert limiter.allow("k")[0] is True
    assert limiter.allow("k")[0] is True


def test_a_negative_cooldown_is_refused() -> None:
    with pytest.raises(ValueError, match="cooldown_seconds"):
        RateLimiter(-1.0)


# --------------------------------------------------------------------------
# AlertManager
# --------------------------------------------------------------------------


def _manager(clock: _ClockBox, **overrides: object) -> AlertManager:
    kwargs: dict[str, object] = dict(
        cooldown_seconds=300.0,
        drawdown_alert_pct=_DRAWDOWN_ALERT_PCT,
        clock=clock,
    )
    kwargs.update(overrides)
    return AlertManager([], **kwargs)  # type: ignore[arg-type]


def test_an_unsupported_channel_is_refused_at_construction() -> None:
    with pytest.raises(AlertConfigurationError, match="slack"):
        AlertManager(["slack"], cooldown_seconds=300.0, drawdown_alert_pct=0.1)


def test_the_log_channel_is_supported() -> None:
    manager = AlertManager(["log"], cooldown_seconds=300.0, drawdown_alert_pct=0.1)
    assert manager.channels == ["log"]


def test_evaluate_delivers_alerts_and_records_them() -> None:
    clock = _ClockBox(_T0)
    manager = _manager(clock)
    delivered = manager.evaluate(_snapshot(broker_connected=False))
    assert _types(delivered) == {AlertType.BROKER_DISCONNECT}
    assert len(manager.history) == 1
    assert manager.history[0].raised_at == _T0


def test_a_persisting_condition_is_rate_limited_across_iterations() -> None:
    clock = _ClockBox(_T0)
    manager = _manager(clock)
    for _ in range(5):
        manager.evaluate(_snapshot(broker_connected=False))
        clock.advance(seconds=30)
    assert len(manager.history) == 1, "a disconnected broker alerted once per iteration"


def test_a_persisting_condition_alerts_again_after_the_cooldown() -> None:
    clock = _ClockBox(_T0)
    manager = _manager(clock)
    manager.evaluate(_snapshot(broker_connected=False))
    clock.advance(seconds=301)
    delivered = manager.evaluate(_snapshot(broker_connected=False))
    assert _types(delivered) == {AlertType.BROKER_DISCONNECT}
    assert len(manager.history) == 2


def test_a_repeated_alert_reports_how_many_it_stands_for() -> None:
    clock = _ClockBox(_T0)
    manager = _manager(clock)
    manager.evaluate(_snapshot(broker_connected=False))
    for _ in range(3):
        clock.advance(seconds=30)
        manager.evaluate(_snapshot(broker_connected=False))

    clock.advance(seconds=301)
    manager.evaluate(_snapshot(broker_connected=False))
    assert manager.history[-1].suppressed_since_last == 3


def test_evaluate_remembers_the_previous_snapshot_for_change_based_rules() -> None:
    clock = _ClockBox(_T0)
    manager = _manager(clock)
    manager.evaluate(_snapshot(rejected_orders=1))
    clock.advance(seconds=301)
    # Same count, so no new rejection: nothing further should be delivered.
    delivered = manager.evaluate(_snapshot(rejected_orders=1))
    assert AlertType.ORDER_REJECTION not in _types(delivered)


def test_alerts_are_dispatched_to_a_sink_when_one_is_wired() -> None:
    clock = _ClockBox(_T0)
    received: list[Alert] = []
    manager = _manager(clock, sink=received.append)
    manager.evaluate(_snapshot(broker_connected=False))
    assert [alert.alert_type for alert in received] == [AlertType.BROKER_DISCONNECT]


def test_alerts_are_always_logged_at_their_own_severity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _ClockBox(_T0)
    manager = _manager(clock)
    with caplog.at_level(logging.WARNING, logger="monitoring.alerts"):
        manager.evaluate(_snapshot(broker_connected=False, data_feed=ComponentHealth.DEGRADED))
    levels = {record.levelname for record in caplog.records}
    assert levels == {"CRITICAL", "WARNING"}


def test_history_is_bounded() -> None:
    clock = _ClockBox(_T0)
    manager = _manager(clock, cooldown_seconds=0.0, history_limit=3)
    for _ in range(10):
        manager.evaluate(_snapshot(broker_connected=False))
    assert len(manager.history) == 3
