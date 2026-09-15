"""Unit tests for ``monitoring/terminal_dashboard.py`` (Phase 20): every
field the phase brief asks for appears on screen, formatted so an operator
can read it, and the frame stays intact when a value is long, missing, or
negative.
"""

from __future__ import annotations

import datetime as dt
import io

import pytest

from data.models import SessionType
from execution.reconciliation import ReconciliationMismatch
from monitoring.health import ComponentHealth
from monitoring.snapshot import (
    ExecutionPanel,
    MonitoringSnapshot,
    PortfolioPanel,
    RegimePanel,
    RiskPanel,
    SystemPanel,
)
from monitoring.terminal_dashboard import DEFAULT_WIDTH, TerminalDashboard, render_dashboard
from risk.circuit_breaker import CircuitState

_T0 = dt.datetime(2024, 6, 3, 10, 15, tzinfo=dt.UTC)


def _snapshot(**overrides: object) -> MonitoringSnapshot:
    system = SystemPanel(
        status="running",
        uptime=dt.timedelta(hours=3, minutes=12, seconds=7),
        session_date=dt.date(2024, 6, 3),
        session_type=SessionType.REGULAR,
        session_detail="09:15-15:30",
        data_feed=ComponentHealth.HEALTHY,
        data_feed_detail="6/6 instruments fresh",
        broker_connected=True,
        broker_detail="ok",
        model_version="hmm_2024-05-31_3s_seed17_a1b2",
    )
    portfolio = PortfolioPanel(
        equity=10_432_100.55,
        cash=2_100_000.0,
        gross_exposure_pct=0.7987,
        daily_pnl=43_210.10,
        daily_pnl_pct=0.00416,
        drawdown_pct=0.0231,
        position_count=4,
        cumulative_fill_cash_flow=-8_300_000.0,
    )
    regime = RegimePanel(
        regime="normal_risk",
        label="calm",
        probability=0.87,
        confidence=0.81,
        persistence=0.93,
        india_vix=14.22,
        india_vix_change=-0.45,
        nifty_close=23_145.70,
        nifty_change_pct=0.0042,
        as_of=dt.date(2024, 6, 3),
    )
    execution = ExecutionPanel(
        orders_submitted=7,
        fills=5,
        rejected_orders=1,
        open_orders=2,
        orders_in_unknown_state=0,
        reconciliation_mismatches=(),
    )
    risk = RiskPanel(
        risk_mode=CircuitState.NORMAL,
        circuit_reason=None,
        circuit_triggered_by=None,
        requires_manual_recovery=False,
        max_concentration_pct=0.2914,
        concentration_instrument="NSE:INFY",
        daily_turnover_pct=0.1832,
    )
    panels: dict[str, object] = {
        "system": system,
        "portfolio": portfolio,
        "regime": regime,
        "execution": execution,
        "risk": risk,
    }
    panels.update(overrides)
    return MonitoringSnapshot(as_of=_T0, **panels)  # type: ignore[arg-type]


def _rendered(**overrides: object) -> str:
    return render_dashboard(_snapshot(**overrides))


# --------------------------------------------------------------------------
# Every section and every field the brief names
# --------------------------------------------------------------------------


def test_all_five_sections_are_present() -> None:
    text = _rendered()
    for heading in ("SYSTEM", "PORTFOLIO", "REGIME", "EXECUTION", "RISK"):
        assert heading in text


@pytest.mark.parametrize(
    "label",
    [
        "status",
        "uptime",
        "trading session",
        "data feed",
        "broker",
        "model version",
        "equity",
        "cash",
        "exposure",
        "daily P&L",
        "drawdown",
        "positions",
        "regime",
        "probability",
        "confidence",
        "persistence",
        "India VIX",
        "NIFTY 50",
        "orders submitted",
        "fills",
        "rejected orders",
        "open orders",
        "pending reconciliation",
        "risk mode",
        "circuit breaker",
        "concentration",
        "turnover",
    ],
)
def test_every_required_field_is_labelled(label: str) -> None:
    assert label in _rendered()


def test_system_values_are_rendered() -> None:
    text = _rendered()
    assert "RUNNING" in text
    assert "03h 12m 07s" in text
    assert "2024-06-03 regular (09:15-15:30)" in text
    assert "HEALTHY (6/6 instruments fresh)" in text
    assert "CONNECTED (ok)" in text
    assert "hmm_2024-05-31_3s_seed17_a1b2" in text


def test_portfolio_values_are_rendered() -> None:
    text = _rendered()
    assert "INR 10,432,100.55" in text
    assert "INR 2,100,000.00" in text
    assert "79.87%" in text
    assert "+INR 43,210.10 (+0.42%)" in text
    assert "2.31%" in text


def test_regime_shows_the_allocation_tier_and_the_descriptive_label() -> None:
    """The tier drives exposure; the label is reporting-only. A dashboard
    that showed only the label would imply the system acts on it."""
    text = _rendered()
    assert "NORMAL_RISK (label: calm)" in text
    assert "87.00%" in text
    assert "14.22 (-0.45)" in text
    assert "23,145.70 (+0.42%)" in text


def test_execution_values_are_rendered() -> None:
    text = _rendered()
    assert "orders submitted        7" in text
    assert "fills                   5" in text


def test_risk_values_are_rendered() -> None:
    text = _rendered()
    assert "NORMAL" in text
    assert "none tripped" in text
    assert "29.14% (NSE:INFY)" in text
    assert "18.32%" in text


# --------------------------------------------------------------------------
# Awkward values
# --------------------------------------------------------------------------


def test_a_loss_renders_with_a_minus_sign() -> None:
    text = _rendered(
        portfolio=PortfolioPanel(
            equity=9_000_000.0,
            cash=1_000_000.0,
            gross_exposure_pct=0.5,
            daily_pnl=-125_000.0,
            daily_pnl_pct=-0.0137,
            drawdown_pct=0.12,
            position_count=3,
            cumulative_fill_cash_flow=0.0,
        )
    )
    assert "-INR 125,000.00 (-1.37%)" in text


def test_a_disconnected_broker_says_so() -> None:
    text = _rendered(
        system=SystemPanel(
            status="halted",
            uptime=dt.timedelta(minutes=2),
            session_date=dt.date(2024, 6, 3),
            session_type=SessionType.REGULAR,
            session_detail="09:15-15:30",
            data_feed=ComponentHealth.UNHEALTHY,
            data_feed_detail="no data",
            broker_connected=False,
            broker_detail="connection refused",
            model_version=None,
        )
    )
    assert "HALTED" in text
    assert "DISCONNECTED (connection refused)" in text
    assert "UNHEALTHY (no data)" in text
    assert "NONE APPROVED" in text


def test_a_tripped_circuit_breaker_shows_why_and_whether_a_reset_is_needed() -> None:
    text = _rendered(
        risk=RiskPanel(
            risk_mode=CircuitState.HALTED,
            circuit_reason="daily loss 16% exceeds halt threshold",
            circuit_triggered_by="daily_loss_halt",
            requires_manual_recovery=True,
            max_concentration_pct=0.0,
            concentration_instrument=None,
            daily_turnover_pct=0.0,
        )
    )
    assert "HALTED" in text
    assert "daily_loss_halt" in text
    assert "MANUAL RESET REQUIRED" in text
    assert "no positions" in text


def test_pending_reconciliation_breaks_down_what_is_pending() -> None:
    text = _rendered(
        execution=ExecutionPanel(
            orders_submitted=3,
            fills=1,
            rejected_orders=0,
            open_orders=1,
            orders_in_unknown_state=2,
            reconciliation_mismatches=(
                ReconciliationMismatch("NSE:INFY", 10, 12, "quantity mismatch"),
            ),
        )
    )
    assert "3 (2 unknown order(s), 1 position mismatch" in text


def test_missing_regime_data_renders_as_not_available() -> None:
    text = _rendered(
        regime=RegimePanel(
            regime=None,
            label=None,
            probability=None,
            confidence=None,
            persistence=None,
            india_vix=None,
            india_vix_change=None,
            nifty_close=None,
            nifty_change_pct=None,
            as_of=None,
        )
    )
    assert "UNKNOWN" in text
    assert "n/a" in text


def test_an_index_level_without_a_prior_observation_omits_the_change() -> None:
    text = _rendered(
        regime=RegimePanel(
            regime="high_risk",
            label="crisis",
            probability=0.9,
            confidence=0.9,
            persistence=0.9,
            india_vix=31.5,
            india_vix_change=None,
            nifty_close=21_000.0,
            nifty_change_pct=None,
            as_of=dt.date(2024, 6, 3),
        )
    )
    assert "31.50" in text
    assert "21,000.00" in text


# --------------------------------------------------------------------------
# The frame itself
# --------------------------------------------------------------------------


def test_every_line_is_exactly_the_requested_width() -> None:
    for width in (48, 60, DEFAULT_WIDTH, 120):
        lines = render_dashboard(_snapshot(), width).splitlines()
        assert {len(line) for line in lines} == {width}, f"ragged frame at width {width}"


def test_a_narrow_frame_never_collapses_below_the_minimum() -> None:
    lines = render_dashboard(_snapshot(), width=10).splitlines()
    assert {len(line) for line in lines} == {48}


def test_an_overlong_value_is_clipped_rather_than_breaking_the_frame() -> None:
    text = render_dashboard(
        _snapshot(
            system=SystemPanel(
                status="running",
                uptime=dt.timedelta(minutes=1),
                session_date=dt.date(2024, 6, 3),
                session_type=SessionType.REGULAR,
                session_detail="09:15-15:30",
                data_feed=ComponentHealth.HEALTHY,
                data_feed_detail="x" * 500,
                broker_connected=True,
                broker_detail="ok",
                model_version="m" * 500,
            )
        ),
        width=DEFAULT_WIDTH,
    )
    assert {len(line) for line in text.splitlines()} == {DEFAULT_WIDTH}
    assert "..." in text


def test_the_frame_is_pure_ascii() -> None:
    """Readable over ssh, in a Windows console, and in a CI log."""
    text = _rendered()
    assert text.isascii()


def test_the_header_carries_the_snapshot_timestamp() -> None:
    assert "2024-06-03 10:15:00" in _rendered()


# --------------------------------------------------------------------------
# TerminalDashboard
# --------------------------------------------------------------------------


def test_display_writes_the_rendered_frame_to_the_stream() -> None:
    stream = io.StringIO()
    dashboard = TerminalDashboard(stream)
    snapshot = _snapshot()
    dashboard.display(snapshot)
    assert stream.getvalue() == dashboard.render(snapshot) + "\n"


def test_render_is_a_pure_function_of_the_snapshot() -> None:
    snapshot = _snapshot()
    assert render_dashboard(snapshot) == render_dashboard(snapshot)
