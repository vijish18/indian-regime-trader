"""Unit tests for ``risk.circuit_breaker.CircuitBreaker.force_halt`` and
``live/kill_switch.py`` (Phase 22, pre-live checklist item 13): an
operator can immediately stop trading regardless of the portfolio's
current risk state, the halt persists, and the risk engine respects it
the instant it is engaged.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from core.regime.allocation import AllocationRegime
from live.kill_switch import KillSwitch
from portfolio.portfolio_constructor import TargetPortfolio, TargetPosition, empty_portfolio
from risk.circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitState
from risk.portfolio_risk_state import PortfolioRiskState
from risk.risk_manager import RiskManager
from tests.unit._wf_support import risk_config

_AS_OF = dt.date(2024, 6, 3)


def _healthy_risk_state() -> PortfolioRiskState:
    """A portfolio state with nothing wrong -- daily P&L flat, no
    drawdown -- so any halt observed against it can only have come from
    the kill switch, never from evaluate()'s own threshold logic."""
    return PortfolioRiskState(
        as_of=dt.datetime.combine(_AS_OF, dt.time(15, 30), tzinfo=dt.UTC),
        equity=10_000_000.0,
        positions=(),
        daily_pnl_pct=0.0,
        rolling_pnl_pct=0.0,
        peak_to_trough_drawdown_pct=0.0,
        daily_turnover_pct_so_far=0.0,
        max_pairwise_correlation=None,
        correlated_pair=None,
        system_healthy=True,
        system_detail=None,
        broker_connected=True,
        broker_detail=None,
    )


@pytest.fixture
def circuit_breaker(tmp_path: Path) -> CircuitBreaker:
    return CircuitBreaker(risk_config(), tmp_path / "cb.json")


@pytest.fixture
def kill_switch(circuit_breaker: CircuitBreaker) -> KillSwitch:
    return KillSwitch(circuit_breaker)


# --------------------------------------------------------------------------
# CircuitBreaker.force_halt
# --------------------------------------------------------------------------


def test_force_halt_halts_a_perfectly_healthy_portfolio(circuit_breaker: CircuitBreaker) -> None:
    """The whole point: force_halt bypasses evaluate()'s threshold logic
    entirely, so it must halt even when nothing would normally trip."""
    assert circuit_breaker.evaluate(_healthy_risk_state()).state is CircuitState.NORMAL
    status = circuit_breaker.force_halt("ops@example.com", "suspicious fills, stop now")
    assert status.state is CircuitState.HALTED
    assert status.triggered_by == "kill_switch"
    assert status.requires_manual_recovery is True


def test_force_halt_persists(tmp_path: Path) -> None:
    path = tmp_path / "cb.json"
    CircuitBreaker(risk_config(), path).force_halt("ops", "emergency stop")
    reloaded = CircuitBreaker(risk_config(), path)
    assert reloaded.current_status().state is CircuitState.HALTED
    assert reloaded.current_status().triggered_by == "kill_switch"


def test_force_halt_requires_a_non_empty_operator_and_reason(
    circuit_breaker: CircuitBreaker,
) -> None:
    with pytest.raises(CircuitBreakerError, match="operator and reason"):
        circuit_breaker.force_halt("", "reason")
    with pytest.raises(CircuitBreakerError, match="operator and reason"):
        circuit_breaker.force_halt("ops", "")


def test_force_halt_engaged_twice_records_the_second_invocation(
    circuit_breaker: CircuitBreaker,
) -> None:
    circuit_breaker.force_halt("ops-1", "first stop")
    second = circuit_breaker.force_halt("ops-2", "second, more urgent stop")
    assert second.reason is not None
    assert "ops-2" in second.reason
    assert "second, more urgent stop" in second.reason


def test_evaluate_never_auto_clears_a_kill_switch_halt(circuit_breaker: CircuitBreaker) -> None:
    circuit_breaker.force_halt("ops", "stop")
    # A subsequent evaluate() call, even against a perfectly healthy
    # state, must not clear it -- only manual_reset ever does.
    status = circuit_breaker.evaluate(_healthy_risk_state())
    assert status.state is CircuitState.HALTED


def test_manual_reset_clears_a_kill_switch_halt(circuit_breaker: CircuitBreaker) -> None:
    circuit_breaker.force_halt("ops", "stop")
    status = circuit_breaker.manual_reset("ops", "confirmed false alarm")
    assert status.state is CircuitState.NORMAL


# --------------------------------------------------------------------------
# KillSwitch
# --------------------------------------------------------------------------


def test_kill_switch_is_not_engaged_by_default(kill_switch: KillSwitch) -> None:
    assert kill_switch.is_engaged() is False


def test_kill_switch_engage_halts_immediately(kill_switch: KillSwitch) -> None:
    kill_switch.engage("ops@example.com", "manual emergency stop")
    assert kill_switch.is_engaged() is True


def test_kill_switch_release_clears_it(kill_switch: KillSwitch) -> None:
    kill_switch.engage("ops", "stop")
    kill_switch.release("ops", "resolved")
    assert kill_switch.is_engaged() is False


def test_kill_switch_release_requires_it_to_be_engaged(kill_switch: KillSwitch) -> None:
    with pytest.raises(CircuitBreakerError, match="not halted"):
        kill_switch.release("ops", "nothing to release")


# --------------------------------------------------------------------------
# The risk engine respects the kill switch immediately
# --------------------------------------------------------------------------


def test_risk_manager_rejects_every_position_once_the_kill_switch_is_engaged(
    circuit_breaker: CircuitBreaker,
) -> None:
    kill_switch = KillSwitch(circuit_breaker)
    kill_switch.engage("ops", "stop trading now")

    proposed = TargetPortfolio(
        as_of=_AS_OF,
        positions=(
            TargetPosition(
                instrument_id="NSE:INFY",
                symbol="INFY",
                target_weight=0.10,
                sector="IT",
                rank=1,
                score=1.0,
                binding_constraint="unconstrained",
            ),
        ),
        cash_weight=0.90,
        regime=AllocationRegime.NORMAL_RISK,
        gross_exposure=0.10,
    )
    risk_manager = RiskManager(risk_config(), circuit_breaker)
    decisions = risk_manager.evaluate(proposed, _healthy_risk_state())
    assert len(decisions) == 1
    assert decisions[0].approved is False
    assert decisions[0].circuit_state is CircuitState.HALTED


def test_an_empty_target_portfolio_is_unaffected_since_there_is_nothing_to_approve(
    circuit_breaker: CircuitBreaker,
) -> None:
    """required_trades' EXIT actions are not risk-evaluated at all (only
    proposed.positions are) -- documented here so a reader does not
    conclude the kill switch failed to do anything when the proposal is
    already empty."""
    kill_switch = KillSwitch(circuit_breaker)
    kill_switch.engage("ops", "stop")
    risk_manager = RiskManager(risk_config(), circuit_breaker)
    decisions = risk_manager.evaluate(
        empty_portfolio(_AS_OF, AllocationRegime.UNCERTAIN), _healthy_risk_state()
    )
    assert decisions == []
