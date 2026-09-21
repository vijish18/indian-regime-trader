"""Circuit breaker: NORMAL / REDUCED_RISK / HALTED transitions, persistence
across a simulated restart, manual-recovery-only escape from HALTED, and
that every transition is logged.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pytest

from config.models import RiskConfig
from risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerStatus,
    CircuitState,
)
from risk.portfolio_risk_state import PortfolioRiskState

AS_OF = dt.datetime(2023, 6, 1, 10, 0, tzinfo=dt.UTC)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def risk_config(**overrides: object) -> RiskConfig:
    defaults: dict[str, object] = {
        "max_gross_exposure": 0.75,
        "max_leverage": 1.0,
        "max_single_name_pct": 0.15,
        "max_sector_pct": 0.30,
        "max_concurrent_positions": 5,
        "max_risk_per_position_pct": 0.01,
        "daily_loss_warning_pct": 0.01,
        "daily_loss_reduce_pct": 0.02,
        "daily_loss_halt_pct": 0.03,
        "rolling_loss_reduce_pct": 0.04,
        "rolling_loss_halt_pct": 0.06,
        "peak_to_trough_drawdown_halt_pct": 0.10,
        "stale_data_max_minutes": 30,
        "max_pairwise_correlation": 0.85,
        "max_adv_participation_pct": 0.08,
        "max_spread_bps": 100.0,
        "max_daily_turnover_pct": 0.50,
        "reduced_risk_exposure_multiplier": 0.50,
        # Per-position stops are OFF in this fixture so every expectation
        # below reads as it did before risk/stop_loss.py existed -- a stop
        # firing mid-run would change trade counts and returns for reasons
        # that have nothing to do with what these tests assert. Tests that
        # want the stops enable them explicitly.
        "stop_loss": {
            "enabled": False,
            "hard_stop_pct": 0.03,
            "trail_drop_pct": 0.02,
            "trail_arm_net_profit_pct": 0.03,
            "close_on_arm": True,
        },
    }
    defaults.update(overrides)
    return RiskConfig.model_validate(defaults)


def healthy_state(**overrides: object) -> PortfolioRiskState:
    defaults: dict[str, object] = dict(
        as_of=AS_OF,
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
    defaults.update(overrides)
    return PortfolioRiskState(**defaults)  # type: ignore[arg-type]


def breaker(tmp_path: Path, config: RiskConfig | None = None) -> CircuitBreaker:
    return CircuitBreaker(config or risk_config(), tmp_path / "circuit_breaker_state.json")


# --------------------------------------------------------------------------
# Basic transitions
# --------------------------------------------------------------------------


def test_normal_when_nothing_is_triggered(tmp_path: Path) -> None:
    status = breaker(tmp_path).evaluate(healthy_state())
    assert status.state is CircuitState.NORMAL
    assert status.requires_manual_recovery is False
    assert status.halted_at is None


def test_warning_threshold_leaves_state_normal(tmp_path: Path) -> None:
    cfg = risk_config()
    status = breaker(tmp_path, cfg).evaluate(
        healthy_state(daily_pnl_pct=cfg.daily_loss_warning_pct)
    )
    assert status.state is CircuitState.NORMAL
    assert status.triggered_by == "daily_loss_warning"


@pytest.mark.parametrize(
    "field", ["daily_pnl_pct", "rolling_pnl_pct"]
)
def test_reduce_threshold_triggers_reduced_risk(tmp_path: Path, field: str) -> None:
    cfg = risk_config()
    limit = cfg.daily_loss_reduce_pct if field == "daily_pnl_pct" else cfg.rolling_loss_reduce_pct
    status = breaker(tmp_path, cfg).evaluate(healthy_state(**{field: limit}))
    assert status.state is CircuitState.REDUCED_RISK
    assert status.requires_manual_recovery is False
    assert status.halted_at is None


@pytest.mark.parametrize(
    ("field", "limit_attr"),
    [
        ("daily_pnl_pct", "daily_loss_halt_pct"),
        ("rolling_pnl_pct", "rolling_loss_halt_pct"),
        ("peak_to_trough_drawdown_pct", "peak_to_trough_drawdown_halt_pct"),
    ],
)
def test_halt_threshold_triggers_halted(tmp_path: Path, field: str, limit_attr: str) -> None:
    cfg = risk_config()
    limit = getattr(cfg, limit_attr)
    status = breaker(tmp_path, cfg).evaluate(healthy_state(**{field: limit}))
    assert status.state is CircuitState.HALTED
    assert status.requires_manual_recovery is True
    assert status.halted_at is not None


def test_broker_disconnected_forces_halt_regardless_of_pnl(tmp_path: Path) -> None:
    status = breaker(tmp_path).evaluate(
        healthy_state(broker_connected=False, broker_detail="timeout")
    )
    assert status.state is CircuitState.HALTED
    assert status.triggered_by == "broker_connectivity"
    assert "timeout" in (status.reason or "")


def test_system_unhealthy_forces_halt(tmp_path: Path) -> None:
    status = breaker(tmp_path).evaluate(
        healthy_state(system_healthy=False, system_detail="disk full")
    )
    assert status.state is CircuitState.HALTED
    assert status.triggered_by == "system_health"
    assert "disk full" in (status.reason or "")


def test_connectivity_and_health_checked_before_pnl_tiers(tmp_path: Path) -> None:
    """A broker outage during an otherwise-fine day must still halt --
    operational failure takes precedence over financial thresholds."""
    cfg = risk_config()
    status = breaker(tmp_path, cfg).evaluate(
        healthy_state(broker_connected=False, daily_pnl_pct=0.0)
    )
    assert status.state is CircuitState.HALTED
    assert status.triggered_by == "broker_connectivity"


# --------------------------------------------------------------------------
# HALTED is sticky: no automatic recovery, survives a restart
# --------------------------------------------------------------------------


def test_halt_does_not_auto_recover_on_next_evaluate(tmp_path: Path) -> None:
    cfg = risk_config()
    cb = breaker(tmp_path, cfg)
    cb.evaluate(healthy_state(daily_pnl_pct=cfg.daily_loss_halt_pct))
    # The world is healthy again, but the breaker must stay halted.
    status = cb.evaluate(healthy_state())
    assert status.state is CircuitState.HALTED


def test_halt_persists_across_a_new_circuit_breaker_instance(tmp_path: Path) -> None:
    """Simulates an application restart: a brand-new CircuitBreaker object
    pointed at the same state file must see the halt, not start fresh."""
    cfg = risk_config()
    state_path = tmp_path / "circuit_breaker_state.json"
    first = CircuitBreaker(cfg, state_path)
    first.evaluate(healthy_state(broker_connected=False))
    assert first.current_status().state is CircuitState.HALTED

    restarted = CircuitBreaker(cfg, state_path)
    assert restarted.current_status().state is CircuitState.HALTED
    # Even a fully healthy state passed to the restarted process must not
    # clear the halt automatically.
    assert restarted.evaluate(healthy_state()).state is CircuitState.HALTED


def test_reduced_risk_recovers_to_normal_automatically(tmp_path: Path) -> None:
    """REDUCED_RISK is not "critical" -- it clears on its own once the
    triggering metric recovers, unlike HALTED."""
    cfg = risk_config()
    cb = breaker(tmp_path, cfg)
    cb.evaluate(healthy_state(daily_pnl_pct=cfg.daily_loss_reduce_pct))
    status = cb.evaluate(healthy_state())
    assert status.state is CircuitState.NORMAL


# --------------------------------------------------------------------------
# Manual recovery
# --------------------------------------------------------------------------


def test_manual_reset_clears_a_halted_breaker(tmp_path: Path) -> None:
    cfg = risk_config()
    cb = breaker(tmp_path, cfg)
    cb.evaluate(healthy_state(system_healthy=False))
    assert cb.current_status().state is CircuitState.HALTED

    reset_status = cb.manual_reset(operator="ops-oncall", reason="confirmed false alarm")
    assert reset_status.state is CircuitState.NORMAL
    assert reset_status.requires_manual_recovery is False
    assert cb.current_status().state is CircuitState.NORMAL


def test_manual_reset_raises_when_not_halted(tmp_path: Path) -> None:
    cb = breaker(tmp_path)
    cb.evaluate(healthy_state())
    with pytest.raises(CircuitBreakerError):
        cb.manual_reset(operator="ops-oncall", reason="nothing to reset")


def test_manual_reset_requires_operator_and_reason(tmp_path: Path) -> None:
    cb = breaker(tmp_path)
    cb.evaluate(healthy_state(broker_connected=False))
    with pytest.raises(CircuitBreakerError):
        cb.manual_reset(operator="", reason="missing operator")
    with pytest.raises(CircuitBreakerError):
        cb.manual_reset(operator="ops-oncall", reason="")


def test_evaluate_never_performs_the_halted_to_normal_transition(tmp_path: Path) -> None:
    """No code path inside evaluate() may leave HALTED -- only manual_reset
    can. Exercised across many evaluate() calls with fully healthy input."""
    cfg = risk_config()
    cb = breaker(tmp_path, cfg)
    cb.evaluate(healthy_state(daily_pnl_pct=cfg.daily_loss_halt_pct))
    for _ in range(20):
        assert cb.evaluate(healthy_state()).state is CircuitState.HALTED


# --------------------------------------------------------------------------
# Logging: every circuit-breaker event must be logged
# --------------------------------------------------------------------------


def test_state_transition_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cfg = risk_config()
    with caplog.at_level(logging.WARNING, logger="risk.circuit_breaker"):
        breaker(tmp_path, cfg).evaluate(healthy_state(daily_pnl_pct=cfg.daily_loss_reduce_pct))
    events = [
        fields
        for record in caplog.records
        if (fields := getattr(record, "extra_fields", None))
    ]
    assert any(event["event"] == "circuit_breaker_transition" for event in events)


def test_halt_transition_is_logged_at_critical(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="risk.circuit_breaker"):
        breaker(tmp_path).evaluate(healthy_state(system_healthy=False))
    critical_records = [record for record in caplog.records if record.levelno == logging.CRITICAL]
    assert len(critical_records) == 1


def test_manual_reset_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cb = breaker(tmp_path)
    cb.evaluate(healthy_state(broker_connected=False))
    with caplog.at_level(logging.WARNING, logger="risk.circuit_breaker"):
        cb.manual_reset(operator="ops-oncall", reason="verified safe")
    events = [
        fields
        for record in caplog.records
        if (fields := getattr(record, "extra_fields", None))
    ]
    assert any(event["event"] == "circuit_breaker_manual_reset" for event in events)


def test_warning_without_state_change_is_still_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = risk_config()
    with caplog.at_level(logging.WARNING, logger="risk.circuit_breaker"):
        breaker(tmp_path, cfg).evaluate(healthy_state(daily_pnl_pct=cfg.daily_loss_warning_pct))
    assert any(
        getattr(record, "extra_fields", {}).get("event") == "risk_warning"
        for record in caplog.records
    )


# --------------------------------------------------------------------------
# CircuitBreakerStatus's own invariants
# --------------------------------------------------------------------------


def test_status_post_init_rejects_halted_without_halted_at() -> None:
    with pytest.raises(ValueError, match="halted_at"):
        CircuitBreakerStatus(
            state=CircuitState.HALTED,
            as_of=AS_OF,
            reason="x",
            triggered_by="x",
            requires_manual_recovery=True,
            halted_at=None,
        )


def test_status_post_init_rejects_halted_without_manual_recovery_flag() -> None:
    with pytest.raises(ValueError, match="manual recovery"):
        CircuitBreakerStatus(
            state=CircuitState.HALTED,
            as_of=AS_OF,
            reason="x",
            triggered_by="x",
            requires_manual_recovery=False,
            halted_at=AS_OF,
        )


def test_status_post_init_rejects_non_halted_with_halted_at() -> None:
    with pytest.raises(ValueError, match="halted_at"):
        CircuitBreakerStatus(
            state=CircuitState.NORMAL,
            as_of=AS_OF,
            reason=None,
            triggered_by=None,
            requires_manual_recovery=False,
            halted_at=AS_OF,
        )


def test_status_post_init_rejects_non_halted_requiring_manual_recovery() -> None:
    with pytest.raises(ValueError, match="only HALTED"):
        CircuitBreakerStatus(
            state=CircuitState.NORMAL,
            as_of=AS_OF,
            reason=None,
            triggered_by=None,
            requires_manual_recovery=True,
            halted_at=None,
        )


def test_status_round_trips_through_json(tmp_path: Path) -> None:
    cb = breaker(tmp_path)
    written = cb.evaluate(healthy_state(broker_connected=False))
    loaded = CircuitBreakerStatus.from_dict(written.to_dict())
    assert loaded == written


# --------------------------------------------------------------------------
# Invariant / property tests
# --------------------------------------------------------------------------


def test_invariant_halted_always_requires_manual_recovery(tmp_path: Path) -> None:
    """Across a randomized sweep of daily/rolling/peak metrics and health
    flags, any HALTED status this breaker ever produces requires manual
    recovery and records when it happened."""
    rng = np.random.default_rng(7)
    cfg = risk_config()
    for i in range(40):
        state = healthy_state(
            as_of=AS_OF + dt.timedelta(days=i),
            daily_pnl_pct=float(rng.uniform(0, 0.10)),
            rolling_pnl_pct=float(rng.uniform(0, 0.10)),
            peak_to_trough_drawdown_pct=float(rng.uniform(0, 0.20)),
            broker_connected=bool(rng.random() > 0.1),
            system_healthy=bool(rng.random() > 0.1),
        )
        cb = CircuitBreaker(cfg, tmp_path / f"state_{i}.json")
        status = cb.evaluate(state)
        if status.state is CircuitState.HALTED:
            assert status.requires_manual_recovery is True
            assert status.halted_at is not None
        else:
            assert status.requires_manual_recovery is False
            assert status.halted_at is None


def test_invariant_once_halted_stays_halted_until_manual_reset(tmp_path: Path) -> None:
    """Across a randomized sweep of *subsequent* healthy-looking states fed
    to an already-halted breaker, the state never changes on its own."""
    rng = np.random.default_rng(11)
    cfg = risk_config()
    cb = breaker(tmp_path, cfg)
    cb.evaluate(healthy_state(daily_pnl_pct=cfg.daily_loss_halt_pct))
    for i in range(30):
        state = healthy_state(
            as_of=AS_OF + dt.timedelta(days=i),
            daily_pnl_pct=float(rng.uniform(0, 0.005)),
            rolling_pnl_pct=float(rng.uniform(0, 0.005)),
        )
        assert cb.evaluate(state).state is CircuitState.HALTED
    cb.manual_reset(operator="ops", reason="sweep complete")
    assert cb.current_status().state is CircuitState.NORMAL
