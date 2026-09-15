"""Unit tests for ``risk/risk_state_builder.py`` (Phase 19): the live
equivalent of ``BacktestEngine._build_risk_state`` -- an explicit,
append-only equity curve plus the market/liquidity facts a
``PortfolioRiskState`` needs.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core.regime.allocation import AllocationRegime
from portfolio.portfolio_constructor import TargetPortfolio, TargetPosition, empty_portfolio
from risk.risk_state_builder import EquityHistory, build_risk_state, max_pairwise_correlation
from tests.unit._wf_support import Environment

_NOW = dt.datetime(2015, 6, 1, 10, 0, tzinfo=dt.UTC)


# --------------------------------------------------------------------------
# EquityHistory
# --------------------------------------------------------------------------


def test_empty_history_reports_no_losses() -> None:
    history = EquityHistory()
    assert history.daily_pnl_pct() == 0.0
    assert history.rolling_pnl_pct(5) == 0.0
    assert history.peak_to_trough_drawdown_pct() == 0.0


def test_single_point_history_reports_no_losses() -> None:
    history = EquityHistory([1_000_000.0])
    assert history.daily_pnl_pct() == 0.0
    assert history.peak_to_trough_drawdown_pct() == 0.0


def test_daily_loss_is_a_positive_fraction() -> None:
    history = EquityHistory([1_000_000.0, 950_000.0])
    assert history.daily_pnl_pct() == pytest.approx(0.05)


def test_a_gain_reports_zero_loss_not_a_negative_one() -> None:
    history = EquityHistory([1_000_000.0, 1_100_000.0])
    assert history.daily_pnl_pct() == 0.0


def test_rolling_loss_uses_the_window_peak() -> None:
    history = EquityHistory([1_000_000.0, 1_200_000.0, 900_000.0])
    assert history.rolling_pnl_pct(3) == pytest.approx(0.25)


def test_rolling_window_ignores_points_outside_it() -> None:
    history = EquityHistory([2_000_000.0, 1_000_000.0, 1_000_000.0, 950_000.0])
    # Window of 3 excludes the 2,000,000 peak entirely.
    assert history.rolling_pnl_pct(3) == pytest.approx(0.05)


def test_peak_to_trough_uses_the_all_time_peak() -> None:
    history = EquityHistory([2_000_000.0, 1_000_000.0, 1_500_000.0])
    assert history.peak_to_trough_drawdown_pct() == pytest.approx(0.25)


def test_append_extends_the_curve() -> None:
    history = EquityHistory()
    history.append(1_000_000.0)
    history.append(900_000.0)
    assert history.values == [1_000_000.0, 900_000.0]
    assert history.daily_pnl_pct() == pytest.approx(0.10)


# --------------------------------------------------------------------------
# build_risk_state
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env() -> Environment:
    return Environment(n_days=120)


def _portfolio(env: Environment, as_of: dt.date) -> TargetPortfolio:
    positions = tuple(
        TargetPosition(
            instrument_id=instrument_id,
            symbol=instrument_id,
            target_weight=0.20,
            sector="IT",
            rank=index + 1,
            score=1.0,
            binding_constraint="unconstrained",
        )
        for index, instrument_id in enumerate(env.instrument_ids[:2])
    )
    gross = sum(p.target_weight for p in positions)
    return TargetPortfolio(
        as_of=as_of,
        positions=positions,
        cash_weight=1.0 - gross,
        regime=AllocationRegime.NORMAL_RISK,
        gross_exposure=gross,
    )


def test_build_risk_state_populates_position_risks(env: Environment) -> None:
    as_of = env.dates[-1]
    portfolio = _portfolio(env, as_of)
    state = build_risk_state(
        portfolio,
        10_000_000.0,
        dt.datetime.combine(as_of, dt.time(15, 30), tzinfo=dt.UTC),
        env.market_data,
        EquityHistory([10_000_000.0]),
    )
    assert len(state.positions) == 2
    assert {p.instrument_id for p in state.positions} == set(env.instrument_ids[:2])
    assert all(p.avg_daily_value_inr > 0 for p in state.positions)
    assert all(p.sector == "IT" for p in state.positions)


def test_build_risk_state_carries_drawdowns_from_the_equity_history(env: Environment) -> None:
    as_of = env.dates[-1]
    state = build_risk_state(
        empty_portfolio(as_of, AllocationRegime.UNCERTAIN),
        900_000.0,
        dt.datetime.combine(as_of, dt.time(15, 30), tzinfo=dt.UTC),
        env.market_data,
        EquityHistory([1_000_000.0, 900_000.0]),
    )
    assert state.daily_pnl_pct == pytest.approx(0.10)
    assert state.peak_to_trough_drawdown_pct == pytest.approx(0.10)


def test_build_risk_state_defaults_report_a_healthy_connected_system(env: Environment) -> None:
    as_of = env.dates[-1]
    state = build_risk_state(
        empty_portfolio(as_of, AllocationRegime.UNCERTAIN),
        1_000_000.0,
        dt.datetime.combine(as_of, dt.time(15, 30), tzinfo=dt.UTC),
        env.market_data,
        EquityHistory([1_000_000.0]),
    )
    assert state.system_healthy is True
    assert state.broker_connected is True


def test_build_risk_state_propagates_an_unhealthy_system(env: Environment) -> None:
    as_of = env.dates[-1]
    state = build_risk_state(
        empty_portfolio(as_of, AllocationRegime.UNCERTAIN),
        1_000_000.0,
        dt.datetime.combine(as_of, dt.time(15, 30), tzinfo=dt.UTC),
        env.market_data,
        EquityHistory([1_000_000.0]),
        system_healthy=False,
        system_detail="data feed down",
        broker_connected=False,
        broker_detail="socket closed",
    )
    assert state.system_healthy is False
    assert state.system_detail == "data feed down"
    assert state.broker_connected is False


def test_empty_portfolio_produces_no_position_risks_and_no_correlation(env: Environment) -> None:
    as_of = env.dates[-1]
    state = build_risk_state(
        empty_portfolio(as_of, AllocationRegime.UNCERTAIN),
        1_000_000.0,
        dt.datetime.combine(as_of, dt.time(15, 30), tzinfo=dt.UTC),
        env.market_data,
        EquityHistory([1_000_000.0]),
    )
    assert state.positions == ()
    assert state.max_pairwise_correlation is None
    assert state.correlated_pair is None


# --------------------------------------------------------------------------
# max_pairwise_correlation
# --------------------------------------------------------------------------


def test_correlation_needs_at_least_two_instruments(env: Environment) -> None:
    assert max_pairwise_correlation(env.market_data, [env.instrument_ids[0]], env.dates[-1]) is None
    assert max_pairwise_correlation(env.market_data, [], env.dates[-1]) is None


def test_correlation_is_computed_for_two_instruments_with_history(env: Environment) -> None:
    result = max_pairwise_correlation(
        env.market_data, list(env.instrument_ids[:2]), env.dates[-1], min_observations=10
    )
    assert result is not None
    value, pair = result
    assert -1.0 <= value <= 1.0
    assert set(pair) == set(env.instrument_ids[:2])


def test_correlation_returns_none_when_history_is_too_short(env: Environment) -> None:
    result = max_pairwise_correlation(
        env.market_data,
        list(env.instrument_ids[:2]),
        env.dates[-1],
        lookback_days=5,
        min_observations=1000,
    )
    assert result is None


def test_correlation_skips_instruments_with_no_data(env: Environment) -> None:
    result = max_pairwise_correlation(
        env.market_data,
        [env.instrument_ids[0], "NSE:DOES-NOT-EXIST"],
        env.dates[-1],
        min_observations=10,
    )
    assert result is None
