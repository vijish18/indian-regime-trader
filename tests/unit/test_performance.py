"""PerformanceCalculator: CAGR/drawdown/Sharpe-family metrics from a raw
equity curve, gross/net P&L and cost reconciliation, and the
regime-conditional breakdown.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from backtest.performance import PerformanceCalculator


def curve(values: list[float], start: str = "2024-01-02") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def empty_trade_log() -> pd.DataFrame:
    return pd.DataFrame(columns=["cost", "gross_value"])


def trade_log(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Basic return/drawdown metrics
# --------------------------------------------------------------------------


def test_total_return_matches_simple_start_to_end_change() -> None:
    calc = PerformanceCalculator()
    report = calc.compute(curve([100.0, 110.0, 121.0]), empty_trade_log())
    assert report.total_return == pytest.approx(0.21)


def test_flat_curve_has_zero_return_and_zero_drawdown() -> None:
    calc = PerformanceCalculator()
    report = calc.compute(curve([100.0] * 10), empty_trade_log())
    assert report.total_return == pytest.approx(0.0)
    assert report.max_drawdown == pytest.approx(0.0)
    assert report.drawdown_duration_days == 0


def test_max_drawdown_is_reported_as_a_positive_fraction() -> None:
    calc = PerformanceCalculator()
    # Peak at 100, trough at 80 -> a 20% drawdown.
    report = calc.compute(curve([100.0, 90.0, 80.0, 85.0, 95.0]), empty_trade_log())
    assert report.max_drawdown == pytest.approx(0.20)


def test_drawdown_duration_counts_consecutive_sessions_below_the_peak() -> None:
    calc = PerformanceCalculator()
    # Below peak (100) for 3 straight sessions, then a new high.
    report = calc.compute(
        curve([100.0, 95.0, 90.0, 92.0, 101.0]), empty_trade_log()
    )
    assert report.drawdown_duration_days == 3


def test_cagr_is_zero_for_a_single_point_curve() -> None:
    calc = PerformanceCalculator()
    report = calc.compute(curve([100.0]), empty_trade_log())
    assert report.cagr == pytest.approx(0.0)


def test_cagr_annualizes_a_known_daily_growth_rate() -> None:
    calc = PerformanceCalculator()
    n = 253  # 252 trading-day steps -> exactly 1 year at this module's convention
    daily_growth = 1.0007556  # ~ +21%/year over 252 steps
    values = [100.0 * daily_growth**i for i in range(n)]
    report = calc.compute(curve(values), empty_trade_log())
    expected_total = values[-1] / values[0] - 1.0
    assert report.cagr == pytest.approx(expected_total, rel=1e-2)


# --------------------------------------------------------------------------
# Volatility / risk-adjusted metrics
# --------------------------------------------------------------------------


def test_volatility_is_annualized_population_std_of_daily_returns() -> None:
    calc = PerformanceCalculator()
    values = [100.0, 101.0, 99.5, 101.5, 100.0, 102.0]
    report = calc.compute(curve(values), empty_trade_log())

    series = pd.Series(values)
    daily_returns = series.pct_change().dropna()
    expected = float(daily_returns.std(ddof=0) * math.sqrt(252))
    assert report.volatility == pytest.approx(expected)


def test_downside_deviation_ignores_positive_days() -> None:
    calc = PerformanceCalculator()
    # Alternating +5%/-5% days: downside deviation only accumulates the
    # negative-day squared terms (positive days contribute 0), so for this
    # symmetric series it comes out to volatility / sqrt(2), not equal to
    # volatility -- unlike volatility, it is not mean-centered either.
    values = [100.0]
    for i in range(20):
        values.append(values[-1] * (1.05 if i % 2 == 0 else 0.95))
    report = calc.compute(curve(values), empty_trade_log())
    assert report.downside_deviation == pytest.approx(report.volatility / math.sqrt(2), rel=1e-2)
    assert report.downside_deviation < report.volatility


def test_sharpe_is_zero_for_a_zero_volatility_curve() -> None:
    calc = PerformanceCalculator()
    report = calc.compute(curve([100.0] * 10), empty_trade_log())
    assert report.sharpe == 0.0
    assert report.sortino == 0.0


def test_sharpe_is_positive_for_a_steadily_rising_curve() -> None:
    calc = PerformanceCalculator()
    values = [100.0 * 1.001**i for i in range(30)]
    report = calc.compute(curve(values), empty_trade_log())
    assert report.sharpe > 0


def test_calmar_is_cagr_over_max_drawdown() -> None:
    calc = PerformanceCalculator()
    values = [100.0, 110.0, 90.0, 130.0]
    report = calc.compute(curve(values), empty_trade_log())
    assert report.calmar == pytest.approx(report.cagr / report.max_drawdown)


def test_calmar_is_infinite_for_a_rising_curve_with_no_drawdown() -> None:
    calc = PerformanceCalculator()
    values = [100.0 * 1.001**i for i in range(10)]
    report = calc.compute(curve(values), empty_trade_log())
    assert report.max_drawdown == 0.0
    assert report.calmar == math.inf


# --------------------------------------------------------------------------
# Gross P&L / costs / net P&L / turnover / trade_count
# --------------------------------------------------------------------------


def test_net_pnl_is_end_minus_start_equity() -> None:
    calc = PerformanceCalculator()
    report = calc.compute(curve([1_000_000.0, 1_050_000.0]), empty_trade_log())
    assert report.net_pnl == pytest.approx(50_000.0)


def test_gross_pnl_equals_net_pnl_plus_total_costs() -> None:
    calc = PerformanceCalculator()
    log = trade_log(
        [
            {"cost": 100.0, "gross_value": 50_000.0},
            {"cost": 150.0, "gross_value": 60_000.0},
        ]
    )
    report = calc.compute(curve([1_000_000.0, 1_050_000.0]), log)
    assert report.total_costs == pytest.approx(250.0)
    assert report.gross_pnl == pytest.approx(report.net_pnl + report.total_costs)
    assert report.gross_pnl == pytest.approx(50_250.0)


def test_turnover_is_total_traded_value_over_starting_equity() -> None:
    calc = PerformanceCalculator()
    log = trade_log(
        [
            {"cost": 10.0, "gross_value": 200_000.0},
            {"cost": 10.0, "gross_value": 300_000.0},
        ]
    )
    report = calc.compute(curve([1_000_000.0, 1_010_000.0]), log)
    assert report.turnover == pytest.approx(0.5)


def test_trade_count_matches_trade_log_length() -> None:
    calc = PerformanceCalculator()
    log = trade_log([{"cost": 1.0, "gross_value": 1.0} for _ in range(7)])
    report = calc.compute(curve([100.0, 101.0]), log)
    assert report.trade_count == 7


def test_empty_trade_log_produces_zero_costs_turnover_and_trades() -> None:
    calc = PerformanceCalculator()
    report = calc.compute(curve([100.0, 105.0]), empty_trade_log())
    assert report.total_costs == 0.0
    assert report.turnover == 0.0
    assert report.trade_count == 0
    assert report.gross_pnl == pytest.approx(report.net_pnl)
    assert report.cost_pct_of_turnover == 0.0


def test_cost_pct_of_turnover_reuses_the_costs_module_ratio() -> None:
    calc = PerformanceCalculator()
    log = trade_log(
        [
            {"cost": 100.0, "gross_value": 50_000.0},
            {"cost": 150.0, "gross_value": 60_000.0},
        ]
    )
    report = calc.compute(curve([1_000_000.0, 1_050_000.0]), log)
    assert report.cost_pct_of_turnover == pytest.approx(250.0 / 110_000.0)


# --------------------------------------------------------------------------
# Daily win-rate / profit-factor
# --------------------------------------------------------------------------


def test_win_rate_is_the_fraction_of_positive_return_days() -> None:
    calc = PerformanceCalculator()
    # 3 up days, 1 down day, 1 flat day out of 4 return observations.
    values = [100.0, 101.0, 100.5, 102.0, 102.0]
    report = calc.compute(curve(values), empty_trade_log())
    assert report.win_rate == pytest.approx(0.5)  # 2 of 4 daily returns are > 0


def test_profit_factor_is_gains_over_absolute_losses() -> None:
    calc = PerformanceCalculator()
    values = [100.0, 110.0, 99.0, 108.9]  # +10%, -10%, +10%
    report = calc.compute(curve(values), empty_trade_log())
    gains = 0.10 + 0.10
    losses = 0.10
    assert report.profit_factor == pytest.approx(gains / losses, rel=1e-2)


def test_profit_factor_is_infinite_with_no_losing_days() -> None:
    calc = PerformanceCalculator()
    values = [100.0 * 1.01**i for i in range(5)]
    report = calc.compute(curve(values), empty_trade_log())
    assert report.profit_factor == math.inf


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_compute_rejects_an_empty_equity_curve() -> None:
    calc = PerformanceCalculator()
    with pytest.raises(ValueError, match="empty"):
        calc.compute(pd.Series(dtype=float), empty_trade_log())


def test_compute_rejects_nan_in_the_equity_curve() -> None:
    calc = PerformanceCalculator()
    series = curve([100.0, 105.0, 110.0])
    series.iloc[1] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        calc.compute(series, empty_trade_log())


def test_compute_rejects_a_non_positive_equity_value() -> None:
    calc = PerformanceCalculator()
    with pytest.raises(ValueError, match="non-positive"):
        calc.compute(curve([100.0, 0.0, 105.0]), empty_trade_log())


# --------------------------------------------------------------------------
# by_regime
# --------------------------------------------------------------------------


def test_by_regime_attributes_each_session_return_to_its_active_regime() -> None:
    calc = PerformanceCalculator()
    equity = curve([100.0, 105.0, 110.0, 99.0, 108.0])
    regimes = pd.Series(
        ["low_risk", "low_risk", "low_risk", "high_risk", "high_risk"], index=equity.index
    )
    report = calc.by_regime(equity, regimes)

    assert set(report.index) == {"low_risk", "high_risk"}
    assert report.loc["low_risk", "sessions"] == 2  # 2 return observations within low_risk
    assert report.loc["high_risk", "sessions"] == 2


def test_by_regime_cumulative_return_compounds_within_the_regime() -> None:
    calc = PerformanceCalculator()
    equity = curve([100.0, 110.0, 121.0])  # +10%, +10%
    regimes = pd.Series(["low_risk", "low_risk", "low_risk"], index=equity.index)
    report = calc.by_regime(equity, regimes)
    assert report.loc["low_risk", "cumulative_return"] == pytest.approx(0.21)


def test_by_regime_rejects_missing_coverage() -> None:
    calc = PerformanceCalculator()
    equity = curve([100.0, 105.0, 110.0])
    regimes = pd.Series(["low_risk"], index=equity.index[:1])
    with pytest.raises(ValueError, match="no entry"):
        calc.by_regime(equity, regimes)


def test_by_regime_rejects_an_empty_curve() -> None:
    calc = PerformanceCalculator()
    with pytest.raises(ValueError, match="empty"):
        calc.by_regime(pd.Series(dtype=float), pd.Series(dtype=object))
