"""Strategy comparison: per-metric deltas between the HMM and a baseline,
and the caveats every comparison carries so a high Sharpe ratio never
stands alone as a verdict.
"""

from __future__ import annotations

import math

import pytest

from backtest.comparison import compare_all, compare_to_baseline
from backtest.performance import PerformanceReport


def performance_report(**overrides: object) -> PerformanceReport:
    defaults: dict[str, object] = dict(
        cagr=0.10,
        total_return=0.10,
        max_drawdown=0.08,
        drawdown_duration_days=10,
        recovery_duration_days=5,
        volatility=0.15,
        downside_deviation=0.10,
        sharpe=0.60,
        sortino=0.80,
        calmar=1.25,
        turnover=1.5,
        average_holding_period_days=12.0,
        pct_invested=0.70,
        pct_cash=0.30,
        trade_count=120,
        win_rate=0.52,
        profit_factor=1.20,
        gross_pnl=120_000.0,
        total_costs=20_000.0,
        net_pnl=100_000.0,
        gross_return=0.12,
        net_return=0.10,
        cost_pct_of_turnover=0.002,
    )
    defaults.update(overrides)
    return PerformanceReport(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# compare_to_baseline: deltas
# --------------------------------------------------------------------------


def test_delta_is_hmm_minus_baseline_for_every_metric() -> None:
    hmm = performance_report(cagr=0.15, max_drawdown=0.05)
    baseline = performance_report(cagr=0.05, max_drawdown=0.10)
    comparison = compare_to_baseline(hmm, baseline, "buy_and_hold")

    assert comparison.delta_for("cagr").delta == pytest.approx(0.10)
    assert comparison.delta_for("max_drawdown").delta == pytest.approx(-0.05)


def test_lower_is_better_is_flagged_only_for_cost_and_drawdown_metrics() -> None:
    hmm = performance_report()
    baseline = performance_report()
    comparison = compare_to_baseline(hmm, baseline, "buy_and_hold")

    assert comparison.delta_for("max_drawdown").lower_is_better is True
    assert comparison.delta_for("total_costs").lower_is_better is True
    assert comparison.delta_for("cost_pct_of_turnover").lower_is_better is True
    assert comparison.delta_for("cagr").lower_is_better is False
    assert comparison.delta_for("sharpe").lower_is_better is False


def test_delta_for_raises_for_an_unknown_metric() -> None:
    comparison = compare_to_baseline(performance_report(), performance_report(), "x")
    with pytest.raises(KeyError):
        comparison.delta_for("not_a_real_metric")


def test_compare_all_produces_one_comparison_per_non_hmm_strategy() -> None:
    reports = {
        "hmm": performance_report(),
        "buy_and_hold": performance_report(),
        "rolling_volatility": performance_report(),
        "moving_average_trend": performance_report(),
        "shuffled_regime_control": performance_report(),
    }
    comparisons = compare_all(reports, hmm_key="hmm")
    assert set(comparisons) == {
        "buy_and_hold",
        "rolling_volatility",
        "moving_average_trend",
        "shuffled_regime_control",
    }
    for name, comparison in comparisons.items():
        assert comparison.baseline_name == name
        assert comparison.hmm is reports["hmm"]


def test_compare_all_raises_without_the_hmm_key() -> None:
    with pytest.raises(ValueError, match="hmm"):
        compare_all({"buy_and_hold": performance_report()}, hmm_key="hmm")


# --------------------------------------------------------------------------
# Caveats: "do not declare success because Sharpe is high"
# --------------------------------------------------------------------------


def test_every_comparison_carries_the_standing_robustness_caveat() -> None:
    """No comparison, however good it looks, is ever caveat-free."""
    hmm = performance_report(sharpe=3.0, cagr=0.30, max_drawdown=0.02, trade_count=500)
    baseline = performance_report(sharpe=-1.0, cagr=-0.05)
    comparison = compare_to_baseline(hmm, baseline, "buy_and_hold")
    assert comparison.caveats  # never empty
    assert any("robustness diagnostics" in caveat for caveat in comparison.caveats)


def test_low_trade_count_triggers_a_statistical_significance_caveat() -> None:
    hmm = performance_report(trade_count=5)
    baseline = performance_report()
    comparison = compare_to_baseline(hmm, baseline, "buy_and_hold")
    assert any("trade_count" in caveat for caveat in comparison.caveats)


def test_high_trade_count_does_not_trigger_the_low_count_caveat() -> None:
    hmm = performance_report(trade_count=500)
    baseline = performance_report()
    comparison = compare_to_baseline(hmm, baseline, "buy_and_hold")
    assert not any("trade_count is only" in caveat for caveat in comparison.caveats)


def test_high_sharpe_with_large_drawdown_triggers_a_caveat() -> None:
    hmm = performance_report(sharpe=2.5, max_drawdown=0.35)
    baseline = performance_report()
    comparison = compare_to_baseline(hmm, baseline, "buy_and_hold")
    assert any("hides the magnitude of loss" in caveat for caveat in comparison.caveats)


def test_high_sharpe_with_small_drawdown_does_not_trigger_that_caveat() -> None:
    hmm = performance_report(sharpe=2.5, max_drawdown=0.03)
    baseline = performance_report()
    comparison = compare_to_baseline(hmm, baseline, "buy_and_hold")
    assert not any("hides the magnitude of loss" in caveat for caveat in comparison.caveats)


def test_outperformance_with_worse_drawdown_triggers_a_compensation_caveat() -> None:
    hmm = performance_report(total_return=0.20, max_drawdown=0.30)
    baseline = performance_report(total_return=0.05, max_drawdown=0.10)
    comparison = compare_to_baseline(hmm, baseline, "buy_and_hold")
    assert any("not for free" in caveat for caveat in comparison.caveats)


def test_outperformance_with_better_drawdown_does_not_trigger_that_caveat() -> None:
    hmm = performance_report(total_return=0.20, max_drawdown=0.05)
    baseline = performance_report(total_return=0.05, max_drawdown=0.10)
    comparison = compare_to_baseline(hmm, baseline, "buy_and_hold")
    assert not any("not for free" in caveat for caveat in comparison.caveats)


def test_infinite_profit_factor_triggers_a_caveat() -> None:
    hmm = performance_report(profit_factor=math.inf)
    baseline = performance_report()
    comparison = compare_to_baseline(hmm, baseline, "buy_and_hold")
    assert any("infinite" in caveat for caveat in comparison.caveats)


def test_finite_profit_factor_does_not_trigger_that_caveat() -> None:
    hmm = performance_report(profit_factor=1.3)
    baseline = performance_report()
    comparison = compare_to_baseline(hmm, baseline, "buy_and_hold")
    assert not any("infinite" in caveat for caveat in comparison.caveats)


def test_high_cost_share_of_gross_pnl_triggers_a_caveat() -> None:
    hmm = performance_report(gross_pnl=100_000.0, total_costs=60_000.0, net_pnl=40_000.0)
    baseline = performance_report()
    comparison = compare_to_baseline(hmm, baseline, "buy_and_hold")
    assert any("transaction costs consume" in caveat for caveat in comparison.caveats)
