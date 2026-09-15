"""RobustnessSuite: running a batch of pre-built variant closures and
measuring how much their key metrics disagree with each other -- and that
no single dispersion number is treated as an automatic verdict.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from backtest.performance import PerformanceReport
from backtest.robustness import (
    RobustnessDimension,
    RobustnessError,
    RobustnessSuite,
)


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


def make_factory(report: PerformanceReport) -> Callable[[], PerformanceReport]:
    return lambda: report


# --------------------------------------------------------------------------
# Running a sweep
# --------------------------------------------------------------------------


def test_run_produces_one_variant_result_per_input() -> None:
    suite = RobustnessSuite()
    variants = {
        "60d": make_factory(performance_report(cagr=0.08)),
        "90d": make_factory(performance_report(cagr=0.10)),
        "120d": make_factory(performance_report(cagr=0.12)),
    }
    report = suite.run(RobustnessDimension.TRAINING_WINDOW, variants)

    assert len(report.variants) == 3
    assert {v.variant_label for v in report.variants} == {"60d", "90d", "120d"}
    assert all(v.dimension is RobustnessDimension.TRAINING_WINDOW for v in report.variants)


def test_run_rejects_an_empty_variant_set() -> None:
    suite = RobustnessSuite()
    with pytest.raises(RobustnessError, match="at least one variant"):
        suite.run(RobustnessDimension.TRAINING_WINDOW, {})


def test_run_covers_every_key_metric() -> None:
    suite = RobustnessSuite()
    report = suite.run(
        RobustnessDimension.TRANSACTION_COST,
        {"low": make_factory(performance_report()), "high": make_factory(performance_report())},
    )
    assert set(report.dispersion) == {
        "cagr",
        "total_return",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
    }


# --------------------------------------------------------------------------
# Dispersion statistics
# --------------------------------------------------------------------------


def test_dispersion_reports_min_max_mean_stdev() -> None:
    suite = RobustnessSuite()
    variants = {
        "a": make_factory(performance_report(cagr=0.05)),
        "b": make_factory(performance_report(cagr=0.10)),
        "c": make_factory(performance_report(cagr=0.15)),
    }
    report = suite.run(RobustnessDimension.SLIPPAGE, variants)
    dispersion = report.dispersion["cagr"]

    assert dispersion.minimum == pytest.approx(0.05)
    assert dispersion.maximum == pytest.approx(0.15)
    assert dispersion.mean == pytest.approx(0.10)
    assert dispersion.values == (0.05, 0.10, 0.15)


def test_relative_range_is_zero_for_an_identical_metric_across_variants() -> None:
    suite = RobustnessSuite()
    variants = {
        "a": make_factory(performance_report(cagr=0.10)),
        "b": make_factory(performance_report(cagr=0.10)),
    }
    report = suite.run(RobustnessDimension.UNIVERSE_SIZE, variants)
    assert report.dispersion["cagr"].relative_range == pytest.approx(0.0)


def test_relative_range_grows_with_dispersion_relative_to_the_mean() -> None:
    suite = RobustnessSuite()
    tight = suite.run(
        RobustnessDimension.MARKET_PERIOD,
        {
            "a": make_factory(performance_report(cagr=0.099)),
            "b": make_factory(performance_report(cagr=0.101)),
        },
    )
    wide = suite.run(
        RobustnessDimension.MARKET_PERIOD,
        {
            "a": make_factory(performance_report(cagr=0.01)),
            "b": make_factory(performance_report(cagr=0.20)),
        },
    )
    assert tight.dispersion["cagr"].relative_range < wide.dispersion["cagr"].relative_range


def test_relative_range_is_zero_when_mean_and_spread_are_both_near_zero() -> None:
    suite = RobustnessSuite()
    report = suite.run(
        RobustnessDimension.PARAMETER_PERTURBATION,
        {
            "a": make_factory(performance_report(cagr=1e-13)),
            "b": make_factory(performance_report(cagr=-1e-13)),
        },
    )
    assert report.dispersion["cagr"].relative_range == pytest.approx(0.0)


def test_relative_range_is_infinite_when_mean_is_zero_but_values_differ() -> None:
    suite = RobustnessSuite()
    report = suite.run(
        RobustnessDimension.PARAMETER_PERTURBATION,
        {
            "a": make_factory(performance_report(cagr=-0.10)),
            "b": make_factory(performance_report(cagr=0.10)),
        },
    )
    assert report.dispersion["cagr"].relative_range == math.inf


def test_non_finite_values_are_excluded_from_dispersion_but_kept_in_values() -> None:
    suite = RobustnessSuite()
    report = suite.run(
        RobustnessDimension.REBALANCE_THRESHOLD,
        {
            "a": make_factory(performance_report(calmar=math.inf)),
            "b": make_factory(performance_report(calmar=1.5)),
            "c": make_factory(performance_report(calmar=2.5)),
        },
    )
    dispersion = report.dispersion["calmar"]
    assert dispersion.values == (math.inf, 1.5, 2.5)
    assert dispersion.mean == pytest.approx(2.0)  # only the finite values


# --------------------------------------------------------------------------
# is_stable: an explicit, opt-in check, never an automatic verdict
# --------------------------------------------------------------------------


def test_is_stable_true_within_the_given_tolerance() -> None:
    suite = RobustnessSuite()
    variants = {
        "a": make_factory(performance_report(cagr=0.095)),
        "b": make_factory(performance_report(cagr=0.105)),
    }
    report = suite.run(RobustnessDimension.TRAINING_WINDOW, variants)
    assert report.is_stable("cagr", max_relative_range=0.5) is True


def test_is_stable_false_outside_the_given_tolerance() -> None:
    suite = RobustnessSuite()
    variants = {
        "a": make_factory(performance_report(cagr=0.02)),
        "b": make_factory(performance_report(cagr=0.20)),
    }
    report = suite.run(RobustnessDimension.TRAINING_WINDOW, variants)
    assert report.is_stable("cagr", max_relative_range=0.5) is False


def test_is_stable_raises_for_an_unknown_metric() -> None:
    suite = RobustnessSuite()
    report = suite.run(
        RobustnessDimension.TRAINING_WINDOW,
        {"a": make_factory(performance_report())},
    )
    with pytest.raises(KeyError):
        report.is_stable("not_a_real_metric")
