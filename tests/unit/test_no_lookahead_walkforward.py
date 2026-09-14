"""Look-ahead and leakage tests for the walk-forward pipeline itself.

Every unit-level causality property (features, the scaler, the HMM's
forward filter, stock selection) already has its own dedicated tests in
earlier phases. What is new and uniquely at risk here is *composition*:
does wiring all of those pieces into a fit -> freeze -> OOS -> advance ->
retrain loop, plus a day-by-day execution engine, accidentally reintroduce
a leak that none of the individual pieces has on its own?

The technique throughout is **truncation invariance**: build two
environments from the *same seed*, differing only in how many days of
data exist. ``numpy.random.Generator`` draws are a deterministic stream,
so the first N observations of a size-M draw (M > N) are bit-identical to
a size-N draw with the same seed (asserted directly below, once, as the
premise the rest of this file depends on). That means "does the answer
for day T change if more future data exists" can be tested by literally
comparing two runs against differently-truncated data, rather than trying
to construct an adversarial mutation -- either failing to find one, or
constructing something unrealistic.

A second, complementary technique is **direct mutation**: overwrite only
the last few days of an otherwise-identical dataset with something
wildly different (a crash), and confirm nothing computed for the earlier,
untouched days changed. This catches a bug the truncation test alone
would not: a leak that only manifests when future data has a *different
value*, not merely when it exists at all (an unlikely class of bug here,
but cheap to rule out directly).
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from core.regime.allocation import AllocationRegime, AllocationTarget
from data.models import DailyBar, IndexObservation
from tests.unit._wf_support import Environment

# --------------------------------------------------------------------------
# The premise this whole file depends on
# --------------------------------------------------------------------------


def test_premise_numpy_rng_prefix_is_stable_across_sample_sizes() -> None:
    longer = np.random.default_rng(5).normal(0, 1, 150)
    shorter = np.random.default_rng(5).normal(0, 1, 100)
    assert np.array_equal(longer[:100], shorter)


def test_premise_two_environments_share_an_identical_data_prefix() -> None:
    short_env = Environment(n_days=100, n_stocks=4, seed=5)
    long_env = Environment(n_days=150, n_stocks=4, seed=5)
    assert short_env.dates == long_env.dates[:100]
    for instrument_id in short_env.instrument_ids:
        short_bars = short_env.market_data.get_equity_bars(
            instrument_id, short_env.dates[0], short_env.dates[-1]
        )
        long_bars = long_env.market_data.get_equity_bars(
            instrument_id, short_env.dates[0], short_env.dates[-1]
        )
        assert short_bars == long_bars


# --------------------------------------------------------------------------
# HMM fit: never affected by data past the training window
# --------------------------------------------------------------------------


def test_hmm_fit_is_identical_regardless_of_how_much_future_data_exists(
    tmp_path: Path,
) -> None:
    short_env = Environment(n_days=100, n_stocks=4, seed=5)
    long_env = Environment(n_days=150, n_stocks=4, seed=5)

    train_start = short_env.dates[0]
    train_end = short_env.dates[79]

    short_validator = short_env.validator(tmp_path / "short")
    long_validator = long_env.validator(tmp_path / "long")

    short_model, short_params, _engine1, short_model_id = short_validator._fit_fold(
        train_start, train_end
    )
    long_model, long_params, _engine2, long_model_id = long_validator._fit_fold(
        train_start, train_end
    )

    assert short_model_id == long_model_id
    assert short_model.training_result.bic == long_model.training_result.bic
    assert short_model.training_result.n_observations == long_model.training_result.n_observations
    np.testing.assert_array_equal(
        short_model.parameters.means, long_model.parameters.means
    )
    np.testing.assert_array_equal(
        short_model.parameters.transition_matrix, long_model.parameters.transition_matrix
    )
    assert short_params.means == long_params.means
    assert short_params.stds == long_params.stds


# --------------------------------------------------------------------------
# Regime/exposure targets for OOS days: unaffected by data past that day
# --------------------------------------------------------------------------


def test_hmm_exposure_targets_for_oos_days_are_unaffected_by_later_data(
    tmp_path: Path,
) -> None:
    short_env = Environment(n_days=100, n_stocks=4, seed=5)
    long_env = Environment(n_days=150, n_stocks=4, seed=5)

    train_start = short_env.dates[0]
    train_end = short_env.dates[79]
    test_start = short_env.dates[80]
    test_end = short_env.dates[99]

    short_validator = short_env.validator(tmp_path / "short")
    long_validator = long_env.validator(tmp_path / "long")

    short_model, short_params, short_engine, _ = short_validator._fit_fold(train_start, train_end)
    long_model, long_params, long_engine, _ = long_validator._fit_fold(train_start, train_end)

    short_targets, _short_states = short_validator._hmm_exposure_targets(
        short_model, short_params, short_engine, test_start, test_end
    )
    long_targets, _long_states = long_validator._hmm_exposure_targets(
        long_model, long_params, long_engine, test_start, test_end
    )

    assert set(short_targets) == set(long_targets) == set(
        short_env.calendar.trading_days_between(test_start, test_end)
    )
    for day in short_targets:
        assert short_targets[day].regime == long_targets[day].regime
        assert short_targets[day].target_gross_exposure == pytest.approx(
            long_targets[day].target_gross_exposure
        )
        assert short_targets[day].confidence == pytest.approx(long_targets[day].confidence)
        assert short_targets[day].expected_volatility == pytest.approx(
            long_targets[day].expected_volatility
        )


# --------------------------------------------------------------------------
# BacktestEngine: a session's decision is unaffected by data past its own
# execution date
# --------------------------------------------------------------------------


def test_backtest_engine_orders_are_identical_regardless_of_future_data(
    tmp_path: Path,
) -> None:
    """Runs the identical signal_dates/exposure_targets through two engines
    built from environments that agree on every date up to and including
    the last execution date, and diverge (by simply not existing) after
    that. The orders -- which encode the decision, not the fill -- must be
    bit-identical.
    """
    short_env = Environment(n_days=100, n_stocks=4, seed=5)
    long_env = Environment(n_days=150, n_stocks=4, seed=5)

    signal_dates = short_env.dates[80:95]  # last execution date is well within day 100
    band = short_env.regime_policy.band_for(AllocationRegime.LOW_RISK)

    exposure_targets = {
        day: AllocationTarget(
            as_of=day,
            regime=AllocationRegime.LOW_RISK,
            target_gross_exposure=band.max_gross_exposure,
            min_gross_exposure=band.max_gross_exposure,
            max_gross_exposure=band.max_gross_exposure,
            allow_new_positions=True,
            confidence=1.0,
            expected_volatility=0.1,
            reason="test",
        )
        for day in signal_dates
    }

    short_engine = short_env.engine(tmp_path / "short")
    long_engine = long_env.engine(tmp_path / "long")

    short_result = short_engine.run("t", exposure_targets, signal_dates, 1_000_000.0)
    long_result = long_engine.run("t", exposure_targets, signal_dates, 1_000_000.0)

    assert len(short_result.orders) == len(long_result.orders)
    for short_order, long_order in zip(short_result.orders, long_result.orders, strict=True):
        assert short_order == long_order

    assert len(short_result.fills) == len(long_result.fills)
    for short_fill, long_fill in zip(short_result.fills, long_result.fills, strict=True):
        assert short_fill.fill_price == pytest.approx(long_fill.fill_price)
        assert short_fill.quantity == long_fill.quantity

    pd_short = short_result.equity_curve
    pd_long = long_result.equity_curve
    assert list(pd_short.index) == list(pd_long.index)
    for value_short, value_long in zip(pd_short.to_numpy(), pd_long.to_numpy(), strict=True):
        assert value_short == pytest.approx(value_long)


# --------------------------------------------------------------------------
# WalkForwardValidator.run(): folds inside the shared range are identical
# regardless of a longer requested overall end date
# --------------------------------------------------------------------------


def test_walk_forward_run_folds_are_identical_regardless_of_a_later_requested_end(
    tmp_path: Path,
) -> None:
    """``short_env`` still has materially less data than ``long_env``
    (125 vs 150 days), but enough margin past its own last fold's test
    window for the engine's next-open "resting order" search
    (``BacktestEngine.max_fill_search_days``) to always find a fill
    without spilling past ``short_env``'s own last date -- otherwise a fill
    genuinely could (correctly) differ between the two near that edge,
    which is a data-availability effect, not a look-ahead bug. See
    ``test_backtest_engine_orders_are_identical_regardless_of_future_data``
    for the version of this property that isolates *decisions* from that
    edge effect entirely.
    """
    short_env = Environment(n_days=125, n_stocks=4, seed=5)
    long_env = Environment(n_days=150, n_stocks=4, seed=5)

    short_validator = short_env.validator(tmp_path / "short")
    long_validator = long_env.validator(tmp_path / "long")

    short_folds = short_validator.run(short_env.dates[0], short_env.dates[-1])
    long_folds = long_validator.run(long_env.dates[0], long_env.dates[-1])

    # Exclude any fold whose test_end sits within the engine's fill-search
    # margin of short_env's own last date -- there, a fill can legitimately
    # (and correctly) differ simply because short_env has no more data to
    # search forward into, a data-availability effect rather than a leak.
    margin_start = short_env.dates[-11]
    comparable_short_folds = [fold for fold in short_folds if fold.test_end < margin_start]
    assert comparable_short_folds, "test setup should produce at least one comparable fold"
    for short_fold in comparable_short_folds:
        matching = [
            fold
            for fold in long_folds
            if fold.train_start == short_fold.train_start
            and fold.train_end == short_fold.train_end
            and fold.test_start == short_fold.test_start
            and fold.test_end == short_fold.test_end
        ]
        assert len(matching) == 1, "every short-run fold must also appear in the long run"
        (long_fold,) = matching
        assert short_fold.model_id == long_fold.model_id
        assert short_fold.performance.net_pnl == pytest.approx(long_fold.performance.net_pnl)
        assert short_fold.performance.trade_count == long_fold.performance.trade_count
        assert short_fold.performance.total_return == pytest.approx(
            long_fold.performance.total_return
        )


# --------------------------------------------------------------------------
# Direct mutation: overwriting only the tail of the data must not change
# anything computed from the untouched head
# --------------------------------------------------------------------------


def _crash_the_tail(
    market_data: object, instrument_ids: list[str], index_symbol: str, from_date: dt.date
) -> None:
    """Directly overwrites every bar/observation on or after ``from_date``
    with an extreme, obviously-different value -- a crash -- while leaving
    everything before it untouched. Reaches into the fake provider's
    internal dicts on purpose: this is a white-box test of exactly the
    "does the future leak backward" question.
    """
    bars: dict[str, list[DailyBar]] = market_data._bars  # type: ignore[attr-defined]
    for instrument_id in instrument_ids:
        bars[instrument_id] = [
            bar
            if bar.session_date < from_date
            else DailyBar(
                instrument_id=instrument_id,
                session_date=bar.session_date,
                open=Decimal("1.00"),
                high=Decimal("1.05"),
                low=Decimal("0.95"),
                close=Decimal("1.00"),
                volume=1,
            )
            for bar in bars[instrument_id]
        ]
    index: dict[str, list[IndexObservation]] = market_data._index  # type: ignore[attr-defined]
    index[index_symbol] = [
        observation
        if observation.session_date < from_date
        else IndexObservation(index_symbol, observation.session_date, close=Decimal("1000.0"))
        for observation in index[index_symbol]
    ]


def test_mutating_only_future_bars_does_not_change_an_earlier_days_decision(
    tmp_path: Path,
) -> None:
    baseline_env = Environment(n_days=120, n_stocks=4, seed=5)
    mutated_env = Environment(n_days=120, n_stocks=4, seed=5)

    crash_from = mutated_env.dates[100]
    _crash_the_tail(
        mutated_env.market_data, mutated_env.instrument_ids, "NIFTY50", crash_from
    )

    signal_date = baseline_env.dates[80]
    band = baseline_env.regime_policy.band_for(AllocationRegime.LOW_RISK)
    target = AllocationTarget(
        as_of=signal_date,
        regime=AllocationRegime.LOW_RISK,
        target_gross_exposure=band.max_gross_exposure,
        min_gross_exposure=band.max_gross_exposure,
        max_gross_exposure=band.max_gross_exposure,
        allow_new_positions=True,
        confidence=1.0,
        expected_volatility=0.1,
        reason="test",
    )

    baseline_engine = baseline_env.engine(tmp_path / "baseline")
    mutated_engine = mutated_env.engine(tmp_path / "mutated")

    baseline_result = baseline_engine.run("t", {signal_date: target}, [signal_date], 1_000_000.0)
    mutated_result = mutated_engine.run("t", {signal_date: target}, [signal_date], 1_000_000.0)

    assert len(baseline_result.orders) == len(mutated_result.orders)
    for baseline_order, mutated_order in zip(
        baseline_result.orders, mutated_result.orders, strict=True
    ):
        assert baseline_order == mutated_order
    for baseline_fill, mutated_fill in zip(
        baseline_result.fills, mutated_result.fills, strict=True
    ):
        assert baseline_fill.fill_price == pytest.approx(mutated_fill.fill_price)
        assert baseline_fill.quantity == mutated_fill.quantity
