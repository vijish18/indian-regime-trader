"""WalkForwardValidator: fold generation (configurable train/test/step
windows), the fit -> freeze -> OOS -> advance -> retrain loop, the
five-strategy comparison, and the shuffled-regime control. Look-ahead
properties specifically are covered in
``tests/unit/test_no_lookahead_walkforward.py``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from backtest.performance import PerformanceReport
from backtest.walk_forward import (
    BUY_AND_HOLD,
    HMM,
    STRATEGY_NAMES,
    WalkForwardError,
    WalkForwardFold,
    WalkForwardValidator,
)
from core.regime.allocation import AllocationRegime, AllocationTarget
from tests.unit._wf_support import Environment

# --------------------------------------------------------------------------
# Fold generation
# --------------------------------------------------------------------------


def test_generate_folds_respects_configured_window_sizes(tmp_path: Path) -> None:
    env = Environment(n_days=150)
    validator = env.validator(
        tmp_path,
        config=type(env.backtest_cfg).model_validate(
            {
                **env.backtest_cfg.model_dump(),
                "training_window_sessions": 50,
                "test_window_sessions": 10,
                "roll_step_sessions": 10,
            }
        ),
    )
    folds = validator.generate_folds(env.dates[0], env.dates[-1])
    assert folds  # at least one fold fits in 150 trading days

    train_start, train_end, test_start, test_end = folds[0]
    trading_days = env.calendar.trading_days_between(env.dates[0], env.dates[-1])
    assert train_start == trading_days[0]
    assert train_end == trading_days[49]
    assert test_start == trading_days[50]
    assert test_end == trading_days[59]


def test_generate_folds_rolls_the_training_window_forward_not_expanding(
    tmp_path: Path,
) -> None:
    env = Environment(n_days=200)
    validator = env.validator(
        tmp_path,
        config=type(env.backtest_cfg).model_validate(
            {
                **env.backtest_cfg.model_dump(),
                "training_window_sessions": 40,
                "test_window_sessions": 10,
                "roll_step_sessions": 10,
            }
        ),
    )
    folds = validator.generate_folds(env.dates[0], env.dates[-1])
    assert len(folds) >= 2

    for train_start, train_end, _test_start, _test_end in folds:
        trading_days_in_window = env.calendar.trading_days_between(train_start, train_end)
        assert len(trading_days_in_window) == 40  # constant length, never growing

    # Consecutive folds' training windows start `roll_step_sessions` apart.
    starts = [fold[0] for fold in folds]
    for earlier, later in zip(starts, starts[1:], strict=False):
        gap = env.calendar.trading_days_between(earlier, later)
        assert len(gap) - 1 == 10


def test_generate_folds_is_empty_when_no_fold_fits(tmp_path: Path) -> None:
    env = Environment(n_days=30)
    validator = env.validator(
        tmp_path,
        config=type(env.backtest_cfg).model_validate(
            {
                **env.backtest_cfg.model_dump(),
                "training_window_sessions": 100,
                "test_window_sessions": 20,
                "roll_step_sessions": 20,
            }
        ),
    )
    folds = validator.generate_folds(env.dates[0], env.dates[-1])
    assert folds == []


# --------------------------------------------------------------------------
# run(): the HMM strategy, walk-forward proper
# --------------------------------------------------------------------------


def test_run_produces_one_fold_per_window(tmp_path: Path) -> None:
    env = Environment(n_days=150)
    validator = env.validator(tmp_path)
    folds = validator.run(env.dates[0], env.dates[-1])
    expected = validator.generate_folds(env.dates[0], env.dates[-1])
    assert len(folds) == len(expected)
    for fold in folds:
        assert isinstance(fold, WalkForwardFold)
        assert isinstance(fold.performance, PerformanceReport)


def test_run_raises_when_no_fold_fits_the_range(tmp_path: Path) -> None:
    env = Environment(n_days=30)
    validator = env.validator(
        tmp_path,
        config=type(env.backtest_cfg).model_validate(
            {
                **env.backtest_cfg.model_dump(),
                "training_window_sessions": 100,
                "test_window_sessions": 20,
                "roll_step_sessions": 20,
            }
        ),
    )
    with pytest.raises(WalkForwardError):
        validator.run(env.dates[0], env.dates[-1])


def test_each_fold_fits_a_model_on_its_own_training_window_only(tmp_path: Path) -> None:
    """A crude but direct check that fitting actually happened per fold:
    each fold's model_id encodes its own train_end date."""
    env = Environment(n_days=150)
    validator = env.validator(tmp_path)
    folds = validator.run(env.dates[0], env.dates[-1])
    for fold in folds:
        assert fold.train_end.isoformat() in fold.model_id


# --------------------------------------------------------------------------
# run_all_strategies(): the five-way comparison
# --------------------------------------------------------------------------


def _single_fold_validator(env: Environment, tmp_path: Path) -> WalkForwardValidator:
    """A config that fits exactly one fold over the environment's data --
    keeps the 5-strategy runs (each a full pipeline pass) fast without
    losing coverage of the comparison itself."""
    return env.validator(
        tmp_path,
        config=type(env.backtest_cfg).model_validate(
            {
                **env.backtest_cfg.model_dump(),
                "training_window_sessions": 60,
                "test_window_sessions": 15,
                "roll_step_sessions": 15,
            }
        ),
    )


def test_run_all_strategies_returns_every_named_strategy(tmp_path: Path) -> None:
    env = Environment(n_days=90, n_stocks=4)
    validator = _single_fold_validator(env, tmp_path)
    results = validator.run_all_strategies(env.dates[0], env.dates[-1])

    assert set(results) == set(STRATEGY_NAMES)
    for name in STRATEGY_NAMES:
        assert isinstance(results[name], PerformanceReport)


def test_progress_is_reported_once_per_fold_and_is_optional(tmp_path: Path) -> None:
    """The full run over real data takes most of a day. A heartbeat is the
    only thing separating "slow" from "hung" while it does."""
    env = Environment(n_days=90, n_stocks=4)
    validator = _single_fold_validator(env, tmp_path)
    folds = validator.generate_folds(env.dates[0], env.dates[-1])

    seen: list[str] = []
    validator.run_all_strategies(env.dates[0], env.dates[-1], progress=seen.append)

    assert len(seen) == len(folds)
    assert f"fold 1/{len(folds)}" in seen[0]


def test_run_all_strategies_all_have_trades_over_the_same_window(tmp_path: Path) -> None:
    """Not a claim that every strategy trades identically -- but every one
    of them should have SOME trade activity given a full test window with
    real, tradable synthetic data, confirming they all actually ran the
    full pipeline rather than one silently no-op'ing."""
    env = Environment(n_days=90, n_stocks=4)
    validator = _single_fold_validator(env, tmp_path)
    results = validator.run_all_strategies(env.dates[0], env.dates[-1])

    for name in STRATEGY_NAMES:
        assert results[name].trade_count > 0, f"{name} recorded no trades"


def test_a_subset_run_is_identical_to_the_same_names_in_a_whole_run(tmp_path: Path) -> None:
    """The guarantee the parallel split rests on.

    The full five-way comparison measured at roughly 24 CPU hours over real
    data, so it is split across processes, one per strategy. That is only
    legitimate if a strategy's numbers do not depend on which other
    strategies happened to run beside it -- otherwise the split would
    quietly produce a different answer than the thing it replaces.

    They do not: each strategy chains equity only through its own folds,
    accumulates its own curves and trade logs, and gets its own
    circuit-breaker state file keyed by strategy name. This asserts it
    rather than trusting the reading.
    """
    env = Environment(n_days=90, n_stocks=4)
    whole = _single_fold_validator(env, tmp_path / "whole").run_all_strategies(
        env.dates[0], env.dates[-1]
    )
    part = _single_fold_validator(env, tmp_path / "part").run_all_strategies(
        env.dates[0], env.dates[-1], strategies=[HMM, BUY_AND_HOLD]
    )

    assert set(part) == {HMM, BUY_AND_HOLD}
    for name in (HMM, BUY_AND_HOLD):
        assert part[name].cagr == whole[name].cagr
        assert part[name].sharpe == whole[name].sharpe
        assert part[name].max_drawdown == whole[name].max_drawdown
        assert part[name].trade_count == whole[name].trade_count
        assert part[name].total_costs == whole[name].total_costs


def test_an_unknown_strategy_name_is_refused(tmp_path: Path) -> None:
    """A typo in a launcher script would otherwise run four strategies and
    silently report a four-row comparison as though it were five."""
    env = Environment(n_days=90, n_stocks=4)
    validator = _single_fold_validator(env, tmp_path)

    with pytest.raises(WalkForwardError, match="unknown strategy"):
        validator.run_all_strategies(env.dates[0], env.dates[-1], strategies=["hmmm"])


def test_run_all_strategies_raises_when_no_fold_fits(tmp_path: Path) -> None:
    env = Environment(n_days=30)
    validator = env.validator(
        tmp_path,
        config=type(env.backtest_cfg).model_validate(
            {
                **env.backtest_cfg.model_dump(),
                "training_window_sessions": 100,
                "test_window_sessions": 20,
                "roll_step_sessions": 20,
            }
        ),
    )
    with pytest.raises(WalkForwardError):
        validator.run_all_strategies(env.dates[0], env.dates[-1])


def test_buy_and_hold_never_reports_uncertain(tmp_path: Path) -> None:
    env = Environment(n_days=150)
    validator = env.validator(tmp_path)
    dates = env.dates[100:110]
    targets = validator._buy_and_hold_targets(dates)
    assert all(target.regime.value != "uncertain" for target in targets.values())
    assert all(target.confidence == 1.0 for target in targets.values())


# --------------------------------------------------------------------------
# run_shuffled_regime_control()
# --------------------------------------------------------------------------


def test_run_shuffled_regime_control_returns_one_overall_report(tmp_path: Path) -> None:
    env = Environment(n_days=150)
    validator = env.validator(tmp_path)
    report = validator.run_shuffled_regime_control(env.dates[0], env.dates[-1])
    assert isinstance(report, PerformanceReport)


def _targets(
    dates: list[dt.date], exposures: list[float]
) -> dict[dt.date, AllocationTarget]:
    """Finished exposure decisions, as ``_hmm_exposure_targets`` returns them."""
    return {
        day: AllocationTarget(
            as_of=day,
            regime=AllocationRegime.LOW_RISK if e > 0.5 else AllocationRegime.HIGH_RISK,
            target_gross_exposure=e,
            min_gross_exposure=0.0,
            max_gross_exposure=1.0,
            allow_new_positions=True,
            confidence=0.8,
            expected_volatility=0.10 if e > 0.5 else 0.35,
            reason="test",
        )
        for day, e in zip(dates, exposures, strict=True)
    }


def test_the_shuffled_control_preserves_the_exposure_distribution_exactly(
    tmp_path: Path,
) -> None:
    """The control must not fabricate exposures the HMM never chose: every
    date's target is one the HMM actually produced this fold, just
    probably not on that date.

    Preserving the multiset is what makes it a *control* -- the same time
    spent at each exposure level, only the timing destroyed. Any
    difference in outcome is then attributable to timing alone.
    """
    env = Environment(n_days=150)
    validator = env.validator(tmp_path)
    dates = env.dates[100:110]
    original = _targets(dates, [0.2, 0.9, 0.9, 0.2, 0.6, 0.8, 0.3, 0.9, 0.4, 0.7])

    shuffled = validator._shuffled_exposure_targets(original, seed=3)

    assert set(shuffled) == set(original)
    assert sorted(x.target_gross_exposure for x in shuffled.values()) == sorted(
        x.target_gross_exposure for x in original.values()
    )


def test_the_shuffled_control_can_actually_trade(tmp_path: Path) -> None:
    """The defect this replaced.

    The control used to shuffle ``RegimeState``s and re-run them through
    ``RegimeAllocationEngine``, whose confirmation bars and flicker guard
    exist to smooth a *real* sequence. A random one defeats them by
    construction: the confirmed tier changes almost every session,
    ``max_flicker_transitions`` trips, and the engine returns UNCERTAIN,
    which sets ``allow_new_positions=False``.

    Measured on a realistic 120-session path, the ordered sequence allowed
    new positions on 120 of 120 sessions and the shuffled one on 5. Every
    backtest duly reported the control as 0.00% on zero trades, and the
    question it exists to ask -- whether the HMM's *timing* matters -- went
    unanswered.
    """
    env = Environment(n_days=150)
    validator = env.validator(tmp_path)
    dates = env.dates[100:120]
    original = _targets(dates, [0.9] * 15 + [0.2] * 5)

    shuffled = validator._shuffled_exposure_targets(original, seed=7)

    assert all(x.allow_new_positions for x in shuffled.values())
    assert sum(x.target_gross_exposure for x in shuffled.values()) > 0


def test_the_shuffled_control_actually_reorders(tmp_path: Path) -> None:
    env = Environment(n_days=150)
    validator = env.validator(tmp_path)
    dates = env.dates[100:120]
    original = _targets(dates, [0.05 * (i + 1) for i in range(20)])

    shuffled = validator._shuffled_exposure_targets(original, seed=42)

    moved = sum(
        1
        for day in dates
        if shuffled[day].target_gross_exposure != original[day].target_gross_exposure
    )
    assert moved > 0


def test_the_shuffled_control_is_deterministic_given_the_same_seed(tmp_path: Path) -> None:
    """A control whose result changes between runs cannot be compared
    against anything."""
    env = Environment(n_days=150)
    validator = env.validator(tmp_path)
    dates = env.dates[100:110]
    original = _targets(dates, [0.1 * (i + 1) for i in range(10)])

    first = validator._shuffled_exposure_targets(original, seed=99)
    second = validator._shuffled_exposure_targets(original, seed=99)

    for day in first:
        assert first[day].target_gross_exposure == second[day].target_gross_exposure
        assert first[day].regime == second[day].regime
