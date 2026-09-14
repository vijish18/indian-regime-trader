"""WalkForwardValidator: fold generation (configurable train/test/step
windows), the fit -> freeze -> OOS -> advance -> retrain loop, the
five-strategy comparison, and the shuffled-regime control. Look-ahead
properties specifically are covered in
``tests/unit/test_no_lookahead_walkforward.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backtest.performance import PerformanceReport
from backtest.walk_forward import (
    STRATEGY_NAMES,
    WalkForwardError,
    WalkForwardFold,
    WalkForwardValidator,
)
from core.regime.hmm_engine import RegimeLabel, RegimeState
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


def test_shuffled_targets_preserve_the_same_multiset_of_states_reassigned_to_dates(
    tmp_path: Path,
) -> None:
    """The shuffled control must not fabricate new regime readings -- every
    date's target comes from *some* state the HMM actually produced this
    fold, just possibly not the one it produced for that date."""
    env = Environment(n_days=150)
    validator = env.validator(tmp_path)

    states = [
        RegimeState(
            as_of=env.dates[100 + i],
            state_id=i % 2,
            label=RegimeLabel.CALM if i % 2 == 0 else RegimeLabel.ELEVATED,
            probabilities=(0.8, 0.2) if i % 2 == 0 else (0.2, 0.8),
            confidence=0.8,
            expected_volatility=0.10 if i % 2 == 0 else 0.35,
            expected_return=0.0,
            persistence=0.9,
        )
        for i in range(10)
    ]

    shuffled_targets = validator._shuffled_exposure_targets(states, seed=3)

    assert set(shuffled_targets) == {state.as_of for state in states}
    original_volatilities = sorted(state.expected_volatility for state in states)
    shuffled_volatilities = sorted(
        target.expected_volatility for target in shuffled_targets.values()
    )
    assert shuffled_volatilities == original_volatilities


def test_shuffled_targets_are_a_different_assignment_than_the_original_with_high_probability(
    tmp_path: Path,
) -> None:
    env = Environment(n_days=150)
    validator = env.validator(tmp_path)
    states = [
        RegimeState(
            as_of=env.dates[100 + i],
            state_id=i % 3,
            label=RegimeLabel.NORMAL,
            probabilities=(0.34, 0.33, 0.33),
            confidence=0.7,
            expected_volatility=0.05 * (i + 1),
            expected_return=0.0,
            persistence=0.8,
        )
        for i in range(20)
    ]
    shuffled_targets = validator._shuffled_exposure_targets(states, seed=42)
    original_by_date = {state.as_of: state.expected_volatility for state in states}
    differing = sum(
        1
        for date, target in shuffled_targets.items()
        if target.expected_volatility != original_by_date[date]
    )
    assert differing > 0


def test_shuffled_control_is_deterministic_given_the_same_seed(tmp_path: Path) -> None:
    env = Environment(n_days=150)
    validator = env.validator(tmp_path)
    states = [
        RegimeState(
            as_of=env.dates[100 + i],
            state_id=i % 2,
            label=RegimeLabel.CALM,
            probabilities=(0.6, 0.4),
            confidence=0.75,
            expected_volatility=0.1 + 0.01 * i,
            expected_return=0.0,
            persistence=0.85,
        )
        for i in range(10)
    ]
    first = validator._shuffled_exposure_targets(states, seed=99)
    second = validator._shuffled_exposure_targets(states, seed=99)
    for date in first:
        assert first[date].expected_volatility == second[date].expected_volatility
        assert first[date].regime == second[date].regime
