"""WalkForwardValidator: fold generation (configurable train/test/step
windows), the fit -> freeze -> OOS -> advance -> retrain loop, the
five-strategy comparison, and the shuffled-regime control. Look-ahead
properties specifically are covered in
``tests/unit/test_no_lookahead_walkforward.py``.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import pandas as pd
import pytest

from backtest.engine import BacktestResult
from backtest.performance import PerformanceReport
from backtest.walk_forward import (
    BUY_AND_HOLD,
    HMM,
    STRATEGY_NAMES,
    CompletedFolds,
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


# --------------------------------------------------------------------------
# Folds must tile, not overlap
# --------------------------------------------------------------------------


def test_overlapping_test_windows_are_refused_at_construction(tmp_path: Path) -> None:
    """Regression: this silently corrupted every chained result.

    With test_window_sessions=126 and roll_step_sessions=63, consecutive
    folds overlapped by 63 sessions. Each fold's equity curve is concatenated
    into one series, so half the out-of-sample period was counted twice and
    every fold seam appeared as a return no market delivered -- 4,030 rows
    over 2,078 distinct dates, ~100% annualised volatility, and buy-and-hold
    showing a 50% loss across a period the market roughly doubled in.

    It is refused rather than de-duplicated because the fold-to-fold chain
    requires each fold to begin where the previous one ended, and overlapping
    windows have no such point.
    """
    env = Environment(n_days=150)

    with pytest.raises(WalkForwardError, match="would overlap"):
        env.validator(
            tmp_path,
            config=type(env.backtest_cfg).model_validate(
                {
                    **env.backtest_cfg.model_dump(),
                    "training_window_sessions": 50,
                    "test_window_sessions": 20,
                    "roll_step_sessions": 10,
                }
            ),
        )


def test_equal_window_and_step_tile_the_period_exactly(tmp_path: Path) -> None:
    """The property the refusal protects: every out-of-sample session belongs
    to exactly one fold, so concatenating folds yields a curve with no
    repeated dates."""
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

    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier[3] < later[2], "a fold's test window must end before the next begins"


def test_percent_invested_is_reported_not_nan(tmp_path: Path) -> None:
    """cash_history was never threaded through, so pct_invested and pct_cash
    came back nan on every walk-forward run -- the one figure showing how
    much of the period was actually spent in the market."""
    env = Environment(n_days=90, n_stocks=4)
    validator = _single_fold_validator(env, tmp_path)

    results = validator.run_all_strategies(env.dates[0], env.dates[-1], strategies=[BUY_AND_HOLD])

    assert not math.isnan(results[BUY_AND_HOLD].pct_invested)
    assert 0.0 <= results[BUY_AND_HOLD].pct_invested <= 1.0


# --------------------------------------------------------------------------
# Resuming a run that died partway
# --------------------------------------------------------------------------


def _multi_fold_validator(env: Environment, tmp_path: Path) -> WalkForwardValidator:
    """Four tiling folds, so there is a "partway" to resume from.

    A 60-session training window, not less: the HMM needs enough
    observations after feature warmup to fit at all, and a shorter one fails
    model selection rather than testing anything about resuming."""
    return env.validator(
        tmp_path,
        config=type(env.backtest_cfg).model_validate(
            {
                **env.backtest_cfg.model_dump(),
                "training_window_sessions": 60,
                "test_window_sessions": 20,
                "roll_step_sessions": 20,
            }
        ),
    )


def _capture_folds(
    validator: WalkForwardValidator, env: Environment, name: str, stop_after: int
) -> CompletedFolds:
    """Run the folds and keep exactly what a crash after ``stop_after`` of
    them would have left on disk."""
    equity: list[pd.Series] = []
    trades: list[pd.DataFrame] = []
    cash: list[pd.Series] = []

    def on_fold(strategy: str, fold_index: int, result: BacktestResult) -> None:
        if strategy != name or fold_index >= stop_after:
            return
        equity.append(result.equity_curve)
        trades.append(result.trade_log)
        cash.append(result.cash_history)

    validator.run_all_strategies(
        env.dates[0], env.dates[-1], strategies=[name], on_fold=on_fold
    )
    return CompletedFolds(
        folds_done=stop_after,
        equity=pd.concat(equity).sort_index(),
        trades=pd.concat(trades, ignore_index=True),
        cash=pd.concat(cash).sort_index(),
    )


def test_a_resumed_run_matches_the_run_it_resumes(tmp_path: Path) -> None:
    """The property everything else here rests on.

    A run that died at fold 3 and was resumed must report exactly what one
    uninterrupted pass would have reported. If it does not, resuming is not
    a recovery -- it is a second, different experiment wearing the first
    one name, and nobody could tell from the output.
    """
    env = Environment(n_days=150, n_stocks=4)
    whole = _multi_fold_validator(env, tmp_path / "whole").run_all_strategies(
        env.dates[0], env.dates[-1], strategies=[BUY_AND_HOLD]
    )

    partial = _capture_folds(
        _multi_fold_validator(env, tmp_path / "died"), env, BUY_AND_HOLD, stop_after=2
    )
    resumed = _multi_fold_validator(env, tmp_path / "resumed").run_all_strategies(
        env.dates[0],
        env.dates[-1],
        strategies=[BUY_AND_HOLD],
        resume={BUY_AND_HOLD: partial},
    )

    assert resumed[BUY_AND_HOLD].cagr == pytest.approx(whole[BUY_AND_HOLD].cagr)
    assert resumed[BUY_AND_HOLD].total_return == pytest.approx(
        whole[BUY_AND_HOLD].total_return
    )
    assert resumed[BUY_AND_HOLD].max_drawdown == pytest.approx(
        whole[BUY_AND_HOLD].max_drawdown
    )
    assert resumed[BUY_AND_HOLD].trade_count == whole[BUY_AND_HOLD].trade_count
    # The one a naive resume gets wrong: cash is not in the equity curve, so
    # a resume that cannot read it back reports nan for the whole period.
    assert resumed[BUY_AND_HOLD].pct_invested == pytest.approx(
        whole[BUY_AND_HOLD].pct_invested
    )


def test_a_resumed_run_does_not_recompute_the_folds_it_was_given(
    tmp_path: Path,
) -> None:
    """Otherwise "resume" saves nothing, which is the entire point on a run
    measured in hours."""
    env = Environment(n_days=150, n_stocks=4)
    partial = _capture_folds(
        _multi_fold_validator(env, tmp_path / "died"), env, BUY_AND_HOLD, stop_after=2
    )

    seen: list[int] = []

    def on_fold(name: str, fold_index: int, result: BacktestResult) -> None:
        seen.append(fold_index)

    validator = _multi_fold_validator(env, tmp_path / "resumed")
    total = len(validator.generate_folds(env.dates[0], env.dates[-1]))
    validator.run_all_strategies(
        env.dates[0],
        env.dates[-1],
        strategies=[BUY_AND_HOLD],
        on_fold=on_fold,
        resume={BUY_AND_HOLD: partial},
    )

    assert total > 2
    assert seen == list(range(2, total))


def test_equity_continues_from_where_the_crashed_run_reached(tmp_path: Path) -> None:
    """Equity chains across folds. Resuming at the initial equity instead of
    the checkpoint would silently restate the whole run."""
    env = Environment(n_days=150, n_stocks=4)
    partial = _capture_folds(
        _multi_fold_validator(env, tmp_path / "died"), env, BUY_AND_HOLD, stop_after=2
    )

    starts: list[float] = []
    validator = _multi_fold_validator(env, tmp_path / "resumed")

    def on_fold(name: str, fold_index: int, result: BacktestResult) -> None:
        starts.append(float(result.equity_curve.iloc[0]))

    validator.run_all_strategies(
        env.dates[0],
        env.dates[-1],
        strategies=[BUY_AND_HOLD],
        on_fold=on_fold,
        resume={BUY_AND_HOLD: partial},
    )

    assert starts
    # Within one session of where the checkpoint ended, not back at the start.
    assert abs(starts[0] / partial.running_equity - 1.0) < 0.25
    assert abs(starts[0] / validator.initial_equity - 1.0) > 1e-9


def test_a_resume_state_for_a_strategy_not_being_run_is_refused(
    tmp_path: Path,
) -> None:
    env = Environment(n_days=150, n_stocks=4)
    validator = _multi_fold_validator(env, tmp_path)
    state = CompletedFolds(
        folds_done=1,
        equity=pd.Series({env.dates[0]: 1.0}),
        trades=pd.DataFrame(),
        cash=pd.Series({env.dates[0]: 1.0}),
    )

    with pytest.raises(WalkForwardError, match="not being run"):
        validator.run_all_strategies(
            env.dates[0], env.dates[-1], strategies=[BUY_AND_HOLD], resume={HMM: state}
        )


def test_a_resume_claiming_more_folds_than_exist_is_refused(tmp_path: Path) -> None:
    """A checkpoint from a different date range or window size. Splicing it in
    would report two different experiments as one."""
    env = Environment(n_days=150, n_stocks=4)
    validator = _multi_fold_validator(env, tmp_path)
    state = CompletedFolds(
        folds_done=999,
        equity=pd.Series({env.dates[0]: 1.0}),
        trades=pd.DataFrame(),
        cash=pd.Series({env.dates[0]: 1.0}),
    )

    with pytest.raises(WalkForwardError, match="different run"):
        validator.run_all_strategies(
            env.dates[0],
            env.dates[-1],
            strategies=[BUY_AND_HOLD],
            resume={BUY_AND_HOLD: state},
        )


def test_a_resume_with_completed_folds_but_no_equity_is_refused() -> None:
    """The checkpoint existed but had nothing in it. Continuing from an empty
    curve would restart at the initial equity while reporting folds done."""
    with pytest.raises(WalkForwardError, match="not usable"):
        CompletedFolds(
            folds_done=3,
            equity=pd.Series(dtype=float),
            trades=pd.DataFrame(),
            cash=pd.Series(dtype=float),
        )


def test_an_empty_resume_state_behaves_exactly_like_no_resume(tmp_path: Path) -> None:
    env = Environment(n_days=150, n_stocks=4)
    plain = _multi_fold_validator(env, tmp_path / "plain").run_all_strategies(
        env.dates[0], env.dates[-1], strategies=[BUY_AND_HOLD]
    )
    zeroed = _multi_fold_validator(env, tmp_path / "zero").run_all_strategies(
        env.dates[0],
        env.dates[-1],
        strategies=[BUY_AND_HOLD],
        resume={
            BUY_AND_HOLD: CompletedFolds(
                folds_done=0,
                equity=pd.Series(dtype=float),
                trades=pd.DataFrame(),
                cash=pd.Series(dtype=float),
            )
        },
    )

    assert zeroed[BUY_AND_HOLD].cagr == pytest.approx(plain[BUY_AND_HOLD].cagr)
