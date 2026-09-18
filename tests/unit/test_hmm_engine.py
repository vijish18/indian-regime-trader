"""The regime engine: model selection and its validation gates, measured
state statistics, labelling by measured volatility, and the requirement that
regime *names* never drive behavior.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from config.models import HMMConfig
from core.regime.gaussian_hmm import fit_gaussian_hmm
from core.regime.hmm_engine import (
    FittedRegimeModel,
    HMMRegimeEngine,
    InsufficientHistoryError,
    ModelSelectionError,
    RegimeLabel,
    assign_labels,
    characterize_states,
    volatility_ordered_states,
)


def hmm_config(**overrides: object) -> HMMConfig:
    """A small, fast config; overrides let a test target one gate."""
    defaults: dict[str, object] = {
        "candidate_states": [2, 3],
        "covariance_type": "diag",
        "training_window_days": 252,
        "retrain_interval_sessions": 63,
        "min_confidence": 0.6,
        "confirmation_bars": 2,
        "flicker_window_sessions": 20,
        "max_covariance_condition_number": 1_000_000.0,
        "random_seeds": [17, 29],
        "max_iterations": 200,
        "convergence_tolerance": 0.0001,
        "covariance_regularization": 0.000001,
        "min_state_occupancy": 0.02,
    }
    defaults.update(overrides)
    return HMMConfig.model_validate(defaults)


def market(n: int = 300, seed: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """A two-regime synthetic market: alternating calm and volatile blocks.

    Returns the (already-scaled) feature matrix and the raw daily returns the
    engine uses to describe -- not to fit -- the resulting states.
    """
    rng = np.random.default_rng(seed)
    calm = (np.arange(n) // 50) % 2 == 0
    daily_volatility = np.where(calm, 0.004, 0.022)
    returns = rng.normal(0.0002, daily_volatility)
    index = pd.bdate_range("2021-01-04", periods=n)

    series = pd.Series(returns, index=index)
    features = pd.DataFrame(
        {
            "abs_return": series.abs(),
            "rolling_vol": series.rolling(10).std(),
        },
        index=index,
    ).dropna()
    standardized = (features - features.mean()) / features.std(ddof=0)
    return standardized, series.reindex(standardized.index)


FittedFixture = tuple[HMMRegimeEngine, FittedRegimeModel, pd.DataFrame, "pd.Series[float]"]


@pytest.fixture(scope="module")
def fitted() -> FittedFixture:
    features, returns = market()
    engine = HMMRegimeEngine(hmm_config())
    return engine, engine.fit(features, returns), features, returns


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------


def test_fit_selects_the_lowest_bic_among_accepted_candidates(fitted: FittedFixture) -> None:
    _, model, _, _ = fitted
    accepted = [candidate for candidate in model.candidates if candidate.accepted]

    assert accepted, "expected at least one candidate to pass validation"
    assert model.training_result.bic == min(candidate.bic for candidate in accepted)
    assert model.training_result.accepted


def test_fit_evaluates_every_state_count_and_seed_combination(fitted: FittedFixture) -> None:
    _, model, _, _ = fitted
    config = hmm_config()
    assert len(model.candidates) == len(config.candidate_states) * len(config.random_seeds)


def test_every_candidate_is_retained_for_audit(fitted: FittedFixture) -> None:
    """A selection decision can only be reviewed if the rejected options were
    recorded along with why they were rejected.
    """
    _, model, _, _ = fitted
    for candidate in model.candidates:
        assert candidate.seed in hmm_config().random_seeds
        assert candidate.n_states in hmm_config().candidate_states
        if not candidate.accepted:
            assert candidate.rejection_reason


def test_fit_is_deterministic(fitted: FittedFixture) -> None:
    engine, model, features, returns = fitted
    again = engine.fit(features, returns)

    assert again.training_result.seed == model.training_result.seed
    assert again.training_result.n_states == model.training_result.n_states
    assert again.training_result.bic == model.training_result.bic
    np.testing.assert_array_equal(again.parameters.means, model.parameters.means)
    np.testing.assert_array_equal(
        again.parameters.transition_matrix, model.parameters.transition_matrix
    )


def test_selected_model_converged(fitted: FittedFixture) -> None:
    _, model, _, _ = fitted
    assert model.training_result.converged
    assert model.training_result.iterations >= 1


# --------------------------------------------------------------------------
# Validation gates -- each one fails closed
# --------------------------------------------------------------------------


def test_non_convergence_rejects_every_candidate() -> None:
    features, returns = market(n=200)
    engine = HMMRegimeEngine(hmm_config(max_iterations=1, convergence_tolerance=1e-12))

    with pytest.raises(ModelSelectionError, match="converge"):
        engine.fit(features, returns)


def test_degenerate_states_reject_every_candidate() -> None:
    """A state nothing ever occupies is not a regime; a model containing one
    is rejected rather than used with a phantom state.
    """
    features, returns = market(n=200)
    engine = HMMRegimeEngine(hmm_config(min_state_occupancy=0.99))

    with pytest.raises(ModelSelectionError, match="degenerate state"):
        engine.fit(features, returns)


def test_near_singular_covariance_rejects_every_candidate() -> None:
    features, returns = market(n=200)
    engine = HMMRegimeEngine(hmm_config(max_covariance_condition_number=1.0000001))

    with pytest.raises(ModelSelectionError, match="near-singular covariance"):
        engine.fit(features, returns)


def test_insufficient_history_rejects_candidates_with_a_clear_reason() -> None:
    """A model may not have more free parameters than observations."""
    features, returns = market(n=300)
    short_features = features.iloc[:12]
    short_returns = returns.iloc[:12]
    engine = HMMRegimeEngine(hmm_config(candidate_states=[8], random_seeds=[17]))

    with pytest.raises(ModelSelectionError, match="insufficient history"):
        engine.fit(short_features, short_returns)


def test_empty_training_data_raises() -> None:
    engine = HMMRegimeEngine(hmm_config())
    with pytest.raises(InsufficientHistoryError, match="empty training window"):
        engine.fit(pd.DataFrame(), pd.Series(dtype=float))


def test_nan_features_raise_rather_than_being_silently_dropped() -> None:
    features, returns = market(n=120)
    features = features.copy()
    features.iloc[5, 0] = np.nan
    engine = HMMRegimeEngine(hmm_config())

    with pytest.raises(InsufficientHistoryError, match="drop warm-up rows"):
        engine.fit(features, returns)


def test_returns_must_cover_every_training_session() -> None:
    features, returns = market(n=120)
    engine = HMMRegimeEngine(hmm_config())

    with pytest.raises(InsufficientHistoryError, match="do not cover every training session"):
        engine.fit(features, returns.iloc[:-10])


# --------------------------------------------------------------------------
# Filtered inference through the engine
# --------------------------------------------------------------------------


def test_engine_filtering_is_unaffected_by_future_sessions(fitted: FittedFixture) -> None:
    """The causality guarantee, asserted at the level a caller actually
    consumes: RegimeState objects, not raw arrays.
    """
    engine, model, features, _ = fitted
    cutoff = 150

    truncated = engine.filter(model, features.iloc[:cutoff])
    full = engine.filter(model, features)

    for early, complete in zip(truncated, full[:cutoff], strict=True):
        assert early == complete


def test_filter_latest_matches_the_last_filtered_row(fitted: FittedFixture) -> None:
    engine, model, features, _ = fitted
    assert engine.filter_latest(model, features) == engine.filter(model, features)[-1]


def test_regime_states_carry_measured_statistics_not_just_a_label(fitted: FittedFixture) -> None:
    engine, model, features, _ = fitted
    states = engine.filter(model, features)

    for state in states[:20]:
        statistic = model.statistics_for(state.state_id)
        assert state.expected_volatility == statistic.expected_volatility
        assert state.expected_return == statistic.expected_return
        assert state.persistence == statistic.self_transition_probability
        assert math.isfinite(state.expected_volatility)


def test_filtered_probabilities_are_distributions(fitted: FittedFixture) -> None:
    engine, model, features, _ = fitted
    for state in engine.filter(model, features):
        assert sum(state.probabilities) == pytest.approx(1.0)
        assert 0.0 <= state.confidence <= 1.0
        assert state.confidence == max(state.probabilities)


def test_filter_rejects_a_different_feature_set(fitted: FittedFixture) -> None:
    """A model may only be applied to the features it was trained on."""
    engine, model, features, _ = fitted
    renamed = features.rename(columns={"abs_return": "something_else"})

    with pytest.raises(ValueError, match="do not match the model's training columns"):
        engine.filter(model, renamed)


def test_filter_rejects_an_empty_window(fitted: FittedFixture) -> None:
    engine, model, features, _ = fitted
    with pytest.raises(ValueError, match="empty feature window"):
        engine.filter(model, features.iloc[:0])


def test_engine_filtering_ignores_the_models_fitted_start_distribution(
    fitted: FittedFixture,
) -> None:
    """``filter`` is handed a window beginning wherever the caller asked --
    a backtest fold's warmup buffer, or the last N live sessions -- which has
    no relationship to the day the training sequence began.

    ``start_probabilities`` answers a question about that training day only.
    Fitted on a single sequence, Baum-Welch drives it to a one-hot vector
    (true in all 32 walk-forward folds on real data), so filtering under it
    asserts the market reopened in whichever state training started in. When
    that is wrong and the first observation sits far in that state's tails,
    the posterior loses all its mass and inference *raises* -- walk-forward
    fold 11 died exactly this way on 2020-04-22.

    So the engine must not consult it at all, which is what this asserts:
    replacing it with a maximally misleading one-hot changes nothing.
    """
    engine, model, features, _ = fitted
    one_hot = np.zeros(model.parameters.n_states, dtype=float)
    one_hot[-1] = 1.0
    misleading = dataclasses.replace(
        model,
        parameters=dataclasses.replace(model.parameters, start_probabilities=one_hot),
    )

    assert engine.filter(misleading, features) == engine.filter(model, features)


# --------------------------------------------------------------------------
# Measured state statistics
# --------------------------------------------------------------------------


def test_states_separate_the_two_volatility_regimes(fitted: FittedFixture) -> None:
    """The engine must actually measure something: with a market built from a
    quiet and a turbulent regime, the fitted states' measured volatilities
    should be clearly different.
    """
    _, model, _, _ = fitted
    volatilities = sorted(statistic.expected_volatility for statistic in model.statistics)

    assert volatilities[0] > 0
    assert volatilities[-1] > 2 * volatilities[0]


def test_state_statistics_are_internally_consistent(fitted: FittedFixture) -> None:
    _, model, _, _ = fitted

    assert sum(statistic.occupancy for statistic in model.statistics) == pytest.approx(1.0)
    for statistic in model.statistics:
        assert 0.0 <= statistic.occupancy <= 1.0
        assert statistic.expected_volatility >= 0.0
        assert statistic.downside_volatility >= 0.0
        assert statistic.expected_duration >= 1.0
        assert 0.0 <= statistic.self_transition_probability <= 1.0


def test_expected_duration_matches_the_transition_matrix(fitted: FittedFixture) -> None:
    _, model, _, _ = fitted
    for statistic in model.statistics:
        state_id = statistic.state_id
        implied = 1.0 / (1.0 - model.parameters.transition_matrix[state_id, state_id])
        assert statistic.expected_duration == pytest.approx(implied, rel=1e-9)


def test_characterize_states_rejects_mismatched_lengths(fitted: FittedFixture) -> None:
    _, model, _, returns = fitted
    responsibilities = np.full((10, model.n_states), 1.0 / model.n_states)

    with pytest.raises(ValueError, match="responsibilities cover"):
        characterize_states(
            model.parameters, responsibilities, returns.to_numpy().astype(np.float64)
        )


def test_statistics_use_returns_conditioned_on_each_state() -> None:
    """A state that owns only the turbulent sessions must measure a higher
    volatility than one that owns only the quiet sessions -- the statistics
    describe the market while a state was in force.
    """
    returns = np.concatenate([np.full(50, 0.001), np.full(50, -0.05)])
    responsibilities = np.zeros((100, 2))
    responsibilities[:50, 0] = 1.0
    responsibilities[50:, 1] = 1.0

    fit = fit_gaussian_hmm(returns.reshape(-1, 1), n_states=2, seed=3, max_iterations=30)
    statistics = characterize_states(
        fit.parameters, responsibilities, np.asarray(returns, dtype=np.float64)
    )

    quiet, turbulent = statistics[0], statistics[1]
    assert quiet.expected_return > 0
    assert turbulent.expected_return < 0
    assert turbulent.downside_volatility > quiet.downside_volatility


# --------------------------------------------------------------------------
# Labelling: names follow measurements, and never drive behavior
# --------------------------------------------------------------------------


def test_labels_are_assigned_by_measured_volatility_rank() -> None:
    labels = assign_labels([0.40, 0.08, 0.22, 0.14])

    assert labels[1] is RegimeLabel.CALM  # lowest measured volatility
    assert labels[0] is RegimeLabel.CRISIS  # highest
    assert labels[3] is RegimeLabel.NORMAL
    assert labels[2] is RegimeLabel.ELEVATED


def test_labels_follow_statistics_when_state_ids_are_shuffled() -> None:
    """The same four states presented in a different order must receive the
    same labels -- a raw state ID carries no meaning.
    """
    volatilities = [0.40, 0.08, 0.22, 0.14]
    shuffled = [volatilities[index] for index in (2, 0, 3, 1)]

    original = assign_labels(volatilities)
    reordered = assign_labels(shuffled)

    assert reordered[1] is original[0]
    assert reordered[3] is original[1]
    assert reordered[0] is original[2]
    assert reordered[2] is original[3]


@pytest.mark.parametrize("n_states", [2, 3, 4, 5, 6])
def test_labelling_spans_the_scale_for_any_state_count(n_states: int) -> None:
    volatilities = [0.05 * (index + 1) for index in range(n_states)]
    labels = assign_labels(volatilities)

    assert len(labels) == n_states
    assert labels[0] is RegimeLabel.CALM
    assert labels[n_states - 1] is RegimeLabel.CRISIS


def test_more_states_than_labels_may_share_a_label() -> None:
    """Deliberate and harmless: labels are for reading, not for deciding."""
    labels = assign_labels([0.05 * (index + 1) for index in range(6)])
    assert len(set(labels.values())) < 6


def test_permuting_state_ids_leaves_every_risk_relevant_output_unchanged(
    fitted: FittedFixture,
) -> None:
    """The requirement stated plainly: regime *names* and state IDs must not
    determine behavior. Relabel a fitted model's states and every measured
    quantity a policy consumes stays identical, session by session -- only
    the arbitrary integer changes.
    """
    engine, model, features, _ = fitted
    order = np.arange(model.n_states)[::-1].copy()

    permuted_statistics = tuple(
        dataclasses.replace(model.statistics_for(int(old_id)), state_id=new_id)
        for new_id, old_id in enumerate(order)
    )
    permuted = FittedRegimeModel(
        parameters=model.parameters.relabel(order),
        statistics=permuted_statistics,
        training_result=model.training_result,
    )

    before = engine.filter(model, features)
    after = engine.filter(permuted, features)

    assert [state.expected_volatility for state in after] == [
        state.expected_volatility for state in before
    ]
    assert [state.expected_return for state in after] == [state.expected_return for state in before]
    assert [state.label for state in after] == [state.label for state in before]
    assert [state.confidence for state in after] == pytest.approx(
        [state.confidence for state in before]
    )
    # The arbitrary part did change, which is what makes the above meaningful.
    assert [state.state_id for state in after] != [state.state_id for state in before]


def test_volatility_ordered_states_ranks_by_measurement(fitted: FittedFixture) -> None:
    _, model, _, _ = fitted
    order = volatility_ordered_states(model.statistics)

    volatilities = [model.statistics_for(state_id).expected_volatility for state_id in order]
    assert volatilities == sorted(volatilities)


def test_statistics_for_an_unknown_state_raises(fitted: FittedFixture) -> None:
    _, model, _, _ = fitted
    with pytest.raises(KeyError, match="no statistics for state"):
        model.statistics_for(99)
