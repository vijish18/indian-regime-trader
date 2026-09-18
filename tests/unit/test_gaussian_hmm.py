"""Gaussian HMM mathematics: the forward filter's causality guarantee, EM
correctness invariants, and numerical failure handling.

The single most important test in this file is
``test_filtered_belief_at_t_is_identical_with_and_without_future_data``: the
whole design rests on the live regime call at time t being a function of
observations 1..t only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.regime.gaussian_hmm import (
    GaussianHMMParameters,
    HMMNumericalError,
    compute_aic,
    compute_bic,
    covariance_condition_number,
    filter_step,
    fit_gaussian_hmm,
    forward_backward,
    forward_filter,
    log_emission_probabilities,
    predict_step,
    update_step,
)


def two_state_model() -> GaussianHMMParameters:
    """A hand-checkable 2-state, 1-feature model."""
    return GaussianHMMParameters(
        start_probabilities=np.array([0.6, 0.4]),
        transition_matrix=np.array([[0.7, 0.3], [0.4, 0.6]]),
        means=np.array([[0.0], [3.0]]),
        covariances=np.array([[[1.0]], [[1.0]]]),
        covariance_type="full",
    )


def regime_switching_observations(
    n: int = 240, seed: int = 3, n_features: int = 2
) -> np.ndarray:
    """Two clearly separated volatility regimes alternating in blocks."""
    rng = np.random.default_rng(seed)
    calm = (np.arange(n) // 40) % 2 == 0
    scale = np.where(calm, 0.4, 2.5)
    return rng.normal(0.0, 1.0, size=(n, n_features)) * scale[:, None]


def _normal_pdf(x: float, mean: float, variance: float) -> float:
    return math.exp(-0.5 * (x - mean) ** 2 / variance) / math.sqrt(2 * math.pi * variance)


# --------------------------------------------------------------------------
# The causality guarantee
# --------------------------------------------------------------------------


def test_filtered_belief_at_t_is_identical_with_and_without_future_data() -> None:
    """The core requirement: the regime inferred for session T must not change
    when sessions after T arrive.

    A smoothed posterior or a Viterbi path would fail this outright -- both
    are functions of the entire sequence.
    """
    model = two_state_model()
    observations = regime_switching_observations(n=200, n_features=1)

    cutoff = 120
    truncated = forward_filter(observations[:cutoff], model)
    full = forward_filter(observations, model)

    np.testing.assert_array_equal(
        full.filtered_probabilities[:cutoff], truncated.filtered_probabilities
    )


@pytest.mark.parametrize("cutoff", [1, 2, 17, 99, 199])
def test_every_prefix_length_gives_identical_history(cutoff: int) -> None:
    """Not just one cutoff: filtering any prefix reproduces exactly the rows
    the full run produced for those sessions.
    """
    model = two_state_model()
    observations = regime_switching_observations(n=200, n_features=1)

    prefix = forward_filter(observations[:cutoff], model)
    full = forward_filter(observations, model)

    np.testing.assert_array_equal(
        full.filtered_probabilities[:cutoff], prefix.filtered_probabilities
    )


def test_appending_an_extreme_future_observation_changes_nothing_earlier() -> None:
    """The sharpest version: a wild future print would visibly move earlier
    beliefs under any non-causal implementation.
    """
    model = two_state_model()
    observations = regime_switching_observations(n=80, n_features=1)
    shocked = np.vstack([observations, np.array([[50.0]])])

    before = forward_filter(observations, model)
    after = forward_filter(shocked, model)

    np.testing.assert_array_equal(
        after.filtered_probabilities[: len(observations)], before.filtered_probabilities
    )


def test_forward_filter_equals_repeated_filter_step() -> None:
    """The sequence filter is exactly the online step applied repeatedly --
    so live incremental inference and backtest inference cannot diverge.
    """
    model = two_state_model()
    observations = regime_switching_observations(n=60, n_features=1)

    result = forward_filter(observations, model)

    belief, _ = update_step(
        model.start_probabilities, log_emission_probabilities(observations[:1], model)[0]
    )
    np.testing.assert_allclose(result.filtered_probabilities[0], belief)
    for index in range(1, len(observations)):
        belief, _ = filter_step(belief, observations[index], model)
        np.testing.assert_allclose(result.filtered_probabilities[index], belief, atol=1e-12)


def test_filtered_and_smoothed_posteriors_differ() -> None:
    """Proof the filter is not accidentally returning smoothed values.

    Smoothing conditions on the whole sequence, so it disagrees with the
    filter everywhere except the final observation. If this test ever passes
    trivially (all equal), the filter has silently become a smoother.
    """
    model = two_state_model()
    observations = regime_switching_observations(n=120, n_features=1)

    filtered = forward_filter(observations, model).filtered_probabilities
    log_emissions = log_emission_probabilities(observations, model)
    smoothed, _, _ = forward_backward(log_emissions, model)

    assert not np.allclose(filtered[:-1], smoothed[:-1])
    # At the final observation there is no future left, so they must agree.
    np.testing.assert_allclose(filtered[-1], smoothed[-1], atol=1e-10)


# --------------------------------------------------------------------------
# Forward filter correctness
# --------------------------------------------------------------------------


def test_first_step_matches_hand_computed_bayes_update() -> None:
    model = two_state_model()
    observation = 0.0

    likelihood_calm = _normal_pdf(observation, 0.0, 1.0)
    likelihood_stress = _normal_pdf(observation, 3.0, 1.0)
    unnormalized = np.array([0.6 * likelihood_calm, 0.4 * likelihood_stress])
    expected = unnormalized / unnormalized.sum()

    result = forward_filter(np.array([[observation]]), model)
    np.testing.assert_allclose(result.filtered_probabilities[0], expected, rtol=1e-12)


def test_second_step_matches_hand_computed_predict_then_update() -> None:
    model = two_state_model()
    observations = np.array([[0.0], [3.0]])

    first = forward_filter(observations[:1], model).filtered_probabilities[0]
    prior = first @ model.transition_matrix
    likelihoods = np.array([_normal_pdf(3.0, 0.0, 1.0), _normal_pdf(3.0, 3.0, 1.0)])
    expected = (prior * likelihoods) / (prior * likelihoods).sum()

    result = forward_filter(observations, model)
    np.testing.assert_allclose(result.filtered_probabilities[1], expected, rtol=1e-12)


def test_filtered_probabilities_are_valid_distributions() -> None:
    model = two_state_model()
    result = forward_filter(regime_switching_observations(n=150, n_features=1), model)

    np.testing.assert_allclose(result.filtered_probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert (result.filtered_probabilities >= 0).all()
    assert math.isfinite(result.log_likelihood)


def test_log_likelihood_matches_forward_backward() -> None:
    """Two independent routes to log P(observations) must agree."""
    model = two_state_model()
    observations = regime_switching_observations(n=90, n_features=1)

    filtered = forward_filter(observations, model)
    _, _, smoothed_likelihood = forward_backward(
        log_emission_probabilities(observations, model), model
    )
    assert filtered.log_likelihood == pytest.approx(smoothed_likelihood, rel=1e-10)


def test_extreme_observation_does_not_underflow_the_belief() -> None:
    """An observation far in the tails must not collapse every state's
    likelihood to exactly zero and produce NaN.
    """
    model = two_state_model()
    result = forward_filter(np.array([[0.0], [40.0], [0.0]]), model)

    assert np.all(np.isfinite(result.filtered_probabilities))
    np.testing.assert_allclose(result.filtered_probabilities.sum(axis=1), 1.0, atol=1e-12)


def _one_hot_start_model() -> GaussianHMMParameters:
    """A model whose fitted start distribution is one-hot, which is what
    Baum-Welch always produces from a single training sequence -- verified
    across all 32 walk-forward folds on real data."""
    return GaussianHMMParameters(
        start_probabilities=np.array([1.0, 0.0]),
        transition_matrix=np.array([[0.7, 0.3], [0.4, 0.6]]),
        means=np.array([[0.0], [60.0]]),
        covariances=np.array([[[1.0]], [[1.0]]]),
        covariance_type="full",
    )


def test_a_one_hot_start_collapses_on_an_observation_that_state_cannot_explain() -> None:
    """The trap the default start carries, stated as a test.

    State 1 explains the observation; state 0 holds all the prior mass and
    explains it about 1,800 nats worse, which underflows ``exp`` (anything
    below about -745 does). Nothing is left to normalise.

    This is not hypothetical: fold 11 of the walk-forward hit it on
    2020-04-22 with 2,019 nats of disagreement, and it killed the run.
    """
    with pytest.raises(HMMNumericalError, match="belief collapsed"):
        forward_filter(np.array([[60.0]]), _one_hot_start_model())


def test_an_explicit_uniform_start_survives_the_same_observation() -> None:
    """Uniform keeps positive mass on every state, so the state that does
    explain the observation always survives the update. The collapse becomes
    impossible by construction rather than merely unlikely."""
    result = forward_filter(
        np.array([[60.0]]), _one_hot_start_model(), start=np.array([0.5, 0.5])
    )

    assert np.all(np.isfinite(result.filtered_probabilities))
    assert result.filtered_probabilities[0].argmax() == 1


@pytest.mark.parametrize(
    ("start", "message"),
    [
        (np.array([0.5, 0.5, 0.0]), "expected"),
        (np.array([0.5, -0.5]), "non-negative"),
        (np.array([0.5, 0.2]), "sum to 1"),
        (np.array([np.nan, 0.5]), "finite"),
    ],
)
def test_a_malformed_start_is_refused(start: np.ndarray, message: str) -> None:
    """A start that is not a distribution produces filtered "probabilities"
    that are not probabilities, and nothing downstream would notice."""
    with pytest.raises(ValueError, match=message):
        forward_filter(np.array([[0.0]]), two_state_model(), start=start)


def test_predict_step_preserves_total_probability() -> None:
    model = two_state_model()
    prior = predict_step(np.array([0.3, 0.7]), model.transition_matrix)
    assert prior.sum() == pytest.approx(1.0)


def test_filtering_an_empty_sequence_raises() -> None:
    with pytest.raises(ValueError, match="empty observation sequence"):
        forward_filter(np.empty((0, 1)), two_state_model())


def test_filtering_rejects_wrong_feature_count() -> None:
    with pytest.raises(ValueError, match="model expects"):
        forward_filter(np.zeros((5, 3)), two_state_model())


def test_filtering_rejects_nan_observations() -> None:
    """Warm-up rows must be dropped explicitly, not silently filtered over."""
    observations = np.array([[0.0], [np.nan], [1.0]])
    with pytest.raises(HMMNumericalError, match="NaN"):
        forward_filter(observations, two_state_model())


# --------------------------------------------------------------------------
# Parameter validation and invalid covariance handling
# --------------------------------------------------------------------------


def test_non_positive_definite_covariance_is_rejected_at_use() -> None:
    """An indefinite covariance has no valid density; it must fail loudly
    rather than produce meaningless numbers.
    """
    model = GaussianHMMParameters(
        start_probabilities=np.array([0.5, 0.5]),
        transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
        means=np.zeros((2, 2)),
        covariances=np.array([[[1.0, 2.0], [2.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]),
    )
    with pytest.raises(HMMNumericalError, match="positive definite"):
        log_emission_probabilities(np.zeros((3, 2)), model)


def test_non_finite_parameters_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        GaussianHMMParameters(
            start_probabilities=np.array([0.5, 0.5]),
            transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
            means=np.array([[0.0], [np.nan]]),
            covariances=np.array([[[1.0]], [[1.0]]]),
        )


def test_transition_rows_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="rows sum"):
        GaussianHMMParameters(
            start_probabilities=np.array([0.5, 0.5]),
            transition_matrix=np.array([[0.5, 0.2], [0.5, 0.5]]),
            means=np.zeros((2, 1)),
            covariances=np.array([[[1.0]], [[1.0]]]),
        )


def test_start_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="start_probabilities sum"):
        GaussianHMMParameters(
            start_probabilities=np.array([0.5, 0.2]),
            transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
            means=np.zeros((2, 1)),
            covariances=np.array([[[1.0]], [[1.0]]]),
        )


def test_condition_number_flags_a_near_singular_covariance() -> None:
    well_conditioned = np.eye(2)
    near_singular = np.array([[1.0, 0.0], [0.0, 1e-14]])

    assert covariance_condition_number(well_conditioned) == pytest.approx(1.0)
    assert covariance_condition_number(near_singular) > 1e12


def test_condition_number_of_a_singular_matrix_is_infinite() -> None:
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert math.isinf(covariance_condition_number(singular))


# --------------------------------------------------------------------------
# EM fitting
# --------------------------------------------------------------------------


def test_em_log_likelihood_is_monotonically_non_decreasing() -> None:
    """EM's defining guarantee. A violation means the E-step and M-step
    disagree -- the strongest single check that the fitting code is correct.
    """
    observations = regime_switching_observations(n=200)
    fit = fit_gaussian_hmm(observations, n_states=2, seed=11, max_iterations=40)

    trace = np.asarray(fit.log_likelihood_trace)
    assert len(trace) >= 2
    assert np.all(np.diff(trace) >= -1e-8), f"log-likelihood decreased: {np.diff(trace).min()}"


def test_same_seed_gives_bit_identical_parameters() -> None:
    observations = regime_switching_observations(n=150)
    first = fit_gaussian_hmm(observations, n_states=2, seed=5)
    second = fit_gaussian_hmm(observations, n_states=2, seed=5)

    np.testing.assert_array_equal(first.parameters.means, second.parameters.means)
    np.testing.assert_array_equal(
        first.parameters.transition_matrix, second.parameters.transition_matrix
    )
    np.testing.assert_array_equal(first.parameters.covariances, second.parameters.covariances)
    assert first.log_likelihood == second.log_likelihood
    assert first.iterations == second.iterations


def test_different_seeds_explore_different_initializations() -> None:
    """Multiple random restarts only add value if they actually differ."""
    observations = regime_switching_observations(n=150)
    first = fit_gaussian_hmm(observations, n_states=3, seed=5, max_iterations=3)
    second = fit_gaussian_hmm(observations, n_states=3, seed=9999, max_iterations=3)

    assert not np.array_equal(first.parameters.means, second.parameters.means)


def test_fit_recovers_two_separated_volatility_regimes() -> None:
    """A sanity check that fitting does something real: with two clearly
    separated volatility regimes, the fitted states' spreads should differ
    substantially.
    """
    observations = regime_switching_observations(n=400, n_features=1)
    fit = fit_gaussian_hmm(observations, n_states=2, seed=17, max_iterations=100)

    spreads = sorted(math.sqrt(fit.parameters.covariances[state][0, 0]) for state in range(2))
    assert spreads[1] > 2 * spreads[0]
    assert fit.converged


def test_hitting_the_iteration_cap_reports_non_convergence() -> None:
    observations = regime_switching_observations(n=200)
    fit = fit_gaussian_hmm(observations, n_states=3, seed=2, max_iterations=2, tolerance=1e-12)

    assert not fit.converged
    assert fit.iterations == 2


def test_converged_fit_reports_convergence() -> None:
    observations = regime_switching_observations(n=200)
    fit = fit_gaussian_hmm(observations, n_states=2, seed=2, max_iterations=500, tolerance=1e-4)

    assert fit.converged
    assert fit.iterations < 500


def test_fit_rejects_empty_observations() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fit_gaussian_hmm(np.empty((0, 2)), n_states=2, seed=1)


def test_fit_rejects_nan_observations() -> None:
    observations = regime_switching_observations(n=50)
    observations[10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_gaussian_hmm(observations, n_states=2, seed=1)


def test_fit_rejects_fewer_observations_than_states() -> None:
    with pytest.raises(ValueError, match="cannot fit"):
        fit_gaussian_hmm(np.zeros((3, 2)), n_states=5, seed=1)


def test_fit_rejects_a_single_state_model() -> None:
    with pytest.raises(ValueError, match="n_states must be >= 2"):
        fit_gaussian_hmm(regime_switching_observations(n=50), n_states=1, seed=1)


def test_fitted_parameters_are_valid_by_construction() -> None:
    fit = fit_gaussian_hmm(regime_switching_observations(n=200), n_states=3, seed=4)

    np.testing.assert_allclose(fit.parameters.transition_matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(fit.parameters.start_probabilities.sum(), 1.0)
    assert np.all(np.isfinite(fit.parameters.covariances))


def test_diagonal_fit_produces_diagonal_covariances() -> None:
    fit = fit_gaussian_hmm(
        regime_switching_observations(n=200, n_features=3),
        n_states=2,
        seed=4,
        covariance_type="diag",
    )
    for state in range(2):
        covariance = fit.parameters.covariances[state]
        off_diagonal = covariance - np.diag(np.diag(covariance))
        np.testing.assert_allclose(off_diagonal, 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# Model-selection arithmetic
# --------------------------------------------------------------------------


def test_bic_and_aic_match_their_definitions() -> None:
    assert compute_bic(-100.0, 10, 500) == pytest.approx(200.0 + 10 * math.log(500))
    assert compute_aic(-100.0, 10) == pytest.approx(200.0 + 20.0)


def test_bic_penalizes_parameters_more_than_aic() -> None:
    """BIC is the selection rule precisely because of this, for N > 7."""
    log_likelihood, n_observations = -250.0, 500
    simple = compute_bic(log_likelihood, 10, n_observations) - compute_aic(log_likelihood, 10)
    complex_ = compute_bic(log_likelihood, 40, n_observations) - compute_aic(log_likelihood, 40)
    assert complex_ > simple


def test_free_parameter_count_distinguishes_diag_from_full() -> None:
    def build(covariance_type: str) -> GaussianHMMParameters:
        states, features = 3, 4
        return GaussianHMMParameters(
            start_probabilities=np.full(states, 1 / states),
            transition_matrix=np.full((states, states), 1 / states),
            means=np.zeros((states, features)),
            covariances=np.repeat(np.eye(features)[None], states, axis=0),
            covariance_type=covariance_type,
        )

    # start (2) + transitions (6) + means (12) = 20, plus covariance terms.
    assert build("diag").free_parameter_count() == 20 + 3 * 4
    assert build("full").free_parameter_count() == 20 + 3 * 10


# --------------------------------------------------------------------------
# Derived model quantities
# --------------------------------------------------------------------------


def test_expected_duration_follows_the_self_transition_probability() -> None:
    model = GaussianHMMParameters(
        start_probabilities=np.array([0.5, 0.5]),
        transition_matrix=np.array([[0.9, 0.1], [0.5, 0.5]]),
        means=np.zeros((2, 1)),
        covariances=np.array([[[1.0]], [[1.0]]]),
    )
    durations = model.expected_durations()
    assert durations[0] == pytest.approx(10.0)
    assert durations[1] == pytest.approx(2.0)


def test_stationary_distribution_is_a_fixed_point_of_the_chain() -> None:
    model = two_state_model()
    stationary = model.stationary_distribution()

    np.testing.assert_allclose(stationary @ model.transition_matrix, stationary, atol=1e-10)
    assert stationary.sum() == pytest.approx(1.0)


def test_relabelling_produces_a_statistically_identical_model() -> None:
    """State IDs are arbitrary: permuting them must not change the model's
    likelihood or its per-observation beliefs, only which index they sit at.
    """
    model = two_state_model()
    observations = regime_switching_observations(n=100, n_features=1)
    order = np.array([1, 0])

    original = forward_filter(observations, model)
    permuted = forward_filter(observations, model.relabel(order))

    assert permuted.log_likelihood == pytest.approx(original.log_likelihood, rel=1e-12)
    np.testing.assert_allclose(
        permuted.filtered_probabilities, original.filtered_probabilities[:, order], atol=1e-12
    )


def test_relabel_rejects_a_non_permutation() -> None:
    with pytest.raises(ValueError, match="permutation"):
        two_state_model().relabel(np.array([0, 0]))
