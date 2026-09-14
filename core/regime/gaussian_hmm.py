"""Gaussian HMM mathematics: parameters, emissions, filtering, and fitting.

This module knows nothing about market regimes, risk budgets, or trading. It
is pure inference machinery, deliberately separated from
``core/regime/hmm_engine.py`` so that the single most correctness-critical
routine in the system -- the forward filter -- can be read, reasoned about and
tested with no strategy context attached.

## Filtered inference, never smoothed, never Viterbi

The live "what regime are we in right now" question must be answered by

    P(state_t | observations_1..t)

and nothing else. Two standard HMM routines would silently answer a different
question:

- **Smoothing** (forward-backward, the ``gamma`` posterior) computes
  ``P(state_t | observations_1..T)`` using the *whole* sequence, including
  observations after ``t``. Many libraries expose exactly this as
  ``predict_proba``. In a backtest it is a look-ahead leak that flatters every
  regime call: the model "knows" how the episode turned out.
- **Viterbi** computes the single most likely *path*, which is also a
  full-sequence quantity -- appending tomorrow's observation can retroactively
  change which state yesterday is assigned to.

:func:`forward_filter` therefore exists as a dedicated implementation, built
from :func:`filter_step`, which by construction takes only the previous belief
and one new observation. There is no code path in this module by which a
future observation can reach a past belief; that is a structural property, not
a convention, and ``tests/unit/test_gaussian_hmm.py`` asserts it behaviorally
by appending future data and checking earlier output is bit-identical.

Smoothing *is* implemented (:func:`forward_backward`) because Baum-Welch
training requires it -- but it is training-only, and named so it cannot be
mistaken for the live path.

## Numerical approach

Emissions are evaluated in log space. The forward recursion runs in linear
space with per-step normalization (the standard scaled algorithm, matching
docs/SPECIFICATION.md section 6.2's pseudocode), with each step's maximum log
emission factored out before exponentiation. That maximum cancels in the
normalization and is accumulated into the log-likelihood, so an observation
deep in the tails cannot underflow the whole belief vector to zeros.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_LOG_2PI = math.log(2.0 * math.pi)


class HMMNumericalError(RuntimeError):
    """A fit or filter step produced numerically unusable values.

    Raised rather than returning NaN so a caller cannot accidentally trade on
    a belief vector that silently became meaningless.
    """


@dataclass(frozen=True)
class GaussianHMMParameters:
    """A fitted Gaussian HMM: everything the forward filter needs, and nothing
    else.

    Covariances are always stored as full ``(n_states, n_features,
    n_features)`` matrices even when fitted diagonally, so inference has one
    code path. ``covariance_type`` records how they were *fitted*, which is
    what the free-parameter count (and therefore BIC) depends on.
    """

    start_probabilities: NDArray[np.float64]
    transition_matrix: NDArray[np.float64]
    means: NDArray[np.float64]
    covariances: NDArray[np.float64]
    covariance_type: str = "full"

    def __post_init__(self) -> None:
        n_states, n_features = self.means.shape
        if self.start_probabilities.shape != (n_states,):
            raise ValueError(
                f"start_probabilities has shape {self.start_probabilities.shape}, "
                f"expected ({n_states},)"
            )
        if self.transition_matrix.shape != (n_states, n_states):
            raise ValueError(
                f"transition_matrix has shape {self.transition_matrix.shape}, "
                f"expected ({n_states}, {n_states})"
            )
        if self.covariances.shape != (n_states, n_features, n_features):
            raise ValueError(
                f"covariances has shape {self.covariances.shape}, "
                f"expected ({n_states}, {n_features}, {n_features})"
            )
        if self.covariance_type not in ("diag", "full"):
            raise ValueError(
                f"covariance_type must be 'diag' or 'full', got {self.covariance_type!r}"
            )
        for name, array in (
            ("start_probabilities", self.start_probabilities),
            ("transition_matrix", self.transition_matrix),
            ("means", self.means),
            ("covariances", self.covariances),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")
        if not math.isclose(float(self.start_probabilities.sum()), 1.0, abs_tol=1e-8):
            raise ValueError(
                f"start_probabilities sum to {self.start_probabilities.sum()}, expected 1.0"
            )
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError(f"transition_matrix rows sum to {row_sums}, expected all 1.0")
        if np.any(self.start_probabilities < 0) or np.any(self.transition_matrix < 0):
            raise ValueError("probabilities must be non-negative")

    @property
    def n_states(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])

    def free_parameter_count(self) -> int:
        """Number of free parameters, for BIC/AIC.

        Start probabilities and each transition row are simplexes, so each
        contributes one fewer free parameter than it has entries.
        """
        states, features = self.n_states, self.n_features
        start = states - 1
        transitions = states * (states - 1)
        means = states * features
        if self.covariance_type == "diag":
            covariances = states * features
        else:
            covariances = states * features * (features + 1) // 2
        return start + transitions + means + covariances

    def stationary_distribution(self) -> NDArray[np.float64]:
        """Long-run state occupancy implied by the transition matrix.

        Computed as the normalized left eigenvector for eigenvalue 1. Used for
        reporting; empirical occupancy from the training data is the number
        that state statistics actually report.
        """
        values, vectors = np.linalg.eig(self.transition_matrix.T)
        index = int(np.argmin(np.abs(values - 1.0)))
        vector = np.real(vectors[:, index])
        total = vector.sum()
        if not np.isfinite(total) or math.isclose(total, 0.0, abs_tol=1e-12):
            raise HMMNumericalError("transition matrix has no usable stationary distribution")
        distribution = vector / total
        return np.clip(distribution, 0.0, None) / np.clip(distribution, 0.0, None).sum()

    def expected_durations(self) -> NDArray[np.float64]:
        """Expected sojourn time in each state, ``1 / (1 - A_ii)`` sessions.

        A state's persistence is a property of the fitted transition matrix,
        not of what anyone chose to call it -- this is one of the measured
        characteristics that policy is allowed to act on.
        """
        self_transitions = np.diag(self.transition_matrix)
        escape = np.clip(1.0 - self_transitions, 1e-12, None)
        return 1.0 / escape

    def relabel(self, order: NDArray[np.int_]) -> GaussianHMMParameters:
        """Return an equivalent model with states permuted into ``order``.

        ``order[k]`` is the old index of the state that becomes new state
        ``k``. The permuted model is statistically identical -- state IDs are
        arbitrary (docs/SPECIFICATION.md section 6.1) -- which is exactly why
        nothing downstream may attach meaning to a raw state ID.
        """
        order = np.asarray(order, dtype=int)
        if sorted(order.tolist()) != list(range(self.n_states)):
            raise ValueError(f"order must be a permutation of 0..{self.n_states - 1}, got {order}")
        return GaussianHMMParameters(
            start_probabilities=self.start_probabilities[order],
            transition_matrix=self.transition_matrix[np.ix_(order, order)],
            means=self.means[order],
            covariances=self.covariances[order],
            covariance_type=self.covariance_type,
        )


@dataclass(frozen=True)
class FilterResult:
    """Output of :func:`forward_filter`."""

    filtered_probabilities: NDArray[np.float64]
    """``(T, n_states)``; row ``t`` is ``P(state_t | observations_1..t)``."""

    log_likelihood: float
    """``log P(observations_1..T)`` under the model."""


@dataclass(frozen=True)
class EMFitResult:
    """Output of one Baum-Welch run (training only)."""

    parameters: GaussianHMMParameters
    log_likelihood: float
    converged: bool
    iterations: int
    log_likelihood_trace: tuple[float, ...]
    state_responsibilities: NDArray[np.float64]
    """Smoothed ``gamma`` over the training window. Training-only: these use
    the whole training sequence and must never be used for a live decision."""


# --------------------------------------------------------------------------
# Emissions
# --------------------------------------------------------------------------


def _cholesky_diagnostics(
    covariances: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-state precision matrices and log-determinants.

    Uses Cholesky as the positive-definiteness check: a covariance that is not
    positive definite fails here rather than producing a silently meaningless
    density later.
    """
    n_states = covariances.shape[0]
    precisions = np.empty_like(covariances)
    log_determinants = np.empty(n_states, dtype=np.float64)
    for state in range(n_states):
        matrix = covariances[state]
        try:
            factor = np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as exc:
            raise HMMNumericalError(
                f"covariance for state {state} is not positive definite: {exc}"
            ) from exc
        diagonal = np.diag(factor)
        if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
            raise HMMNumericalError(
                f"covariance for state {state} has a degenerate Cholesky factor"
            )
        log_determinants[state] = 2.0 * float(np.log(diagonal).sum())
        precisions[state] = np.linalg.inv(matrix)
    return precisions, log_determinants


def log_emission_probabilities(
    observations: NDArray[np.float64], parameters: GaussianHMMParameters
) -> NDArray[np.float64]:
    """``(T, n_states)`` matrix of ``log N(x_t; mu_i, Sigma_i)``.

    Each row depends only on that row's observation, so this function cannot
    mix information across time.
    """
    observations = np.atleast_2d(np.asarray(observations, dtype=np.float64))
    if observations.shape[1] != parameters.n_features:
        raise ValueError(
            f"observations have {observations.shape[1]} features, "
            f"model expects {parameters.n_features}"
        )
    if not np.all(np.isfinite(observations)):
        raise HMMNumericalError(
            "observations contain NaN or infinity; feature warm-up rows must be "
            "dropped before inference rather than passed through"
        )

    precisions, log_determinants = _cholesky_diagnostics(parameters.covariances)
    deviations = observations[:, None, :] - parameters.means[None, :, :]
    quadratic = np.einsum("tsd,sde,tse->ts", deviations, precisions, deviations)
    log_probabilities = -0.5 * (
        parameters.n_features * _LOG_2PI + log_determinants[None, :] + quadratic
    )
    return np.asarray(log_probabilities, dtype=np.float64)


def covariance_condition_number(covariance: NDArray[np.float64]) -> float:
    """Condition number of one covariance matrix.

    A large value means the fitted state is nearly singular in some direction
    -- the density is concentrated on a lower-dimensional subspace, and small
    numerical differences swing the likelihood wildly. Used to reject such
    fits (docs/SPECIFICATION.md section 6.3).
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        return math.inf
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    smallest = float(singular_values[-1])
    if smallest <= 0:
        return math.inf
    return float(singular_values[0]) / smallest


# --------------------------------------------------------------------------
# Filtered inference -- the live decision path
# --------------------------------------------------------------------------


def predict_step(
    belief: NDArray[np.float64], transition_matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Propagate a belief one session forward: ``P(state_t | obs_1..t-1)``."""
    return np.asarray(belief, dtype=np.float64) @ transition_matrix


def update_step(
    prior: NDArray[np.float64], log_emission: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Fold one observation into a predicted belief (Bayes update).

    Returns the posterior and this step's log-likelihood contribution. The
    maximum log emission is factored out before exponentiating; it cancels in
    the normalization and is added back into the log-likelihood, so a
    far-tail observation cannot underflow the posterior to all zeros.
    """
    offset = float(np.max(log_emission))
    if not math.isfinite(offset):
        raise HMMNumericalError("emission log-probabilities are not finite")
    weighted = np.asarray(prior, dtype=np.float64) * np.exp(log_emission - offset)
    total = float(weighted.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise HMMNumericalError(
            "belief collapsed to zero probability mass; the observation is "
            "impossible under every state of this model"
        )
    return weighted / total, offset + math.log(total)


def filter_step(
    previous_posterior: NDArray[np.float64],
    observation: NDArray[np.float64],
    parameters: GaussianHMMParameters,
) -> tuple[NDArray[np.float64], float]:
    """One complete filtered step: predict, then update with one observation.

    This is the live inference primitive. Its signature is the guarantee: it
    accepts yesterday's belief and today's observation, and there is no
    argument through which tomorrow's data could enter.
    """
    prior = predict_step(previous_posterior, parameters.transition_matrix)
    log_emission = log_emission_probabilities(np.atleast_2d(observation), parameters)[0]
    return update_step(prior, log_emission)


def forward_filter(
    observations: NDArray[np.float64], parameters: GaussianHMMParameters
) -> FilterResult:
    """Run filtered inference over a sequence: ``P(state_t | obs_1..t)`` for
    every ``t``.

    Row ``t`` of the result is a function of rows ``0..t`` only, so truncating
    the input after ``t`` leaves that row unchanged, bit for bit.
    """
    matrix = np.atleast_2d(np.asarray(observations, dtype=np.float64))
    if matrix.size == 0 or matrix.shape[0] == 0:
        raise ValueError("cannot filter an empty observation sequence")

    log_emissions = log_emission_probabilities(matrix, parameters)
    n_observations = matrix.shape[0]
    filtered = np.empty((n_observations, parameters.n_states), dtype=np.float64)

    belief, log_likelihood = update_step(parameters.start_probabilities, log_emissions[0])
    filtered[0] = belief
    for index in range(1, n_observations):
        prior = predict_step(belief, parameters.transition_matrix)
        belief, increment = update_step(prior, log_emissions[index])
        filtered[index] = belief
        log_likelihood += increment

    return FilterResult(filtered_probabilities=filtered, log_likelihood=log_likelihood)


# --------------------------------------------------------------------------
# Smoothing and fitting -- training only
# --------------------------------------------------------------------------


def forward_backward(
    log_emissions: NDArray[np.float64], parameters: GaussianHMMParameters
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Smoothed posteriors for Baum-Welch. **Training only.**

    Returns ``(gamma, xi_sum, log_likelihood)`` where ``gamma[t]`` is
    ``P(state_t | observations_1..T)`` -- conditioned on the *entire*
    sequence. Correct for parameter estimation over a closed training window;
    never valid as a live regime call, which is what
    :func:`forward_filter` is for.
    """
    n_observations, n_states = log_emissions.shape
    offsets = log_emissions.max(axis=1)
    scaled_emissions = np.exp(log_emissions - offsets[:, None])

    alpha = np.empty((n_observations, n_states), dtype=np.float64)
    scales = np.empty(n_observations, dtype=np.float64)

    unnormalized = parameters.start_probabilities * scaled_emissions[0]
    scales[0] = float(unnormalized.sum())
    if scales[0] <= 0.0:
        raise HMMNumericalError("forward pass collapsed at the first observation")
    alpha[0] = unnormalized / scales[0]

    for index in range(1, n_observations):
        unnormalized = (alpha[index - 1] @ parameters.transition_matrix) * scaled_emissions[index]
        scales[index] = float(unnormalized.sum())
        if scales[index] <= 0.0:
            raise HMMNumericalError(f"forward pass collapsed at observation {index}")
        alpha[index] = unnormalized / scales[index]

    beta = np.ones((n_observations, n_states), dtype=np.float64)
    for index in range(n_observations - 2, -1, -1):
        beta[index] = (
            parameters.transition_matrix
            @ (scaled_emissions[index + 1] * beta[index + 1])
            / scales[index + 1]
        )

    gamma = alpha * beta
    row_sums = gamma.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise HMMNumericalError("smoothed posteriors collapsed to zero")
    gamma = gamma / row_sums

    xi_sum = np.zeros((n_states, n_states), dtype=np.float64)
    for index in range(n_observations - 1):
        xi_sum += (
            np.outer(alpha[index], scaled_emissions[index + 1] * beta[index + 1])
            * parameters.transition_matrix
            / scales[index + 1]
        )

    log_likelihood = float(np.log(scales).sum() + offsets.sum())
    return gamma, xi_sum, log_likelihood


def _initial_parameters(
    observations: NDArray[np.float64],
    n_states: int,
    seed: int,
    covariance_type: str,
    regularization: float,
) -> GaussianHMMParameters:
    """Seeded initialization: distinct observations as means, the pooled
    covariance for every state, and a self-transition-biased chain.

    Fully determined by ``seed``, so a training run is reproducible.
    """
    generator = np.random.default_rng(seed)
    n_observations, n_features = observations.shape
    indices = generator.choice(n_observations, size=n_states, replace=False)
    means = observations[indices].copy()

    pooled = np.cov(observations, rowvar=False)
    pooled = np.atleast_2d(pooled) + regularization * np.eye(n_features)
    if covariance_type == "diag":
        pooled = np.diag(np.diag(pooled))
    covariances = np.repeat(pooled[None, :, :], n_states, axis=0)

    self_transition = 0.8
    off_diagonal = (1.0 - self_transition) / (n_states - 1)
    transition_matrix = np.full((n_states, n_states), off_diagonal, dtype=np.float64)
    np.fill_diagonal(transition_matrix, self_transition)

    return GaussianHMMParameters(
        start_probabilities=np.full(n_states, 1.0 / n_states, dtype=np.float64),
        transition_matrix=transition_matrix,
        means=means,
        covariances=covariances,
        covariance_type=covariance_type,
    )


def fit_gaussian_hmm(
    observations: NDArray[np.float64],
    n_states: int,
    seed: int,
    *,
    covariance_type: str = "diag",
    max_iterations: int = 200,
    tolerance: float = 1e-4,
    regularization: float = 1e-6,
) -> EMFitResult:
    """Fit by Baum-Welch (EM) from a seeded initialization.

    The returned ``log_likelihood_trace`` is monotonically non-decreasing by
    construction of EM; ``tests/unit/test_gaussian_hmm.py`` asserts that,
    which is a strong check that the E- and M-steps are mutually consistent.

    ``converged`` is False when the iteration cap was reached before the
    log-likelihood gain fell below ``tolerance``. Callers should treat a
    non-converged fit as unusable rather than as "good enough"
    (``core.regime.hmm_engine`` rejects it during selection).
    """
    matrix = np.asarray(observations, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("observations must be a non-empty 2-D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("observations contain NaN or infinity")
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states}")
    if matrix.shape[0] < n_states:
        raise ValueError(
            f"cannot fit {n_states} states to {matrix.shape[0]} observations"
        )

    parameters = _initial_parameters(matrix, n_states, seed, covariance_type, regularization)
    n_observations, n_features = matrix.shape
    identity = np.eye(n_features)

    trace: list[float] = []
    previous = -math.inf
    converged = False
    iterations = 0
    gamma = np.full((n_observations, n_states), 1.0 / n_states, dtype=np.float64)

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        log_emissions = log_emission_probabilities(matrix, parameters)
        gamma, xi_sum, log_likelihood = forward_backward(log_emissions, parameters)
        trace.append(log_likelihood)

        if log_likelihood - previous < tolerance and iteration > 1:
            converged = True
            previous = log_likelihood
            break
        previous = log_likelihood

        weights = gamma.sum(axis=0)
        safe_weights = np.clip(weights, 1e-12, None)

        start_probabilities = gamma[0] / gamma[0].sum()
        transition_matrix = xi_sum / np.clip(xi_sum.sum(axis=1, keepdims=True), 1e-12, None)
        means = (gamma.T @ matrix) / safe_weights[:, None]

        covariances = np.empty((n_states, n_features, n_features), dtype=np.float64)
        for state in range(n_states):
            deviations = matrix - means[state]
            weighted = deviations * gamma[:, state : state + 1]
            covariance = (weighted.T @ deviations) / safe_weights[state]
            if covariance_type == "diag":
                covariance = np.diag(np.diag(covariance))
            covariances[state] = covariance + regularization * identity

        parameters = GaussianHMMParameters(
            start_probabilities=start_probabilities,
            transition_matrix=transition_matrix,
            means=means,
            covariances=covariances,
            covariance_type=covariance_type,
        )

    return EMFitResult(
        parameters=parameters,
        log_likelihood=previous,
        converged=converged,
        iterations=iterations,
        log_likelihood_trace=tuple(trace),
        state_responsibilities=gamma,
    )


def compute_bic(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    """``-2 logL + k ln(N)``. Lower is better.

    Used for state-count selection on the *training* window only
    (docs/SPECIFICATION.md section 10) -- selecting on out-of-sample data
    would make the OOS window part of model choice and stop it being
    out-of-sample.
    """
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    return -2.0 * log_likelihood + n_parameters * math.log(n_observations)


def compute_aic(log_likelihood: float, n_parameters: int) -> float:
    """``-2 logL + 2k``. Reported alongside BIC; BIC is the selection rule
    because it penalizes parameters harder, which is the right bias for a
    model that must generalize out of sample.
    """
    return -2.0 * log_likelihood + 2.0 * n_parameters
