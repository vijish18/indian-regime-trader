"""Gaussian HMM market-regime classifier.

The engine's entire job is one arrow:

    market observations -> market risk/volatility regime -> portfolio risk budget

It classifies *how risky the market currently is*. It does not emit buy or
sell signals, does not rank securities, and does not know what the portfolio
holds. Security selection is a separate layer
(``universe/stock_selector.py``); translating a regime into a gross-exposure
budget is another (``core/regime/regime_policy.py``). Keeping the HMM this
narrow is what lets a walk-forward run measure whether the regime layer adds
anything at all (docs/SPECIFICATION.md section 1.2).

## Labels are reporting; statistics are the decision input

A fitted HMM's state IDs are arbitrary: refit with a different seed and
"state 0" may be a different regime. Worse, a *label* like "crisis" is a name
someone chose, and naming is not measurement. So:

- :class:`StateStatistics` carries measured quantities -- annualized expected
  volatility, expected return, downside volatility, empirical occupancy,
  expected duration, self-transition probability -- computed from the training
  window.
- :class:`RegimeLabel` is assigned afterwards by *ranking states on measured
  expected volatility*, and exists for dashboards and audit logs.
- :class:`RegimeState` (what inference returns) carries both, and downstream
  policy is expected to act on ``expected_volatility`` and friends.

Two states can legitimately receive the same label when a model has more
states than labels; that is fine precisely because labels do not drive
behavior. ``tests/unit/test_hmm_engine.py`` asserts that permuting state IDs
leaves every risk-relevant output unchanged.

## Filtered inference only

Live regime calls go through :func:`core.regime.gaussian_hmm.forward_filter`,
which computes ``P(state_t | observations_1..t)``. Full-sequence Viterbi and
smoothed (forward-backward) posteriors are never used for a decision -- both
would let tomorrow's data change today's answer. See that module's docstring
for why this distinction is easy to get wrong and expensive when you do.

## Selection and validation gates

``fit`` tries every (candidate state count x random seed) pair, then rejects
candidates that fail the gates from docs/SPECIFICATION.md section 6.3 --
non-convergence, degenerate (near-empty) states, near-singular covariance --
before choosing the survivor with the lowest BIC. If nothing survives it
raises: an unvalidated regime model is worse than no regime model, because it
would silently size positions from noise.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from config.models import HMMConfig
from core.features.feature_engineering import TRADING_DAYS_PER_YEAR
from core.regime.gaussian_hmm import (
    EMFitResult,
    FilterResult,
    GaussianHMMParameters,
    HMMNumericalError,
    compute_aic,
    compute_bic,
    covariance_condition_number,
    fit_gaussian_hmm,
    forward_filter,
)


class RegimeLabel(StrEnum):
    """Human-readable regime label, assigned post-hoc from measured
    characteristics (docs/SPECIFICATION.md section 6.1) -- never from a raw,
    arbitrary state ID, and never used to decide behavior.
    """

    CALM = "calm"
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"


ORDERED_LABELS: tuple[RegimeLabel, ...] = (
    RegimeLabel.CALM,
    RegimeLabel.NORMAL,
    RegimeLabel.ELEVATED,
    RegimeLabel.CRISIS,
)
"""Labels in ascending order of market risk. The ordering is the only thing
that matters: states are ranked by measured volatility and mapped onto this
scale."""


class ModelSelectionError(RuntimeError):
    """No candidate model passed the validation gates.

    Deliberately fatal: the system fails closed rather than trading on a model
    that converged to a degenerate or numerically unstable fit.
    """


class InsufficientHistoryError(ValueError):
    """The training window has too few observations for the requested model."""


@dataclass(frozen=True)
class StateStatistics:
    """Measured characteristics of one fitted state.

    These are empirical statistics of the *actual market return series*,
    weighted by the state's responsibility over the training window -- not
    properties of the standardized feature space, which would be
    uninterpretable. They are what policy is allowed to act on.
    """

    state_id: int
    expected_return: float
    """Annualized mean return while in this state."""

    expected_volatility: float
    """Annualized standard deviation of returns while in this state -- the
    primary risk measure, and the quantity states are ranked by."""

    downside_volatility: float
    """Annualized standard deviation of negative returns only."""

    occupancy: float
    """Share of the training window spent in this state."""

    expected_duration: float
    """Expected consecutive sessions in this state, ``1 / (1 - A_ii)``."""

    self_transition_probability: float
    label: RegimeLabel
    """Reporting only. Never a decision input."""


@dataclass(frozen=True)
class RegimeState:
    """The output of one filtered-inference step.

    ``label`` is for humans; ``expected_volatility``, ``expected_return`` and
    ``persistence`` are the measured quantities a risk policy should consume.
    """

    as_of: dt.date
    state_id: int
    label: RegimeLabel
    probabilities: tuple[float, ...]
    confidence: float
    expected_volatility: float
    expected_return: float
    persistence: float

    @property
    def is_confident(self) -> bool:
        """Whether the filter is decisive. Compare against
        ``hmm.min_confidence``; an unconfident call should fall back to the
        more conservative regime rather than being acted on as certain.
        """
        return self.confidence >= 0.5


@dataclass(frozen=True)
class HMMTrainingResult:
    """Everything about one fitted candidate, including why it was or was not
    usable. Retained for every candidate so a selection decision can be
    audited after the fact.
    """

    n_states: int
    covariance_type: str
    seed: int
    bic: float
    aic: float
    log_likelihood: float
    converged: bool
    iterations: int
    training_start: dt.date
    training_end: dt.date
    n_observations: int
    feature_columns: tuple[str, ...]
    max_condition_number: float
    min_occupancy: float
    rejection_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.rejection_reason is None


@dataclass(frozen=True)
class FittedRegimeModel:
    """A validated, selected model plus the statistics that describe it."""

    parameters: GaussianHMMParameters
    statistics: tuple[StateStatistics, ...]
    training_result: HMMTrainingResult
    candidates: tuple[HMMTrainingResult, ...] = field(default_factory=tuple)

    def statistics_for(self, state_id: int) -> StateStatistics:
        for statistic in self.statistics:
            if statistic.state_id == state_id:
                return statistic
        raise KeyError(f"no statistics for state {state_id}")

    @property
    def n_states(self) -> int:
        return self.parameters.n_states


class HMMRegimeEngine:
    """Fits and runs forward-filtered inference for a Gaussian HMM over
    market-level features.

    Training uses only data inside the supplied window; inference uses forward
    filtering only, so a regime call for session ``t`` never changes when
    later sessions arrive.
    """

    def __init__(self, config: HMMConfig) -> None:
        self.config = config

    # -- training ---------------------------------------------------------

    def fit(self, features: pd.DataFrame, returns: pd.Series) -> FittedRegimeModel:
        """Fit every candidate, validate, and select the best by BIC.

        Args:
            features: causal, already-scaled market features indexed by
                session date (see core/features/). Must contain no NaN --
                drop warm-up rows with
                ``core.features.feature_engineering.drop_warmup_rows`` first.
            returns: daily market returns over the same index, used *only* to
                describe the resulting states in interpretable units. They do
                not train the model; the features do.

        Raises:
            InsufficientHistoryError: if the window is empty or too short.
            ModelSelectionError: if no candidate passes the validation gates.
        """
        observations, aligned_returns = self._validate_training_inputs(features, returns)
        training_start = _as_date(features.index[0])
        training_end = _as_date(features.index[-1])
        feature_columns = tuple(str(column) for column in features.columns)

        candidates: list[HMMTrainingResult] = []
        accepted: list[tuple[HMMTrainingResult, EMFitResult]] = []

        for n_states in self.config.candidate_states:
            if not self._has_enough_history(observations, n_states):
                candidates.append(
                    self._rejected_result(
                        n_states,
                        seed=0,
                        training_start=training_start,
                        training_end=training_end,
                        n_observations=observations.shape[0],
                        feature_columns=feature_columns,
                        reason=(
                            f"insufficient history: {observations.shape[0]} observations "
                            f"cannot support {n_states} states"
                        ),
                    )
                )
                continue

            for seed in self.config.random_seeds:
                result, fit = self._fit_candidate(
                    observations,
                    n_states=n_states,
                    seed=seed,
                    training_start=training_start,
                    training_end=training_end,
                    feature_columns=feature_columns,
                )
                candidates.append(result)
                if result.accepted and fit is not None:
                    accepted.append((result, fit))

        if not accepted:
            raise ModelSelectionError(
                "no candidate model passed validation. Rejections: "
                + "; ".join(
                    f"{candidate.n_states} states/seed {candidate.seed}: "
                    f"{candidate.rejection_reason}"
                    for candidate in candidates
                )
            )

        best_result, best_fit = min(accepted, key=lambda pair: pair[0].bic)
        statistics = characterize_states(
            best_fit.parameters, best_fit.state_responsibilities, aligned_returns
        )
        return FittedRegimeModel(
            parameters=best_fit.parameters,
            statistics=statistics,
            training_result=best_result,
            candidates=tuple(candidates),
        )

    def _fit_candidate(
        self,
        observations: NDArray[np.float64],
        *,
        n_states: int,
        seed: int,
        training_start: dt.date,
        training_end: dt.date,
        feature_columns: tuple[str, ...],
    ) -> tuple[HMMTrainingResult, EMFitResult | None]:
        try:
            fit = fit_gaussian_hmm(
                observations,
                n_states=n_states,
                seed=seed,
                covariance_type=self.config.covariance_type,
                max_iterations=self.config.max_iterations,
                tolerance=self.config.convergence_tolerance,
                regularization=self.config.covariance_regularization,
            )
        except (HMMNumericalError, ValueError) as exc:
            return (
                self._rejected_result(
                    n_states,
                    seed=seed,
                    training_start=training_start,
                    training_end=training_end,
                    n_observations=observations.shape[0],
                    feature_columns=feature_columns,
                    reason=f"fit failed: {exc}",
                ),
                None,
            )

        occupancy = fit.state_responsibilities.mean(axis=0)
        conditions = [
            covariance_condition_number(fit.parameters.covariances[state])
            for state in range(n_states)
        ]
        max_condition = max(conditions)
        min_occupancy = float(occupancy.min())

        n_parameters = fit.parameters.free_parameter_count()
        result = HMMTrainingResult(
            n_states=n_states,
            covariance_type=self.config.covariance_type,
            seed=seed,
            bic=compute_bic(fit.log_likelihood, n_parameters, observations.shape[0]),
            aic=compute_aic(fit.log_likelihood, n_parameters),
            log_likelihood=fit.log_likelihood,
            converged=fit.converged,
            iterations=fit.iterations,
            training_start=training_start,
            training_end=training_end,
            n_observations=observations.shape[0],
            feature_columns=feature_columns,
            max_condition_number=max_condition,
            min_occupancy=min_occupancy,
            rejection_reason=self._rejection_reason(fit, max_condition, min_occupancy),
        )
        return result, fit

    def _rejection_reason(
        self, fit: EMFitResult, max_condition: float, min_occupancy: float
    ) -> str | None:
        """Apply the validation gates from docs/SPECIFICATION.md section 6.3."""
        if not fit.converged:
            return f"did not converge within {fit.iterations} iterations"
        if min_occupancy < self.config.min_state_occupancy:
            return (
                f"degenerate state: lowest occupancy {min_occupancy:.4f} is below "
                f"{self.config.min_state_occupancy}"
            )
        if max_condition > self.config.max_covariance_condition_number:
            return (
                f"near-singular covariance: condition number {max_condition:.3e} exceeds "
                f"{self.config.max_covariance_condition_number:.3e}"
            )
        if not math.isfinite(fit.log_likelihood):
            return "non-finite log-likelihood"
        return None

    def _rejected_result(
        self,
        n_states: int,
        *,
        seed: int,
        training_start: dt.date,
        training_end: dt.date,
        n_observations: int,
        feature_columns: tuple[str, ...],
        reason: str,
    ) -> HMMTrainingResult:
        return HMMTrainingResult(
            n_states=n_states,
            covariance_type=self.config.covariance_type,
            seed=seed,
            bic=math.inf,
            aic=math.inf,
            log_likelihood=-math.inf,
            converged=False,
            iterations=0,
            training_start=training_start,
            training_end=training_end,
            n_observations=n_observations,
            feature_columns=feature_columns,
            max_condition_number=math.inf,
            min_occupancy=0.0,
            rejection_reason=reason,
        )

    def _validate_training_inputs(
        self, features: pd.DataFrame, returns: pd.Series
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if features.empty:
            raise InsufficientHistoryError(
                "cannot fit a regime model on an empty training window"
            )
        if features.isna().to_numpy().any():
            raise InsufficientHistoryError(
                "training features contain NaN; drop warm-up rows with "
                "core.features.feature_engineering.drop_warmup_rows before fitting"
            )
        aligned = returns.reindex(features.index)
        if aligned.isna().any():
            raise InsufficientHistoryError(
                "returns do not cover every training session; state statistics would "
                "otherwise be computed from a different sample than the model was fit on"
            )
        return (
            features.to_numpy(dtype=np.float64),
            aligned.to_numpy(dtype=np.float64),
        )

    def _has_enough_history(self, observations: NDArray[np.float64], n_states: int) -> bool:
        """A model may not have more free parameters than observations, and
        needs at least two observations per state to estimate any spread.
        """
        n_observations, n_features = observations.shape
        probe = GaussianHMMParameters(
            start_probabilities=np.full(n_states, 1.0 / n_states),
            transition_matrix=np.full((n_states, n_states), 1.0 / n_states),
            means=np.zeros((n_states, n_features)),
            covariances=np.repeat(np.eye(n_features)[None, :, :], n_states, axis=0),
            covariance_type=self.config.covariance_type,
        )
        return bool(
            n_observations >= 2 * n_states
            and n_observations > probe.free_parameter_count()
        )

    # -- inference --------------------------------------------------------

    def filter(self, model: FittedRegimeModel, features: pd.DataFrame) -> list[RegimeState]:
        """Run forward-only filtered inference over ``features``.

        The regime returned for row ``t`` uses observations ``1..t`` only.
        Appending later rows and re-running leaves every earlier row
        identical -- asserted directly in tests, because this is the property
        the whole design rests on.
        """
        if features.empty:
            raise ValueError("cannot filter an empty feature window")
        expected = list(model.training_result.feature_columns)
        actual = [str(column) for column in features.columns]
        if actual != expected:
            raise ValueError(
                f"feature columns {actual} do not match the model's training columns "
                f"{expected}; a model may only be applied to the features it was trained on"
            )

        result = forward_filter(features.to_numpy(dtype=np.float64), model.parameters)
        return self._to_regime_states(model, features.index, result)

    def filter_latest(self, model: FittedRegimeModel, features: pd.DataFrame) -> RegimeState:
        """The current regime: the last row of :meth:`filter`.

        Provided so live code never has to index into a sequence and risk
        picking the wrong row.
        """
        return self.filter(model, features)[-1]

    def _to_regime_states(
        self, model: FittedRegimeModel, index: pd.Index, result: FilterResult
    ) -> list[RegimeState]:
        states: list[RegimeState] = []
        for position, timestamp in enumerate(index):
            probabilities = result.filtered_probabilities[position]
            state_id = int(np.argmax(probabilities))
            statistic = model.statistics_for(state_id)
            states.append(
                RegimeState(
                    as_of=_as_date(timestamp),
                    state_id=state_id,
                    label=statistic.label,
                    probabilities=tuple(float(value) for value in probabilities),
                    confidence=float(probabilities[state_id]),
                    expected_volatility=statistic.expected_volatility,
                    expected_return=statistic.expected_return,
                    persistence=statistic.self_transition_probability,
                )
            )
        return states


# --------------------------------------------------------------------------
# State characterization
# --------------------------------------------------------------------------


def characterize_states(
    parameters: GaussianHMMParameters,
    responsibilities: NDArray[np.float64],
    returns: NDArray[np.float64],
) -> tuple[StateStatistics, ...]:
    """Describe each fitted state in interpretable, measured terms.

    Statistics are probability-weighted over the training window: each
    session contributes to every state in proportion to that state's
    responsibility for it, so a state's "expected volatility" is the
    volatility of the market *while that state was in force*, not a property
    of the standardized feature space.

    Uses smoothed training responsibilities deliberately: this is a
    description of a closed historical window, computed once at training
    time, not a live inference call.
    """
    if responsibilities.shape[0] != returns.shape[0]:
        raise ValueError(
            f"responsibilities cover {responsibilities.shape[0]} sessions but returns "
            f"cover {returns.shape[0]}"
        )

    durations = parameters.expected_durations()
    self_transitions = np.diag(parameters.transition_matrix)
    annualization = math.sqrt(TRADING_DAYS_PER_YEAR)

    measured: list[dict[str, float]] = []
    for state in range(parameters.n_states):
        weights = responsibilities[:, state]
        total = float(weights.sum())
        if total <= 0:
            mean = variance = downside = 0.0
        else:
            mean = float((weights * returns).sum() / total)
            variance = float((weights * (returns - mean) ** 2).sum() / total)
            shortfall = np.minimum(returns, 0.0)
            downside = float((weights * shortfall**2).sum() / total)
        measured.append(
            {
                "expected_return": mean * TRADING_DAYS_PER_YEAR,
                "expected_volatility": math.sqrt(max(variance, 0.0)) * annualization,
                "downside_volatility": math.sqrt(max(downside, 0.0)) * annualization,
                "occupancy": float(weights.mean()),
            }
        )

    labels = assign_labels([entry["expected_volatility"] for entry in measured])
    return tuple(
        StateStatistics(
            state_id=state,
            expected_return=measured[state]["expected_return"],
            expected_volatility=measured[state]["expected_volatility"],
            downside_volatility=measured[state]["downside_volatility"],
            occupancy=measured[state]["occupancy"],
            expected_duration=float(durations[state]),
            self_transition_probability=float(self_transitions[state]),
            label=labels[state],
        )
        for state in range(parameters.n_states)
    )


def assign_labels(expected_volatilities: list[float]) -> dict[int, RegimeLabel]:
    """Map state IDs to labels by rank of measured volatility.

    The lowest-volatility state is CALM and the highest is CRISIS, with
    intermediate states spread across the scale. Labels follow the
    statistics, so refitting with a different seed relabels consistently even
    though the raw state IDs change.

    When a model has more states than labels, two states can share one label.
    That is deliberate and harmless: labels are for reading, and behavior is
    driven by the measured statistics instead.
    """
    n_states = len(expected_volatilities)
    if n_states == 0:
        return {}
    if n_states == 1:
        return {0: RegimeLabel.NORMAL}

    ranked = sorted(range(n_states), key=lambda state: expected_volatilities[state])
    span = len(ORDERED_LABELS) - 1
    labels: dict[int, RegimeLabel] = {}
    for rank, state in enumerate(ranked):
        position = int(round(rank * span / (n_states - 1)))
        labels[state] = ORDERED_LABELS[position]
    return labels


def volatility_ordered_states(statistics: tuple[StateStatistics, ...]) -> tuple[int, ...]:
    """State IDs ordered from calmest to most volatile, by measured
    volatility. The ordering a risk policy should use when it needs relative
    rank rather than an absolute number.
    """
    return tuple(
        statistic.state_id
        for statistic in sorted(statistics, key=lambda item: item.expected_volatility)
    )


def _as_date(value: pd.Timestamp | dt.datetime | dt.date | str) -> dt.date:
    return pd.Timestamp(value).date()
