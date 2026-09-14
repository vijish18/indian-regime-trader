"""Gaussian HMM market-regime classifier.

Implements the model described in docs/SPECIFICATION.md section 6: a state
classifier over market-level volatility/stress features, using forward
filtering only (never full-sequence Viterbi) so that decisions made at time
``t`` cannot be altered by observations after ``t``. This module must not be
given individual stock price series or asked to predict returns -- it answers
"how risky is the market right now", not "what will the market do next".

Not implemented yet (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class RegimeLabel(StrEnum):
    """Human-readable regime label, assigned post-hoc by learned
    characteristics (docs/SPECIFICATION.md section 6.1) -- never by raw,
    arbitrary state ID.
    """

    CALM = "calm"
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"


@dataclass(frozen=True)
class RegimeState:
    """The output of one filtered-inference step."""

    as_of: pd.Timestamp
    state_id: int
    label: RegimeLabel
    probabilities: tuple[float, ...]
    confidence: float


@dataclass(frozen=True)
class HMMTrainingResult:
    """Metadata produced by fitting one candidate model, sufficient to decide
    between candidate state counts via BIC and to persist a reproducible
    artifact (docs/SPECIFICATION.md section 6, "Persisted artifacts").
    """

    n_states: int
    covariance_type: str
    bic: float
    aic: float
    log_likelihood: float
    seed: int
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    feature_columns: tuple[str, ...]


class HMMRegimeEngine:
    """Fits and runs forward-filtered inference for a Gaussian HMM over
    market-level features.

    Training must use only data available as of ``training_end`` (no
    look-ahead); inference must use forward filtering only, so that
    ``filter(...)`` applied through time ``t`` never changes as observations
    after ``t`` are appended, for fixed model parameters.
    """

    def __init__(self, n_states: int, covariance_type: str, random_seed: int) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.random_seed = random_seed

    def fit(self, features: pd.DataFrame) -> HMMTrainingResult:
        """Fit the HMM on a training window of causal, pre-scaled features.

        Args:
            features: rows indexed by session date, columns are the market
                feature set (see core/features/feature_engineering.py). Must
                already be scaled using a scaler fit on this same window.

        Returns:
            Metadata describing the fitted model, for model selection and
            for persistence via ``model_registry.ModelRegistry``.
        """
        raise NotImplementedError("Phase 5: HMM training is not implemented yet.")

    def filter(self, features: pd.DataFrame) -> list[RegimeState]:
        """Run forward-only filtered inference over ``features``.

        Must not use future observations to infer the state at any given
        row. Equivalent to running the standard HMM forward algorithm and
        normalizing alpha at each step (docs/SPECIFICATION.md section 6.2).
        """
        raise NotImplementedError("Phase 5: HMM filtering is not implemented yet.")

    def characterize_states(self, features: pd.DataFrame) -> dict[int, RegimeLabel]:
        """Map arbitrary state IDs to ``RegimeLabel`` using learned
        volatility/downside-volatility/transition characteristics computed
        only within the training window used to fit this model instance.
        """
        raise NotImplementedError("Phase 5: state characterization is not implemented yet.")


def compute_bic(log_likelihood: float, n_params: int, n_observations: int) -> float:
    """Bayesian Information Criterion, used for state-count model selection
    on the training window only (docs/SPECIFICATION.md section 10, "Model
    selection: BIC on training only").
    """
    raise NotImplementedError("Phase 5: BIC computation is not implemented yet.")


def covariance_condition_number(covariance: np.ndarray) -> float:
    """Condition number of a fitted covariance matrix, used to reject
    near-singular fits (docs/SPECIFICATION.md section 6.3).
    """
    raise NotImplementedError("Phase 5: covariance diagnostics are not implemented yet.")
