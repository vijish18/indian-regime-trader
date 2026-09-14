"""Causal feature scaling for walk-forward model fitting.

Scaler parameters are fit on a training window only and frozen (never refit)
when applied to the out-of-sample window that follows it
(docs/SPECIFICATION.md section 5 and section 10). The fitted scaler is
persisted alongside its model in core/regime/model_registry.py so that a
historical decision can always be replayed exactly.

This solves a different problem from
``core.features.feature_engineering.rolling_standardize``: that function
z-scores individual features (e.g. India VIX level) using a continuously
rolling trailing window as part of computing the feature itself, with no
notion of a train/OOS split. This module fits one fixed set of parameters
once, on a designated training window, and reuses those exact frozen
parameters for every subsequent inference call -- the transform a walk-forward
fold applies to its whole feature matrix before handing it to the HMM.

The distinction matters for a specific failure: if the OOS window were
rescaled using its own mean and standard deviation, the model would see
features normalized by statistics that did not exist at decision time, and a
crisis would be silently rescaled to look ordinary because the crisis itself
inflated the denominator.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

_MIN_STD = 1e-12


@dataclass(frozen=True)
class ScalerParams:
    """Frozen per-feature scaling parameters fit on one training window."""

    feature_columns: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    fit_start: dt.date
    fit_end: dt.date

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("ScalerParams needs at least one feature column")
        if not (len(self.feature_columns) == len(self.means) == len(self.stds)):
            raise ValueError(
                "feature_columns, means and stds must have equal length: "
                f"{len(self.feature_columns)}, {len(self.means)}, {len(self.stds)}"
            )
        if any(std <= 0 for std in self.stds):
            raise ValueError("every std must be positive; a zero-variance feature is unusable")
        if self.fit_end < self.fit_start:
            raise ValueError(f"fit_end {self.fit_end} precedes fit_start {self.fit_start}")


class CausalFeatureScaler:
    """Fits on a training window and applies a frozen transform to any data,
    in or out of sample.
    """

    def fit(self, features: pd.DataFrame) -> ScalerParams:
        """Compute and freeze scaling parameters from a training window.

        Raises:
            ValueError: if the window is empty, contains NaN (warm-up rows
                must be dropped first), or has a zero-variance column -- a
                constant feature carries no information and would divide by
                zero.
        """
        if features.empty:
            raise ValueError("cannot fit a scaler on an empty training window")
        if features.isna().to_numpy().any():
            raise ValueError(
                "training window contains NaN; drop feature warm-up rows before "
                "fitting rather than letting them through"
            )

        values = features.to_numpy(dtype=np.float64)
        means = values.mean(axis=0)
        stds = values.std(axis=0, ddof=0)

        degenerate = [
            column for column, std in zip(features.columns, stds, strict=True) if std <= _MIN_STD
        ]
        if degenerate:
            raise ValueError(
                f"zero-variance feature(s) in the training window: {degenerate}; "
                "a constant feature cannot be standardized and tells the model nothing"
            )

        index = features.index
        return ScalerParams(
            feature_columns=tuple(str(column) for column in features.columns),
            means=tuple(float(value) for value in means),
            stds=tuple(float(value) for value in stds),
            fit_start=_as_date(index[0]),
            fit_end=_as_date(index[-1]),
        )

    def transform(self, features: pd.DataFrame, params: ScalerParams) -> pd.DataFrame:
        """Apply previously-frozen scaling parameters to ``features``.

        Never recomputes statistics from ``features`` itself -- doing so would
        reintroduce look-ahead bias into out-of-sample evaluation. The column
        set must match the fitted one exactly, so a feature added or reordered
        after training fails loudly instead of being silently scaled by
        another feature's parameters.
        """
        expected = list(params.feature_columns)
        actual = [str(column) for column in features.columns]
        if actual != expected:
            raise ValueError(
                f"feature columns {actual} do not match the fitted columns {expected}; "
                "a model may only be applied to the feature set it was trained on"
            )

        means = np.asarray(params.means, dtype=np.float64)
        stds = np.asarray(params.stds, dtype=np.float64)
        scaled = (features.to_numpy(dtype=np.float64) - means) / stds
        return pd.DataFrame(scaled, index=features.index, columns=features.columns)

    def fit_transform(self, features: pd.DataFrame) -> tuple[pd.DataFrame, ScalerParams]:
        """Convenience for the training window itself: fit, then apply.

        Deliberately returns the params too, so a caller cannot transform a
        training window without also holding the parameters it must freeze and
        reuse out of sample.
        """
        params = self.fit(features)
        return self.transform(features, params), params


def _as_date(value: pd.Timestamp | dt.datetime | dt.date | str) -> dt.date:
    return pd.Timestamp(value).date()
