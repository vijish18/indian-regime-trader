"""Causal feature scaling for walk-forward model fitting.

Scaler parameters must be fit on a training window only and frozen (never
refit) when applied to the out-of-sample window that follows it
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

Not implemented yet (Phase 5, alongside the HMM engine that consumes it).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ScalerParams:
    """Frozen per-feature scaling parameters fit on one training window."""

    feature_columns: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp


class CausalFeatureScaler:
    """Fits on a training window and applies a frozen transform to any data,
    in or out of sample.
    """

    def fit(self, features: pd.DataFrame) -> ScalerParams:
        """Compute and freeze scaling parameters from a training window."""
        raise NotImplementedError("Phase 5: feature scaling is not implemented yet.")

    def transform(self, features: pd.DataFrame, params: ScalerParams) -> pd.DataFrame:
        """Apply previously-frozen scaling parameters to ``features``.

        Must not recompute statistics from ``features`` itself -- doing so
        would reintroduce look-ahead bias into out-of-sample evaluation.
        """
        raise NotImplementedError("Phase 5: feature scaling is not implemented yet.")
