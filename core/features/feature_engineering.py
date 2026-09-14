"""Computes market-level features for the HMM regime engine from NIFTY 50,
India VIX, and breadth data (docs/SPECIFICATION.md section 5).

Feature choice is deliberately volatility/stress-focused rather than
directional: this is what keeps the HMM a risk-state classifier instead of a
market-direction predictor (docs/SPECIFICATION.md section 1.2, "Too many
directional features"). Any signed-return feature added here needs a
documented reason -- see docs/ARCHITECTURE.md, "Resolved specification
ambiguities" (B5).

Not implemented yet (Phase 4).
"""

from __future__ import annotations

import pandas as pd


class FeatureEngineer:
    """Builds the market-feature matrix used to train and run the HMM.

    All methods must be strictly causal: the feature value at row ``t`` may
    only depend on data with timestamp <= ``t``. This property is enforced by
    the no-look-ahead test suite (tests/unit/test_features_no_lookahead.py,
    added in Phase 4).
    """

    def realized_volatility(self, close: pd.Series, window: int) -> pd.Series:
        """Annualized realized volatility of daily log returns over a
        trailing ``window``-day rolling window.
        """
        raise NotImplementedError("Phase 4: feature engineering is not implemented yet.")

    def downside_volatility(self, close: pd.Series, window: int) -> pd.Series:
        """Std. dev. of negative daily log returns over a trailing window."""
        raise NotImplementedError("Phase 4: feature engineering is not implemented yet.")

    def atr_normalized(
        self, high: pd.Series, low: pd.Series, close: pd.Series, window: int
    ) -> pd.Series:
        """ATR(window) / close."""
        raise NotImplementedError("Phase 4: feature engineering is not implemented yet.")

    def overnight_gap(self, open_: pd.Series, prior_close: pd.Series) -> pd.Series:
        """(open_t / close_{t-1}) - 1."""
        raise NotImplementedError("Phase 4: feature engineering is not implemented yet.")

    def range_expansion(
        self, high: pd.Series, low: pd.Series, close: pd.Series, median_window: int
    ) -> pd.Series:
        """True range relative to its trailing rolling median."""
        raise NotImplementedError("Phase 4: feature engineering is not implemented yet.")

    def vix_percentile(self, vix_close: pd.Series, window: int) -> pd.Series:
        """Trailing-window percentile rank of the VIX level.

        Must use only a trailing/expanding window, never a full-history
        percentile -- a full-history percentile leaks knowledge of future
        extreme VIX prints into earlier observations.
        """
        raise NotImplementedError("Phase 4: feature engineering is not implemented yet.")

    def volume_stress(self, volume: pd.Series, window: int) -> pd.Series:
        """Rolling z-score of log volume (or turnover)."""
        raise NotImplementedError("Phase 4: feature engineering is not implemented yet.")

    def breadth_stress(
        self,
        constituent_closes: pd.DataFrame,
        point_in_time_membership: pd.DataFrame,
        ma_window: int,
    ) -> pd.Series:
        """% of index constituents trading above their ``ma_window``-day
        moving average, computed against point-in-time index membership --
        not today's constituent list -- to avoid leaking future index
        composition into historical breadth values.
        """
        raise NotImplementedError("Phase 4: feature engineering is not implemented yet.")

    def build_feature_matrix(
        self,
        nifty_ohlcv: pd.DataFrame,
        india_vix_ohlcv: pd.DataFrame,
        breadth_inputs: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Assemble the full causal feature matrix used by the HMM engine."""
        raise NotImplementedError("Phase 4: feature engineering is not implemented yet.")
