"""Feature engineering for the regime model: market-level volatility/stress
features and strictly causal scaling. See docs/SPECIFICATION.md section 5.
"""

from core.features.feature_engineering import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_FEATURES,
    FeatureDefinition,
    FeaturePipeline,
    FeatureSnapshot,
    MarketFeatureInputs,
    build_default_feature_definitions,
    rolling_standardize,
    snapshots_to_frame,
)

__all__ = [
    "DEFAULT_FEATURES",
    "DEFAULT_FEATURE_CONFIG",
    "FeatureDefinition",
    "FeaturePipeline",
    "FeatureSnapshot",
    "MarketFeatureInputs",
    "build_default_feature_definitions",
    "rolling_standardize",
    "snapshots_to_frame",
]
