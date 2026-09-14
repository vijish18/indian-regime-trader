"""Market-regime detection: the HMM engine, regime-to-exposure policy, and
model artifact registry. Operates only on market-level features (NIFTY 50,
India VIX, breadth) -- never on individual stock series. See
docs/SPECIFICATION.md section 6 and docs/ARCHITECTURE.md.
"""

from core.regime.gaussian_hmm import (
    FilterResult,
    GaussianHMMParameters,
    HMMNumericalError,
    filter_step,
    forward_filter,
)
from core.regime.hmm_engine import (
    FittedRegimeModel,
    HMMRegimeEngine,
    HMMTrainingResult,
    InsufficientHistoryError,
    ModelSelectionError,
    RegimeLabel,
    RegimeState,
    StateStatistics,
    assign_labels,
    characterize_states,
    volatility_ordered_states,
)
from core.regime.model_registry import (
    ModelArtifact,
    ModelNotFoundError,
    ModelRegistry,
    NoApprovedModelError,
    build_model_id,
)

__all__ = [
    "FilterResult",
    "FittedRegimeModel",
    "GaussianHMMParameters",
    "HMMNumericalError",
    "HMMRegimeEngine",
    "HMMTrainingResult",
    "InsufficientHistoryError",
    "ModelArtifact",
    "ModelNotFoundError",
    "ModelRegistry",
    "ModelSelectionError",
    "NoApprovedModelError",
    "RegimeLabel",
    "RegimeState",
    "StateStatistics",
    "assign_labels",
    "build_model_id",
    "characterize_states",
    "filter_step",
    "forward_filter",
    "volatility_ordered_states",
]
