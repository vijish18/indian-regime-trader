"""Point-in-time tradable universe construction and stock selection.

Kept independent of ``core.regime``: stock selection must not receive the
regime state as an input to its ranking, only as a later constraint on gross
exposure applied by portfolio construction -- this is what lets the system
measure the regime layer's incremental value in isolation
(docs/SPECIFICATION.md section 1.2, "IMPORTANT").
"""

from universe.factor_calculator import (
    FactorCalculator,
    FactorSet,
    InsufficientFactorHistoryError,
)
from universe.stock_selector import (
    CandidateUniverse,
    SelectionExclusion,
    SelectionExclusionReason,
    StockScore,
    StockSelector,
)
from universe.universe import (
    ConstituentSnapshot,
    ExclusionReason,
    UniverseExclusion,
    UniverseProvider,
    UniverseSnapshot,
)

__all__ = [
    "CandidateUniverse",
    "ConstituentSnapshot",
    "ExclusionReason",
    "FactorCalculator",
    "FactorSet",
    "InsufficientFactorHistoryError",
    "SelectionExclusion",
    "SelectionExclusionReason",
    "StockScore",
    "StockSelector",
    "UniverseExclusion",
    "UniverseProvider",
    "UniverseSnapshot",
]
