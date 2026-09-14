"""Persists and loads versioned HMM model artifacts (model, scaler, feature
list, training range, BIC/AIC, seed, metadata) so that any historical
decision can be traced back to the exact model that produced it.
See docs/SPECIFICATION.md section 6, "Persisted artifacts".

Not implemented yet (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.regime.hmm_engine import HMMRegimeEngine, HMMTrainingResult


@dataclass(frozen=True)
class ModelArtifact:
    """A single persisted, versioned model bundle."""

    model_id: str
    engine: HMMRegimeEngine
    training_result: HMMTrainingResult
    scaler_path: Path
    created_at: pd.Timestamp


class ModelRegistry:
    """Read/write access to persisted HMM model artifacts.

    Backed by ``storage`` (table ``hmm_models``, see storage/models.py) and a
    filesystem/object-store path for the serialized model + scaler.
    """

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root

    def save(self, artifact: ModelArtifact) -> None:
        """Persist a fitted model bundle as a new, immutable version."""
        raise NotImplementedError("Phase 5: model persistence is not implemented yet.")

    def load(self, model_id: str) -> ModelArtifact:
        """Load a previously persisted model bundle by ID."""
        raise NotImplementedError("Phase 5: model loading is not implemented yet.")

    def load_current_approved(self) -> ModelArtifact:
        """Load the model bundle currently approved for live/paper use.

        Used at startup (docs/SPECIFICATION.md section 15.1, step 8); must
        fail closed (raise) rather than silently falling back to an
        unapproved model if none is marked approved.
        """
        raise NotImplementedError("Phase 5: approved-model lookup is not implemented yet.")
