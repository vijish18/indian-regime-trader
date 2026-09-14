"""Versioned persistence for fitted regime models.

Every field docs/SPECIFICATION.md section 6 lists under "Persisted artifacts"
is stored, so any historical decision can be traced back to the exact model
that produced it and replayed: the model parameters, the training period, the
feature-set version and column list, the frozen normalization parameters, the
random seed, the selected state count, BIC/AIC, the transition matrix, the
per-state statistics, and the training timestamp.

Artifacts are JSON, not pickle. Three reasons, in order of importance:

1. **Unpickling is code execution.** A model artifact is a file that gets
   loaded by the process that places orders; it should never be able to run
   anything.
2. **Auditable.** A regulator, a reviewer, or a confused engineer at 2am can
   read a JSON artifact and diff two of them. A pickle is opaque.
3. **Stable across library versions.** A pickled numpy array can fail to load
   after an upgrade; a list of floats cannot.

Models are tiny (a handful of states over a handful of features), so there is
no size argument against it.

Approval is a separate, explicit step: saving a model does not make it live.
``load_current_approved`` raises when nothing is approved rather than falling
back to the newest fit, because silently promoting an unreviewed model is
exactly the failure the go-live gate exists to prevent
(docs/SPECIFICATION.md section 15.1 step 8, section 20).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.features.feature_scaler import ScalerParams
from core.regime.gaussian_hmm import GaussianHMMParameters
from core.regime.hmm_engine import (
    FittedRegimeModel,
    HMMTrainingResult,
    RegimeLabel,
    StateStatistics,
)

ARTIFACT_SCHEMA_VERSION = 1
APPROVAL_FILENAME = "approved.json"


class ModelNotFoundError(FileNotFoundError):
    """No artifact exists for the requested model id."""


class NoApprovedModelError(RuntimeError):
    """No model has been approved for live/paper use.

    Fatal by design: the startup sequence must refuse to trade rather than
    guess which model was intended.
    """


@dataclass(frozen=True)
class ModelArtifact:
    """A complete, self-contained record of one trained model."""

    model_id: str
    created_at: dt.datetime
    """Training timestamp (timezone-aware, UTC)."""

    model: FittedRegimeModel
    scaler: ScalerParams
    """Frozen normalization parameters -- without these the persisted model
    cannot be applied to raw features reproducibly."""

    feature_version: str
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        parameters = self.model.parameters
        training = self.model.training_result
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_id": self.model_id,
            "created_at": self.created_at.isoformat(),
            "notes": self.notes,
            "feature": {
                "version": self.feature_version,
                "columns": list(training.feature_columns),
            },
            "training": {
                "start": training.training_start.isoformat(),
                "end": training.training_end.isoformat(),
                "n_observations": training.n_observations,
                "seed": training.seed,
                "n_states": training.n_states,
                "covariance_type": training.covariance_type,
                "bic": training.bic,
                "aic": training.aic,
                "log_likelihood": training.log_likelihood,
                "converged": training.converged,
                "iterations": training.iterations,
                "max_condition_number": training.max_condition_number,
                "min_occupancy": training.min_occupancy,
            },
            "scaler": {
                "feature_columns": list(self.scaler.feature_columns),
                "means": list(self.scaler.means),
                "stds": list(self.scaler.stds),
                "fit_start": self.scaler.fit_start.isoformat(),
                "fit_end": self.scaler.fit_end.isoformat(),
            },
            "parameters": {
                "start_probabilities": parameters.start_probabilities.tolist(),
                "transition_matrix": parameters.transition_matrix.tolist(),
                "means": parameters.means.tolist(),
                "covariances": parameters.covariances.tolist(),
                "covariance_type": parameters.covariance_type,
            },
            "state_statistics": [
                {
                    "state_id": statistic.state_id,
                    "label": statistic.label.value,
                    "expected_return": statistic.expected_return,
                    "expected_volatility": statistic.expected_volatility,
                    "downside_volatility": statistic.downside_volatility,
                    "occupancy": statistic.occupancy,
                    "expected_duration": statistic.expected_duration,
                    "self_transition_probability": statistic.self_transition_probability,
                }
                for statistic in self.model.statistics
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelArtifact:
        version = payload.get("schema_version")
        if version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"artifact schema version {version!r} is not supported by this build "
                f"(expected {ARTIFACT_SCHEMA_VERSION}); refusing to guess its layout"
            )

        feature = payload["feature"]
        training = payload["training"]
        raw_parameters = payload["parameters"]
        scaler = payload["scaler"]

        parameters = GaussianHMMParameters(
            start_probabilities=np.asarray(raw_parameters["start_probabilities"], dtype=np.float64),
            transition_matrix=np.asarray(raw_parameters["transition_matrix"], dtype=np.float64),
            means=np.asarray(raw_parameters["means"], dtype=np.float64),
            covariances=np.asarray(raw_parameters["covariances"], dtype=np.float64),
            covariance_type=raw_parameters["covariance_type"],
        )

        training_result = HMMTrainingResult(
            n_states=training["n_states"],
            covariance_type=training["covariance_type"],
            seed=training["seed"],
            bic=training["bic"],
            aic=training["aic"],
            log_likelihood=training["log_likelihood"],
            converged=training["converged"],
            iterations=training["iterations"],
            training_start=dt.date.fromisoformat(training["start"]),
            training_end=dt.date.fromisoformat(training["end"]),
            n_observations=training["n_observations"],
            feature_columns=tuple(feature["columns"]),
            max_condition_number=training["max_condition_number"],
            min_occupancy=training["min_occupancy"],
        )

        statistics = tuple(
            StateStatistics(
                state_id=entry["state_id"],
                expected_return=entry["expected_return"],
                expected_volatility=entry["expected_volatility"],
                downside_volatility=entry["downside_volatility"],
                occupancy=entry["occupancy"],
                expected_duration=entry["expected_duration"],
                self_transition_probability=entry["self_transition_probability"],
                label=RegimeLabel(entry["label"]),
            )
            for entry in payload["state_statistics"]
        )

        return cls(
            model_id=payload["model_id"],
            created_at=dt.datetime.fromisoformat(payload["created_at"]),
            model=FittedRegimeModel(
                parameters=parameters,
                statistics=statistics,
                training_result=training_result,
            ),
            scaler=ScalerParams(
                feature_columns=tuple(scaler["feature_columns"]),
                means=tuple(scaler["means"]),
                stds=tuple(scaler["stds"]),
                fit_start=dt.date.fromisoformat(scaler["fit_start"]),
                fit_end=dt.date.fromisoformat(scaler["fit_end"]),
            ),
            feature_version=feature["version"],
            notes=payload.get("notes"),
        )


class ModelRegistry:
    """Read/write access to persisted model artifacts on the filesystem."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root

    def artifact_path(self, model_id: str) -> Path:
        return self.artifact_root / f"{_safe_model_id(model_id)}.json"

    def save(self, artifact: ModelArtifact) -> Path:
        """Persist an artifact as a new, immutable version.

        Refuses to overwrite an existing id: a model that a historical
        decision refers to must never change underneath that reference.
        """
        path = self.artifact_path(artifact.model_id)
        if path.exists():
            raise FileExistsError(
                f"model {artifact.model_id!r} already exists at {path}; artifacts are "
                "immutable so that a past decision's model can always be reproduced"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load(self, model_id: str) -> ModelArtifact:
        path = self.artifact_path(model_id)
        if not path.is_file():
            raise ModelNotFoundError(f"no model artifact at {path}")
        return ModelArtifact.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_models(self) -> list[str]:
        """Every stored model id, ascending."""
        if not self.artifact_root.is_dir():
            return []
        return sorted(
            path.stem
            for path in self.artifact_root.glob("*.json")
            if path.name != APPROVAL_FILENAME
        )

    def approve(self, model_id: str) -> None:
        """Mark one stored model as approved for live/paper use.

        A separate, deliberate action: fitting a model and trusting it with
        capital are different decisions, and the go-live gate
        (docs/SPECIFICATION.md section 20) sits between them.
        """
        if not self.artifact_path(model_id).is_file():
            raise ModelNotFoundError(
                f"cannot approve {model_id!r}: no such artifact in {self.artifact_root}"
            )
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        (self.artifact_root / APPROVAL_FILENAME).write_text(
            json.dumps({"model_id": model_id}, indent=2), encoding="utf-8"
        )

    def approved_model_id(self) -> str | None:
        path = self.artifact_root / APPROVAL_FILENAME
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_id = payload.get("model_id")
        return str(model_id) if model_id else None

    def load_current_approved(self) -> ModelArtifact:
        """Load the model approved for live/paper use.

        Raises:
            NoApprovedModelError: if nothing is approved, or the approved id
                no longer resolves to an artifact. Fails closed instead of
                falling back to the newest or "best" model, either of which
                would silently put an unreviewed model into production.
        """
        model_id = self.approved_model_id()
        if model_id is None:
            raise NoApprovedModelError(
                f"no approved model in {self.artifact_root}; refusing to fall back to "
                "an unapproved fit"
            )
        try:
            return self.load(model_id)
        except ModelNotFoundError as exc:
            raise NoApprovedModelError(
                f"approved model {model_id!r} is missing from {self.artifact_root}"
            ) from exc


def _safe_model_id(model_id: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in model_id
    )
    if not cleaned.strip("_") or cleaned in (".", ".."):
        raise ValueError(f"model_id {model_id!r} has no usable filename characters")
    return cleaned


def build_model_id(training_end: dt.date, n_states: int, seed: int, feature_version: str) -> str:
    """A descriptive, sortable, collision-resistant id.

    Encodes what actually distinguishes one fit from another, so two models
    trained on the same window with different state counts or feature sets
    cannot collide.
    """
    return f"hmm_{training_end.isoformat()}_{n_states}s_seed{seed}_{feature_version}"
