"""Model persistence and frozen normalization parameters.

Two properties matter most here: a reloaded model must reproduce the exact
same regime calls (otherwise "reproducible from the audit log" is a claim
without backing), and nothing may go live without an explicit approval step.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.features.feature_engineering import (
    DEFAULT_FEATURE_CONFIG,
    build_default_feature_definitions,
    drop_warmup_rows,
    feature_set_version,
)
from core.features.feature_scaler import CausalFeatureScaler, ScalerParams
from core.regime.hmm_engine import HMMRegimeEngine
from core.regime.model_registry import (
    ARTIFACT_SCHEMA_VERSION,
    ModelArtifact,
    ModelNotFoundError,
    ModelRegistry,
    NoApprovedModelError,
    build_model_id,
)
from tests.unit.test_hmm_engine import hmm_config, market


@pytest.fixture(scope="module")
def artifact() -> ModelArtifact:
    features, returns = market(n=260)
    scaled, scaler_params = CausalFeatureScaler().fit_transform(features)
    model = HMMRegimeEngine(hmm_config()).fit(scaled, returns)
    return ModelArtifact(
        model_id=build_model_id(model.training_result.training_end, model.n_states, 17, "abc123"),
        created_at=dt.datetime(2024, 3, 1, 12, 30, tzinfo=dt.UTC),
        model=model,
        scaler=scaler_params,
        feature_version="abc123",
        notes="fixture model",
    )


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_every_required_field_survives_a_round_trip(
    artifact: ModelArtifact, tmp_path: Path
) -> None:
    """docs/SPECIFICATION.md section 6's "persisted artifacts" list, checked
    field by field.
    """
    registry = ModelRegistry(tmp_path)
    registry.save(artifact)
    loaded = registry.load(artifact.model_id)

    assert loaded.model_id == artifact.model_id
    assert loaded.created_at == artifact.created_at  # training timestamp
    assert loaded.feature_version == artifact.feature_version
    assert loaded.notes == artifact.notes

    original, restored = artifact.model.training_result, loaded.model.training_result
    assert restored.training_start == original.training_start  # training period
    assert restored.training_end == original.training_end
    assert restored.seed == original.seed  # random seed
    assert restored.n_states == original.n_states  # selected state count
    assert restored.bic == original.bic
    assert restored.aic == original.aic
    assert restored.log_likelihood == original.log_likelihood
    assert restored.covariance_type == original.covariance_type
    assert restored.feature_columns == original.feature_columns
    assert restored.converged == original.converged
    assert restored.n_observations == original.n_observations

    # Normalization parameters
    assert loaded.scaler == artifact.scaler

    # Model parameters, including the transition matrix
    np.testing.assert_allclose(
        loaded.model.parameters.transition_matrix, artifact.model.parameters.transition_matrix
    )
    np.testing.assert_allclose(loaded.model.parameters.means, artifact.model.parameters.means)
    np.testing.assert_allclose(
        loaded.model.parameters.covariances, artifact.model.parameters.covariances
    )
    np.testing.assert_allclose(
        loaded.model.parameters.start_probabilities,
        artifact.model.parameters.start_probabilities,
    )

    # State statistics
    assert loaded.model.statistics == artifact.model.statistics


def test_a_reloaded_model_reproduces_identical_regime_calls(
    artifact: ModelArtifact, tmp_path: Path
) -> None:
    """The point of persistence: a decision made months ago can be replayed
    exactly from the stored artifact.
    """
    registry = ModelRegistry(tmp_path)
    registry.save(artifact)
    loaded = registry.load(artifact.model_id)

    features, _ = market(n=260)
    scaled = CausalFeatureScaler().transform(features, loaded.scaler)
    engine = HMMRegimeEngine(hmm_config())

    before = engine.filter(artifact.model, scaled)
    after = engine.filter(loaded.model, scaled)
    assert after == before


def test_artifacts_are_plain_json_not_pickle(artifact: ModelArtifact, tmp_path: Path) -> None:
    """A file loaded by the process that places orders must not be able to
    execute code, and must be readable by a human reviewing an incident.
    """
    registry = ModelRegistry(tmp_path)
    path = registry.save(artifact)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert set(payload) >= {
        "model_id",
        "created_at",
        "feature",
        "training",
        "scaler",
        "parameters",
        "state_statistics",
    }
    assert payload["training"]["n_states"] == artifact.model.n_states
    assert len(payload["parameters"]["transition_matrix"]) == artifact.model.n_states


def test_unsupported_schema_version_is_rejected(artifact: ModelArtifact) -> None:
    payload = artifact.to_dict()
    payload["schema_version"] = 99

    with pytest.raises(ValueError, match="schema version"):
        ModelArtifact.from_dict(payload)


# --------------------------------------------------------------------------
# Immutability and lookup
# --------------------------------------------------------------------------


def test_saving_over_an_existing_model_id_is_refused(
    artifact: ModelArtifact, tmp_path: Path
) -> None:
    """A past decision referring to a model id must always resolve to the same
    model.
    """
    registry = ModelRegistry(tmp_path)
    registry.save(artifact)

    with pytest.raises(FileExistsError, match="immutable"):
        registry.save(artifact)


def test_loading_an_unknown_model_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="no model artifact"):
        ModelRegistry(tmp_path).load("does-not-exist")


def test_list_models_reports_saved_artifacts(artifact: ModelArtifact, tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    assert registry.list_models() == []

    registry.save(artifact)
    registry.approve(artifact.model_id)
    assert registry.list_models() == [artifact.model_id]


def test_build_model_id_distinguishes_differing_fits() -> None:
    end = dt.date(2024, 6, 28)
    assert build_model_id(end, 3, 17, "abc") != build_model_id(end, 4, 17, "abc")
    assert build_model_id(end, 3, 17, "abc") != build_model_id(end, 3, 29, "abc")
    assert build_model_id(end, 3, 17, "abc") != build_model_id(end, 3, 17, "def")


# --------------------------------------------------------------------------
# Approval fails closed
# --------------------------------------------------------------------------


def test_saving_a_model_does_not_make_it_live(artifact: ModelArtifact, tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    registry.save(artifact)

    with pytest.raises(NoApprovedModelError, match="no approved model"):
        registry.load_current_approved()


def test_approved_model_loads(artifact: ModelArtifact, tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path)
    registry.save(artifact)
    registry.approve(artifact.model_id)

    assert registry.load_current_approved().model_id == artifact.model_id


def test_approving_an_unknown_model_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="cannot approve"):
        ModelRegistry(tmp_path).approve("never-saved")


def test_an_approval_pointing_at_a_missing_artifact_fails_closed(
    artifact: ModelArtifact, tmp_path: Path
) -> None:
    """Never silently fall back to another model."""
    registry = ModelRegistry(tmp_path)
    registry.save(artifact)
    registry.approve(artifact.model_id)
    registry.artifact_path(artifact.model_id).unlink()

    with pytest.raises(NoApprovedModelError, match="is missing"):
        registry.load_current_approved()


def test_artifact_requires_a_timezone_aware_timestamp(artifact: ModelArtifact) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        ModelArtifact(
            model_id="x",
            created_at=dt.datetime(2024, 1, 1, 12, 0),  # naive
            model=artifact.model,
            scaler=artifact.scaler,
            feature_version="abc",
        )


# --------------------------------------------------------------------------
# Frozen normalization parameters
# --------------------------------------------------------------------------


def test_out_of_sample_data_is_scaled_with_training_statistics() -> None:
    """The anti-leak property: an OOS window must be normalized by the
    training window's mean and standard deviation, not its own. If it used
    its own, a crisis would inflate its own denominator and be rescaled to
    look ordinary.
    """
    index = pd.bdate_range("2022-01-03", periods=200)
    training = pd.DataFrame({"feature": np.linspace(0.0, 1.0, 100)}, index=index[:100])
    out_of_sample = pd.DataFrame({"feature": np.linspace(5.0, 6.0, 100)}, index=index[100:])

    scaler = CausalFeatureScaler()
    params = scaler.fit(training)
    transformed = scaler.transform(out_of_sample, params)

    # Scaled with the training mean (~0.5), the OOS window sits far above zero.
    assert transformed["feature"].mean() > 10
    # Had it been refit on itself, it would have been centred on zero.
    assert abs(transformed["feature"].mean()) > 1.0


def test_fit_transform_standardizes_the_training_window() -> None:
    index = pd.bdate_range("2022-01-03", periods=120)
    frame = pd.DataFrame(
        {"a": np.random.default_rng(1).normal(5.0, 2.0, 120), "b": np.arange(120.0)},
        index=index,
    )
    scaled, params = CausalFeatureScaler().fit_transform(frame)

    np.testing.assert_allclose(scaled.mean().to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(ddof=0).to_numpy(), 1.0, atol=1e-12)
    assert params.fit_start == index[0].date()
    assert params.fit_end == index[-1].date()


def test_scaler_rejects_a_zero_variance_feature() -> None:
    frame = pd.DataFrame(
        {"constant": np.full(50, 3.0)}, index=pd.bdate_range("2022-01-03", periods=50)
    )
    with pytest.raises(ValueError, match="zero-variance"):
        CausalFeatureScaler().fit(frame)


def test_scaler_rejects_nan_in_the_training_window() -> None:
    values = np.arange(50.0)
    values[7] = np.nan
    frame = pd.DataFrame({"feature": values}, index=pd.bdate_range("2022-01-03", periods=50))

    with pytest.raises(ValueError, match="drop feature warm-up rows"):
        CausalFeatureScaler().fit(frame)


def test_scaler_rejects_an_empty_window() -> None:
    with pytest.raises(ValueError, match="empty training window"):
        CausalFeatureScaler().fit(pd.DataFrame())


def test_transform_rejects_a_changed_feature_set() -> None:
    """A model may only be applied to the feature set it was trained on."""
    index = pd.bdate_range("2022-01-03", periods=60)
    frame = pd.DataFrame({"a": np.arange(60.0), "b": np.arange(60.0) * 2}, index=index)
    scaler = CausalFeatureScaler()
    params = scaler.fit(frame)

    with pytest.raises(ValueError, match="do not match the fitted columns"):
        scaler.transform(frame[["b", "a"]], params)
    with pytest.raises(ValueError, match="do not match the fitted columns"):
        scaler.transform(frame[["a"]], params)


def test_scaler_params_validate_their_own_shape() -> None:
    with pytest.raises(ValueError, match="equal length"):
        ScalerParams(
            feature_columns=("a", "b"),
            means=(0.0,),
            stds=(1.0, 1.0),
            fit_start=dt.date(2022, 1, 3),
            fit_end=dt.date(2022, 6, 3),
        )
    with pytest.raises(ValueError, match="std must be positive"):
        ScalerParams(
            feature_columns=("a",),
            means=(0.0,),
            stds=(0.0,),
            fit_start=dt.date(2022, 1, 3),
            fit_end=dt.date(2022, 6, 3),
        )


# --------------------------------------------------------------------------
# Feature-set versioning
# --------------------------------------------------------------------------


def test_feature_version_is_stable_for_an_unchanged_feature_set() -> None:
    definitions = build_default_feature_definitions(DEFAULT_FEATURE_CONFIG)
    assert feature_set_version(definitions) == feature_set_version(definitions)


def test_feature_version_changes_when_a_window_changes() -> None:
    """A model records the feature version it was trained on, so silently
    changing a lookback and reusing the old model becomes detectable.
    """
    baseline = build_default_feature_definitions(DEFAULT_FEATURE_CONFIG)
    altered_config = DEFAULT_FEATURE_CONFIG.model_copy(update={"trend_window": 150})
    altered = build_default_feature_definitions(altered_config)

    assert feature_set_version(baseline) != feature_set_version(altered)


def test_feature_version_is_order_independent() -> None:
    definitions = list(build_default_feature_definitions(DEFAULT_FEATURE_CONFIG))
    assert feature_set_version(definitions) == feature_set_version(list(reversed(definitions)))


def test_drop_warmup_rows_removes_only_incomplete_rows() -> None:
    index = pd.bdate_range("2022-01-03", periods=6)
    frame = pd.DataFrame(
        {"a": [np.nan, np.nan, 1.0, 2.0, 3.0, 4.0], "b": [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0]},
        index=index,
    )
    cleaned = drop_warmup_rows(frame)

    assert len(cleaned) == 4
    assert not cleaned.isna().to_numpy().any()
    assert cleaned.index[0] == index[2]
