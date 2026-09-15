"""Unit tests for ``orchestration/model_validation.py`` (Phase 19): a
loaded model's metadata is validated before it is trusted for a live
decision -- every problem is independently detectable, and all are
reported together rather than stopping at the first one.

Uses a genuinely fitted model (via ``tests.unit.test_hmm_engine``'s own
``market``/``hmm_config`` fixtures, the same recipe
``tests/unit/test_model_registry.py`` already uses) rather than
hand-constructing ``GaussianHMMParameters`` -- ``model_validation`` only
ever reads ``artifact.model.training_result``/``artifact.feature_version``,
never the fitted parameters themselves, so a real fit is simpler and less
error-prone than faking the internals it doesn't touch.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

from core.features.feature_scaler import CausalFeatureScaler
from core.regime.hmm_engine import HMMRegimeEngine
from core.regime.model_registry import ModelArtifact, build_model_id
from data.calendar import NSETradingCalendar
from orchestration.model_validation import validate_model_metadata
from tests.unit.test_hmm_engine import hmm_config, market

_AS_OF = dt.date(2024, 6, 3)


def _calendar() -> NSETradingCalendar:
    return NSETradingCalendar(holidays={}, covered_years=frozenset(range(2019, 2026)))


def _artifact(feature_version: str = "fv1") -> ModelArtifact:
    features, returns = market(n=260)
    scaled, scaler_params = CausalFeatureScaler().fit_transform(features)
    model = HMMRegimeEngine(hmm_config()).fit(scaled, returns)
    return ModelArtifact(
        model_id=build_model_id(model.training_result.training_end, model.n_states, 17, "fv1"),
        created_at=dt.datetime(2024, 5, 31, 12, 0, tzinfo=dt.UTC),
        model=model,
        scaler=scaler_params,
        feature_version=feature_version,
    )


def test_valid_recent_model_has_no_problems() -> None:
    artifact = _artifact()
    columns = list(artifact.model.training_result.feature_columns)
    recent_as_of = _calendar().sessions_offset(
        artifact.model.training_result.training_end, 5
    )
    problems = validate_model_metadata(
        artifact, columns, "fv1", as_of=recent_as_of, calendar=_calendar(), max_age_sessions=90
    )
    assert problems == []


def test_rejected_training_run_is_a_problem() -> None:
    artifact = _artifact()
    broken_result = replace(artifact.model.training_result, rejection_reason="did not converge")
    broken_model = replace(artifact.model, training_result=broken_result)
    artifact = replace(artifact, model=broken_model)

    problems = validate_model_metadata(
        artifact,
        list(broken_result.feature_columns),
        "fv1",
        as_of=_AS_OF,
        calendar=_calendar(),
        max_age_sessions=90,
    )
    assert any("rejected" in p for p in problems)


def test_feature_column_mismatch_is_a_problem() -> None:
    artifact = _artifact()
    problems = validate_model_metadata(
        artifact, ["not", "the", "real", "columns"], "fv1",
        as_of=_AS_OF, calendar=_calendar(), max_age_sessions=90,
    )
    assert any("feature columns" in p for p in problems)


def test_feature_version_mismatch_is_a_problem() -> None:
    artifact = _artifact(feature_version="fv1")
    columns = list(artifact.model.training_result.feature_columns)
    problems = validate_model_metadata(
        artifact, columns, "fv2", as_of=_AS_OF, calendar=_calendar(), max_age_sessions=90
    )
    assert any("feature_version" in p for p in problems)


def test_stale_model_is_a_problem() -> None:
    artifact = _artifact()
    columns = list(artifact.model.training_result.feature_columns)
    training_end = artifact.model.training_result.training_end
    far_future = training_end + dt.timedelta(days=365)
    problems = validate_model_metadata(
        artifact, columns, "fv1", as_of=far_future, calendar=_calendar(), max_age_sessions=5
    )
    assert any("more than 5 trading sessions" in p for p in problems)


def test_multiple_problems_are_all_reported_together() -> None:
    artifact = _artifact(feature_version="wrong")
    broken_result = replace(artifact.model.training_result, rejection_reason="bad fit")
    broken_model = replace(artifact.model, training_result=broken_result)
    artifact = replace(artifact, model=broken_model)
    training_end = broken_result.training_end
    far_future = training_end + dt.timedelta(days=365)

    problems = validate_model_metadata(
        artifact, ["wrong", "columns"], "fv1",
        as_of=far_future, calendar=_calendar(), max_age_sessions=5,
    )
    assert len(problems) == 4


def test_calendar_coverage_failure_is_reported_not_raised() -> None:
    artifact = _artifact()
    columns = list(artifact.model.training_result.feature_columns)
    narrow_calendar = NSETradingCalendar(holidays={}, covered_years=frozenset({2024}))
    problems = validate_model_metadata(
        artifact, columns, "fv1",
        as_of=dt.date(2019, 1, 1), calendar=narrow_calendar, max_age_sessions=21,
    )
    assert any("could not verify model age" in p for p in problems)
