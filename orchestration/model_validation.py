"""Validates a loaded ``core.regime.model_registry.ModelArtifact``'s
metadata before it is used for a live decision (Phase 19, step 8:
"validate model metadata"). Fail-closed: any problem returned here must
block strategy execution for the day, not just be logged.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from core.regime.model_registry import ModelArtifact
from data.errors import CalendarCoverageError
from data.interfaces import TradingCalendar


def validate_model_metadata(
    artifact: ModelArtifact,
    expected_feature_columns: Sequence[str],
    expected_feature_version: str,
    *,
    as_of: dt.date,
    calendar: TradingCalendar,
    max_age_sessions: int,
) -> list[str]:
    """Returns a list of problems; empty means the model is safe to use
    today. Checks, independently (all are recorded, not just the first):

    - the model's own training run was accepted (never rejected/unusable).
    - its trained feature columns match this deployment's currently
      configured feature set, in order -- a model trained against a
      different feature definition would silently misinterpret today's
      feature vector otherwise.
    - its ``feature_version`` tag matches, as a second, independent check
      of the same concern (``core.features.feature_engineering.feature_set_version``).
    - it was trained recently enough: not more than ``max_age_sessions``
      trading sessions before ``as_of`` (reuses
      ``config.models.HMMConfig.retrain_interval_sessions`` as the
      caller-supplied threshold, rather than inventing a new config knob).
    """
    problems: list[str] = []
    training_result = artifact.model.training_result

    if not training_result.accepted:
        problems.append(
            f"model {artifact.model_id!r} training run was rejected: "
            f"{training_result.rejection_reason}"
        )

    if tuple(training_result.feature_columns) != tuple(expected_feature_columns):
        problems.append(
            f"model {artifact.model_id!r} was trained on feature columns "
            f"{list(training_result.feature_columns)}, this deployment is configured for "
            f"{list(expected_feature_columns)}"
        )

    if artifact.feature_version != expected_feature_version:
        problems.append(
            f"model {artifact.model_id!r} feature_version "
            f"{artifact.feature_version!r} does not match this deployment's "
            f"{expected_feature_version!r}"
        )

    try:
        boundary = calendar.sessions_offset(as_of, -max_age_sessions)
    except (CalendarCoverageError, ValueError) as exc:
        problems.append(f"could not verify model age against the calendar: {exc}")
    else:
        if training_result.training_end < boundary:
            problems.append(
                f"model {artifact.model_id!r} was trained through "
                f"{training_result.training_end}, more than {max_age_sessions} trading "
                f"sessions before {as_of} -- exceeds the configured retrain interval"
            )

    return problems
