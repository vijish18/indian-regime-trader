"""Unit tests for ``orchestration/regime_computation.py`` (Phase 19):
``RegimeComputer`` filters (never re-fits) an already-approved model
through today's trailing window and turns the result into an
``AllocationTarget`` -- mirroring
``backtest.walk_forward.WalkForwardValidator._hmm_exposure_targets``'s own
inference recipe for live use.
"""

from __future__ import annotations

import datetime as dt

import pytest

from core.features.feature_engineering import MarketFeatureInputs, drop_warmup_rows
from core.features.feature_scaler import CausalFeatureScaler
from core.regime.hmm_engine import HMMRegimeEngine
from core.regime.model_registry import ModelArtifact, build_model_id
from orchestration.regime_computation import RegimeComputationError, RegimeComputer
from tests.unit._wf_support import INDEX_SYMBOL, VIX_SYMBOL, Environment


def _fit_artifact(env: Environment, train_end: dt.date) -> ModelArtifact:
    """Mirrors ``WalkForwardValidator._fit_fold`` exactly -- train on
    ``[env.start, train_end]`` only, nothing after it.
    """
    nifty = env.market_data.get_index_observations(INDEX_SYMBOL, env.start, train_end)
    vix = env.market_data.get_index_observations(VIX_SYMBOL, env.start, train_end)
    inputs = MarketFeatureInputs.from_index_observations(nifty, vix)
    matrix = drop_warmup_rows(env.feature_pipeline.compute(inputs))
    scaler = CausalFeatureScaler()
    scaled, scaler_params = scaler.fit_transform(matrix)
    returns = inputs.frame["nifty_close"].pct_change().reindex(scaled.index)

    engine = HMMRegimeEngine(env.hmm_cfg)
    model = engine.fit(scaled, returns)
    return ModelArtifact(
        model_id=build_model_id(
            model.training_result.training_end, model.n_states, model.training_result.seed, "fv1"
        ),
        created_at=dt.datetime.combine(train_end, dt.time(18, 0), tzinfo=dt.UTC),
        model=model,
        scaler=scaler_params,
        feature_version="fv1",
    )


@pytest.fixture(scope="module")
def env() -> Environment:
    return Environment(n_days=260)


@pytest.fixture(scope="module")
def train_end(env: Environment) -> dt.date:
    return env.dates[179]


@pytest.fixture(scope="module")
def artifact(env: Environment, train_end: dt.date) -> ModelArtifact:
    return _fit_artifact(env, train_end)


@pytest.fixture
def computer(env: Environment) -> RegimeComputer:
    return RegimeComputer(
        env.market_data,
        env.hmm_cfg,
        env.allocation_cfg,
        env.regime_policy,
        env.feature_pipeline,
        INDEX_SYMBOL,
        VIX_SYMBOL,
        feature_warmup_buffer_days=120,
    )


def test_compute_today_returns_a_target_and_state_for_the_latest_date(
    computer: RegimeComputer, env: Environment, artifact: ModelArtifact
) -> None:
    as_of = env.dates[-1]
    target, state = computer.compute_today(artifact, as_of)
    assert target.as_of == as_of
    assert state.as_of == as_of
    assert 0.0 <= target.target_gross_exposure <= 1.0


def test_compute_today_works_for_a_date_shortly_after_training(
    computer: RegimeComputer, env: Environment, artifact: ModelArtifact, train_end: dt.date
) -> None:
    as_of = env.dates[181]
    assert as_of > train_end
    target, state = computer.compute_today(artifact, as_of)
    assert target.as_of == as_of
    assert state.as_of == as_of


def test_compute_today_raises_for_a_date_with_no_index_data(
    computer: RegimeComputer, env: Environment, artifact: ModelArtifact
) -> None:
    far_future = env.dates[-1] + dt.timedelta(days=30)
    with pytest.raises(RegimeComputationError):
        computer.compute_today(artifact, far_future)


def test_compute_today_raises_when_no_history_exists_at_all(
    env: Environment, artifact: ModelArtifact
) -> None:
    computer = RegimeComputer(
        env.market_data,
        env.hmm_cfg,
        env.allocation_cfg,
        env.regime_policy,
        env.feature_pipeline,
        INDEX_SYMBOL,
        VIX_SYMBOL,
        feature_warmup_buffer_days=1,
    )
    with pytest.raises(RegimeComputationError):
        computer.compute_today(artifact, env.dates[-1])
