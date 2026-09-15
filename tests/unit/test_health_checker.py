"""Unit tests for ``monitoring/health.py`` (implemented in Phase 19): the
dedicated module the orchestrator's "monitor health" step calls into --
broker connectivity, market-data freshness, instrument-master freshness,
model freshness, and process heartbeat.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from broker.base import (
    Broker,
    BrokerAccount,
    BrokerCapabilities,
    BrokerFill,
    BrokerOrder,
    BrokerPosition,
    BrokerQuote,
    HealthStatus,
)
from core.features.feature_engineering import MarketFeatureInputs, drop_warmup_rows
from core.features.feature_scaler import CausalFeatureScaler
from core.regime.hmm_engine import HMMRegimeEngine
from core.regime.model_registry import ModelArtifact, ModelRegistry, build_model_id
from monitoring.health import ComponentHealth, HealthChecker
from orchestration.heartbeat import Heartbeat
from tests.unit._wf_support import INDEX_SYMBOL, VIX_SYMBOL, Environment


class _StubBroker(Broker):
    def __init__(self) -> None:
        self.healthy = True
        self.detail = "ok"

    def capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        raise NotImplementedError

    def get_account(self) -> BrokerAccount:
        raise NotImplementedError

    def get_positions(self) -> list[BrokerPosition]:
        raise NotImplementedError

    def get_open_orders(self) -> list[BrokerOrder]:
        raise NotImplementedError

    def get_order(self, order_id: str) -> BrokerOrder:
        raise NotImplementedError

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        raise NotImplementedError

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        raise NotImplementedError

    def subscribe_market_data(
        self, instrument_ids: list[str], on_tick: Callable[[BrokerQuote], None]
    ) -> Callable[[], None]:
        raise NotImplementedError

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        raise NotImplementedError

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> BrokerOrder:
        raise NotImplementedError

    def close_position(self, instrument_id: str) -> BrokerOrder:
        raise NotImplementedError

    def close_all_positions(self) -> list[BrokerOrder]:
        raise NotImplementedError

    def health_check(self) -> HealthStatus:
        return HealthStatus(
            healthy=self.healthy,
            detail=self.detail,
            checked_at=dt.datetime(2015, 6, 1, tzinfo=dt.UTC),
            session_active=self.healthy,
        )


class _ClockBox:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start

    def __call__(self) -> dt.datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + dt.timedelta(**kwargs)


@pytest.fixture(scope="module")
def env() -> Environment:
    return Environment(n_days=160)


@pytest.fixture
def broker() -> _StubBroker:
    return _StubBroker()


@pytest.fixture
def clock(env: Environment) -> _ClockBox:
    return _ClockBox(dt.datetime.combine(env.dates[-1], dt.time(16, 0), tzinfo=dt.UTC))


def _approved_registry(env: Environment, tmp_path: Path, train_end: dt.date) -> ModelRegistry:
    nifty = env.market_data.get_index_observations(INDEX_SYMBOL, env.start, train_end)
    vix = env.market_data.get_index_observations(VIX_SYMBOL, env.start, train_end)
    inputs = MarketFeatureInputs.from_index_observations(nifty, vix)
    matrix = drop_warmup_rows(env.feature_pipeline.compute(inputs))
    scaled, scaler_params = CausalFeatureScaler().fit_transform(matrix)
    returns = inputs.frame["nifty_close"].pct_change().reindex(scaled.index)
    model = HMMRegimeEngine(env.hmm_cfg).fit(scaled, returns)

    artifact = ModelArtifact(
        model_id=build_model_id(
            model.training_result.training_end, model.n_states, model.training_result.seed, "fv1"
        ),
        created_at=dt.datetime.combine(train_end, dt.time(18, 0), tzinfo=dt.UTC),
        model=model,
        scaler=scaler_params,
        feature_version="fv1",
    )
    registry = ModelRegistry(tmp_path / "models")
    registry.save(artifact)
    registry.approve(artifact.model_id)
    return registry


def _checker(
    env: Environment,
    broker: Broker,
    registry: ModelRegistry,
    clock: _ClockBox,
    **overrides: object,
) -> HealthChecker:
    kwargs: dict[str, object] = dict(
        max_stale_sessions=1,
        model_max_age_sessions=40,
        heartbeat_interval_seconds=300.0,
        clock=clock,
    )
    kwargs.update(overrides)
    return HealthChecker(
        broker,
        env.market_data,
        env.universe_provider,
        registry,
        env.calendar,
        env.instrument_ids,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# check_broker
# --------------------------------------------------------------------------


def test_healthy_broker_reports_healthy(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    checker = _checker(env, broker, ModelRegistry(tmp_path / "models"), clock)
    report = checker.check_broker()
    assert report.component == "broker"
    assert report.status is ComponentHealth.HEALTHY


def test_unhealthy_broker_reports_unhealthy_with_the_broker_s_own_detail(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    broker.healthy = False
    broker.detail = "session expired"
    checker = _checker(env, broker, ModelRegistry(tmp_path / "models"), clock)
    report = checker.check_broker()
    assert report.status is ComponentHealth.UNHEALTHY
    assert report.detail == "session expired"


# --------------------------------------------------------------------------
# check_market_data_freshness
# --------------------------------------------------------------------------


def test_fresh_market_data_reports_healthy(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    checker = _checker(env, broker, ModelRegistry(tmp_path / "models"), clock)
    report = checker.check_market_data_freshness(env.dates[-1])
    assert report.status is ComponentHealth.HEALTHY


def test_stale_market_data_reports_unhealthy(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    checker = _checker(env, broker, ModelRegistry(tmp_path / "models"), clock)
    # Ask about a date far past the end of the synthetic dataset.
    report = checker.check_market_data_freshness(env.dates[-1] + dt.timedelta(days=60))
    assert report.status is ComponentHealth.UNHEALTHY


def test_partially_missing_instruments_report_degraded(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    checker = _checker(
        env,
        broker,
        ModelRegistry(tmp_path / "models"),
        clock,
    )
    checker.instrument_ids = (*env.instrument_ids, "NSE:NOT-INGESTED")
    report = checker.check_market_data_freshness(env.dates[-1])
    assert report.status is ComponentHealth.DEGRADED
    assert "NSE:NOT-INGESTED" in report.detail


def test_no_configured_instruments_reports_unhealthy(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    checker = _checker(env, broker, ModelRegistry(tmp_path / "models"), clock)
    checker.instrument_ids = ()
    report = checker.check_market_data_freshness(env.dates[-1])
    assert report.status is ComponentHealth.UNHEALTHY


# --------------------------------------------------------------------------
# check_instrument_master_freshness
# --------------------------------------------------------------------------


def test_instrument_master_with_an_eligible_universe_reports_healthy(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    checker = _checker(env, broker, ModelRegistry(tmp_path / "models"), clock)
    report = checker.check_instrument_master_freshness(env.dates[-1])
    assert report.status is ComponentHealth.HEALTHY
    assert str(len(env.instrument_ids)) in report.detail


# --------------------------------------------------------------------------
# check_model_freshness
# --------------------------------------------------------------------------


def test_no_approved_model_reports_unhealthy(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    checker = _checker(env, broker, ModelRegistry(tmp_path / "models"), clock)
    report = checker.check_model_freshness(env.dates[-1])
    assert report.status is ComponentHealth.UNHEALTHY
    assert "no approved model" in report.detail


def test_recently_trained_approved_model_reports_healthy(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    registry = _approved_registry(env, tmp_path, env.dates[-3])
    checker = _checker(env, broker, registry, clock)
    report = checker.check_model_freshness(env.dates[-1])
    assert report.status is ComponentHealth.HEALTHY


def test_an_old_approved_model_reports_degraded_not_unhealthy(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    registry = _approved_registry(env, tmp_path, env.dates[100])
    checker = _checker(env, broker, registry, clock, model_max_age_sessions=5)
    report = checker.check_model_freshness(env.dates[-1])
    assert report.status is ComponentHealth.DEGRADED
    assert "freshness window" in report.detail


# --------------------------------------------------------------------------
# check_heartbeat
# --------------------------------------------------------------------------


def test_no_heartbeat_yet_is_healthy_during_startup_grace(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    checker = _checker(env, broker, ModelRegistry(tmp_path / "models"), clock)
    report = checker.check_heartbeat()
    assert report.status is ComponentHealth.HEALTHY
    assert "no heartbeat recorded yet" in report.detail


def test_a_recent_heartbeat_is_healthy(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    heartbeat = Heartbeat(clock)
    heartbeat.beat()
    checker = _checker(
        env, broker, ModelRegistry(tmp_path / "models"), clock, last_heartbeat=heartbeat.last
    )
    assert checker.check_heartbeat().status is ComponentHealth.HEALTHY


def test_a_late_heartbeat_is_degraded(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    heartbeat = Heartbeat(clock)
    heartbeat.beat()
    checker = _checker(
        env, broker, ModelRegistry(tmp_path / "models"), clock, last_heartbeat=heartbeat.last
    )
    clock.advance(seconds=400)  # > interval (300), < 3x interval
    assert checker.check_heartbeat().status is ComponentHealth.DEGRADED


def test_a_very_late_heartbeat_is_unhealthy(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    heartbeat = Heartbeat(clock)
    heartbeat.beat()
    checker = _checker(
        env, broker, ModelRegistry(tmp_path / "models"), clock, last_heartbeat=heartbeat.last
    )
    clock.advance(seconds=1000)  # > 3x interval
    assert checker.check_heartbeat().status is ComponentHealth.UNHEALTHY


# --------------------------------------------------------------------------
# run_startup_sequence
# --------------------------------------------------------------------------


def test_startup_sequence_runs_every_check_in_order(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    registry = _approved_registry(env, tmp_path, env.dates[-3])
    checker = _checker(env, broker, registry, clock)
    reports = checker.run_startup_sequence(env.dates[-1])
    assert [r.component for r in reports] == [
        "broker",
        "market_data",
        "instrument_master",
        "model",
        "heartbeat",
    ]
    assert all(r.status is ComponentHealth.HEALTHY for r in reports)


def test_startup_sequence_reports_rather_than_raises_when_something_is_wrong(
    env: Environment, broker: _StubBroker, clock: _ClockBox, tmp_path: Path
) -> None:
    broker.healthy = False
    broker.detail = "disconnected"
    checker = _checker(env, broker, ModelRegistry(tmp_path / "models"), clock)
    reports = checker.run_startup_sequence(env.dates[-1])
    unhealthy = {r.component for r in reports if r.status is ComponentHealth.UNHEALTHY}
    assert unhealthy == {"broker", "model"}
