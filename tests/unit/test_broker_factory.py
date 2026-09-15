"""Unit tests for ``broker/factory.py`` (Phase 15) -- the single place
that decides paper vs. live. Default mode must stay PAPER, and
constructing a live broker requires both an explicit config value and an
explicit code-level confirmation; neither alone is enough.
"""

from __future__ import annotations

import pytest

from backtest.cost_schedule import CostScheduleRepository
from backtest.costs import CostModel
from broker.adapters.paper_broker import PaperBroker
from broker.factory import BrokerFactoryError, build_broker
from broker.zerodha.kite_broker import KiteBroker
from config.loader import load_settings
from config.models import Settings
from tests.unit._wf_support import FakeMarketDataProvider, cost_schedule


@pytest.fixture
def settings() -> Settings:
    return load_settings()


@pytest.fixture
def market_data() -> FakeMarketDataProvider:
    return FakeMarketDataProvider()


@pytest.fixture
def cost_model() -> CostModel:
    return CostModel(
        CostScheduleRepository([cost_schedule()]), min_slippage_bps=2.0, impact_coefficient=10.0
    )


def _with_execution_mode(settings: Settings, mode: str) -> Settings:
    return settings.model_copy(
        update={"execution": settings.execution.model_copy(update={"mode": mode})}
    )


def _with_broker_provider(settings: Settings, provider: str) -> Settings:
    updated_broker = settings.broker.model_copy(update={"provider": provider})
    return settings.model_copy(update={"broker": updated_broker})


def test_default_settings_produce_a_paper_broker(
    settings: Settings, market_data: FakeMarketDataProvider, cost_model: CostModel
) -> None:
    broker = build_broker(settings, market_data, cost_model, initial_cash=1_000_000.0)
    assert isinstance(broker, PaperBroker)


def test_paper_mode_never_needs_live_trading_confirmation(
    settings: Settings, market_data: FakeMarketDataProvider, cost_model: CostModel
) -> None:
    paper_settings = _with_execution_mode(settings, "paper")
    broker = build_broker(paper_settings, market_data, cost_model, initial_cash=1_000_000.0)
    assert isinstance(broker, PaperBroker)


def test_live_mode_with_an_unsupported_provider_raises(
    settings: Settings, market_data: FakeMarketDataProvider, cost_model: CostModel
) -> None:
    live_settings = _with_broker_provider(_with_execution_mode(settings, "live"), "paper")
    with pytest.raises(BrokerFactoryError, match="no live adapter"):
        build_broker(live_settings, market_data, cost_model, initial_cash=1_000_000.0)


def test_live_mode_without_explicit_confirmation_raises(
    settings: Settings, market_data: FakeMarketDataProvider, cost_model: CostModel
) -> None:
    live_settings = _with_broker_provider(_with_execution_mode(settings, "live"), "zerodha")
    with pytest.raises(BrokerFactoryError, match="enable_live_trading"):
        build_broker(live_settings, market_data, cost_model, initial_cash=1_000_000.0)


def test_live_mode_confirmed_but_missing_credentials_raises(
    settings: Settings,
    market_data: FakeMarketDataProvider,
    cost_model: CostModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BROKER_API_KEY", raising=False)
    monkeypatch.delenv("BROKER_API_SECRET", raising=False)
    live_settings = _with_broker_provider(_with_execution_mode(settings, "live"), "zerodha")
    with pytest.raises(BrokerFactoryError, match="BROKER_API_KEY"):
        build_broker(
            live_settings,
            market_data,
            cost_model,
            initial_cash=1_000_000.0,
            enable_live_trading=True,
        )


def test_live_mode_fully_confirmed_with_credentials_builds_a_kite_broker(
    settings: Settings,
    market_data: FakeMarketDataProvider,
    cost_model: CostModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BROKER_API_KEY", "testkey")
    monkeypatch.setenv("BROKER_API_SECRET", "testsecret")
    live_settings = _with_broker_provider(_with_execution_mode(settings, "live"), "zerodha")

    broker = build_broker(
        live_settings,
        market_data,
        cost_model,
        initial_cash=1_000_000.0,
        enable_live_trading=True,
    )
    assert isinstance(broker, KiteBroker)
    assert broker.live_trading_enabled is True
