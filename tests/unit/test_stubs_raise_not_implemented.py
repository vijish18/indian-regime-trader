"""Phase 1 must not silently implement trading logic. This is a spot-check,
not exhaustive coverage of every stub, that a representative method from
each of the nine separated layers still raises NotImplementedError with a
message identifying which phase will implement it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.loader import load_settings


def test_portfolio_constructor_is_unimplemented() -> None:
    """Stock selection is implemented (Phase 7), but turning ranked
    candidates into target weights is still later work.
    """
    from portfolio.portfolio_constructor import PortfolioConstructor

    settings = load_settings()
    constructor = PortfolioConstructor(settings.portfolio)
    with pytest.raises(NotImplementedError, match="Phase 7"):
        constructor.raw_weights(candidates=[], volatilities={})


def test_risk_manager_is_unimplemented() -> None:
    from risk.risk_manager import RiskManager

    settings = load_settings()
    manager = RiskManager(settings.risk)
    with pytest.raises(NotImplementedError, match="Phase 7"):
        manager.evaluate(proposed_weights=[], current_exposure=None, circuit_status=None)  # type: ignore[arg-type]


def test_backtest_engine_is_unimplemented() -> None:
    from backtest.engine import BacktestEngine

    with pytest.raises(NotImplementedError, match="Phase 8"):
        BacktestEngine().run(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01"), "model-1")


def test_broker_interface_methods_are_unimplemented() -> None:
    from broker.adapters.paper_broker import PaperBroker

    broker = PaperBroker()
    with pytest.raises(NotImplementedError, match="Phase 10/11"):
        broker.health_check()
