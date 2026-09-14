"""Phase 1 must not silently implement trading logic. This is a spot-check,
not exhaustive coverage of every stub, that a representative method from
each of the nine separated layers still raises NotImplementedError with a
message identifying which phase will implement it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.loader import load_settings


def test_hmm_engine_fit_is_unimplemented() -> None:
    from core.regime.hmm_engine import HMMRegimeEngine

    engine = HMMRegimeEngine(n_states=3, covariance_type="diag", random_seed=0)
    with pytest.raises(NotImplementedError, match="Phase 5"):
        engine.fit(pd.DataFrame())


def test_feature_engineering_is_unimplemented() -> None:
    from core.features.feature_engineering import FeatureEngineer

    with pytest.raises(NotImplementedError, match="Phase 4"):
        FeatureEngineer().realized_volatility(pd.Series(dtype=float), window=20)


def test_stock_selector_is_unimplemented() -> None:
    from universe.stock_selector import StockSelector

    settings = load_settings()
    selector = StockSelector(settings.selection)
    with pytest.raises(NotImplementedError, match="Phase 6"):
        selector.liquidity_filter(universe=None, as_of=None)  # type: ignore[arg-type]


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
