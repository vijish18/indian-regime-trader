"""Phase 1 must not silently implement trading logic. This is a spot-check,
not exhaustive coverage of every stub, that a representative method from
each of the nine separated layers still raises NotImplementedError with a
message identifying which phase will implement it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.loader import load_settings


def test_position_sizer_is_unimplemented() -> None:
    """Portfolio construction and risk management (approve/veto) are
    implemented; converting an approved target weight into a final order
    quantity is still later work (Phase 7c).
    """
    from risk.position_sizer import PositionSizer

    settings = load_settings()
    sizer = PositionSizer(settings.risk)
    with pytest.raises(NotImplementedError, match="Phase 7c"):
        sizer.weight_based_quantity(proposed=None, equity=100.0, price=10.0)  # type: ignore[arg-type]


def test_backtest_engine_is_unimplemented() -> None:
    from backtest.engine import BacktestEngine

    with pytest.raises(NotImplementedError, match="Phase 8"):
        BacktestEngine().run(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-01"), "model-1")


def test_broker_interface_methods_are_unimplemented() -> None:
    from broker.adapters.paper_broker import PaperBroker

    broker = PaperBroker()
    with pytest.raises(NotImplementedError, match="Phase 10/11"):
        broker.health_check()
