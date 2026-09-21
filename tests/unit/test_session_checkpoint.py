from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.unit._wf_support import Environment
from tests.unit.test_backtest_engine import full_exposure_target


def test_crash_resumes_after_last_session_without_replaying_fills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = Environment(n_days=120, n_stocks=3)
    dates = env.dates[80:90]
    targets = {d: full_exposure_target(d, env.regime_policy) for d in dates}
    whole = env.engine(tmp_path / "whole").run("test", targets, dates, 100000)
    interrupted = env.engine(tmp_path / "crashed")
    interrupted.checkpoint_dir = tmp_path / "sessions"
    interrupted.run_identity = "immutable-data-code-config"
    original = interrupted.stock_selector.select
    calls = []

    def failing_select(day):  # type: ignore[no-untyped-def]
        calls.append(day)
        if len(calls) == 5:
            raise RuntimeError("simulated power loss")
        return original(day)

    with monkeypatch.context() as patch:
        patch.setattr(interrupted.stock_selector, "select", failing_select)
        with pytest.raises(RuntimeError, match="power loss"):
            interrupted.run("test", targets, dates, 100000)
    assert len(list((tmp_path / "sessions").glob("*.json"))) == 1
    resumed = env.engine(tmp_path / "crashed")
    resumed.checkpoint_dir = tmp_path / "sessions"
    resumed.run_identity = interrupted.run_identity
    result = resumed.run("test", targets, dates, 100000)
    pd.testing.assert_series_equal(result.equity_curve, whole.equity_curve)
    pd.testing.assert_series_equal(result.cash_history, whole.cash_history)
    pd.testing.assert_frame_equal(result.trade_log, whole.trade_log)
    assert result.positions_history == whole.positions_history
    resumed.run_identity = "different-code"
    with pytest.raises(ValueError, match="identity changed"):
        resumed.run("test", targets, dates, 100000)


def test_fold_end_liquidation_records_sales_costs_and_cash(tmp_path: Path) -> None:
    env = Environment(n_days=120, n_stocks=3)
    dates = env.dates[80:85]
    targets = {d: full_exposure_target(d, env.regime_policy) for d in dates}
    engine = env.engine(tmp_path)
    engine.liquidate_at_end = True
    result = engine.run("test", targets, dates, 100000)
    terminal = result.trade_log[result.trade_log.exit_reason == "fold_end_liquidation"]
    assert not terminal.empty
    assert (terminal.cost > 0).all()
    assert result.cash_history.iloc[-1] == result.equity_curve.iloc[-1]
    assert sum(result.positions_history[max(result.positions_history)].values()) == 0
    buys = result.trade_log[result.trade_log.side == "buy"].groupby("instrument_id").quantity.sum()
    sells = (
        result.trade_log[result.trade_log.side == "sell"].groupby("instrument_id").quantity.sum()
    )
    pd.testing.assert_series_equal(buys, sells)
