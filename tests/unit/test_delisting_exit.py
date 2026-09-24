from dataclasses import replace

import pandas as pd
import pytest

from data.delisting import DelistingNotice
from tests.unit._wf_support import Environment
from tests.unit.test_backtest_engine import full_exposure_target


def test_announced_exit_is_causal_and_prevents_reentry(tmp_path) -> None:
    env = Environment(n_days=120, n_stocks=1)
    dates = env.dates[80:90]
    known = dates[3]
    notice = DelistingNotice(env.instrument_ids[0], known, dates[7], "https://example.test/notice")
    engine = env.engine(tmp_path, delisting_notices=(notice,))
    result = engine.run(
        "delist", {d: full_exposure_target(d, env.regime_policy) for d in dates}, dates, 100_000
    )
    buys = result.trade_log[result.trade_log.side == "buy"]
    exits = result.trade_log[result.trade_log.exit_reason == "announced_delisting_exit"]
    assert not buys.empty
    assert all(d < known for d in buys.signal_date)
    assert len(exits) == 1
    assert exits.iloc[0].signal_date == known
    assert exits.iloc[0].execution_date == env.calendar.next_trading_day(known)
    assert exits.iloc[0].cost > 0
    assert result.positions_history[max(result.positions_history)] == {}


def test_retry_unfilled_delisting_exit_and_resume_exactly(tmp_path, monkeypatch) -> None:
    env = Environment(n_days=120, n_stocks=1)
    dates = env.dates[80:90]
    known = dates[3]
    notice = DelistingNotice(env.instrument_ids[0], known, dates[8], "https://example.test/notice")
    targets = {d: full_exposure_target(d, env.regime_policy) for d in dates}

    def configured(folder, checkpoint=False):
        engine = env.engine(folder, delisting_notices=(notice,))
        if checkpoint:
            engine.checkpoint_dir = tmp_path / "sessions"
        original = engine._session_bar
        engine._session_bar = lambda iid, day: (
            None if day == env.calendar.next_trading_day(known) else original(iid, day)
        )
        return engine

    whole = configured(tmp_path / "whole").run("test", targets, dates, 100_000)
    exits = whole.trade_log[whole.trade_log.exit_reason == "announced_delisting_exit"]
    assert len(exits) == 1
    assert exits.iloc[0].execution_date == env.calendar.next_trading_day(dates[4])
    broken = configured(tmp_path / "broken", checkpoint=True)
    select = broken.stock_selector.select

    def fail(day):
        if day == dates[5]:
            raise RuntimeError("power loss")
        return select(day)

    with monkeypatch.context() as patch:
        patch.setattr(broken.stock_selector, "select", fail)
        with pytest.raises(RuntimeError, match="power loss"):
            broken.run("test", targets, dates, 100_000)
    resumed = configured(tmp_path / "broken", checkpoint=True)
    actual = resumed.run("test", targets, dates, 100_000)
    pd.testing.assert_frame_equal(whole.trade_log, actual.trade_log)
    pd.testing.assert_series_equal(whole.equity_curve, actual.equity_curve)
    resumed.delisting_notices = (replace(notice, known_on=dates[2]),)
    with pytest.raises(ValueError, match="identity changed"):
        resumed.run("test", targets, dates, 100_000)
