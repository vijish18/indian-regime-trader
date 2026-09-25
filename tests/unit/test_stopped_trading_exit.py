"""A held stock that stops trading is exited at its last real trade.

JSLHISAR last traded on 2023-03-08 before merging into JSL. Every strategy
still held it at the fold end on 2023-06-05, there was no bar to sell into,
and all ten runs of the one-rule backtest aborted at fold 21. These tests pin
the replacement: exit at the last price that actually printed, tagged so every
instance can be audited, and only after the stock has been silent long enough
to know it has stopped -- a short suspension must resume normally.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from tests.unit._wf_support import Environment
from tests.unit.test_backtest_engine import full_exposure_target

TAG = "stopped_trading_last_close"


class _GoesSilent:
    """Market data in which one stock prints no bar in ``(last, resume)``."""

    def __init__(self, inner: Any, instrument_id: str, last: dt.date,
                 resume: dt.date | None = None) -> None:
        self._inner = inner
        self._id = instrument_id
        self._last = last
        self._resume = resume

    def get_equity_bars(self, instrument_id: str, start: dt.date, end: dt.date,
                        *args: Any, **kwargs: Any) -> list[Any]:
        bars: list[Any] = self._inner.get_equity_bars(instrument_id, start, end, *args, **kwargs)
        if instrument_id != self._id:
            return bars
        return [
            b for b in bars
            if b.session_date <= self._last
            or (self._resume is not None and b.session_date >= self._resume)
        ]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _run(tmp_path: Path, last_index: int, resume_index: int | None = None,
         liquidate: bool = False, n: int = 20) -> tuple[Environment, Any, list[dt.date]]:
    env = Environment(n_days=140, n_stocks=1)
    dates = env.dates[80:80 + n]
    iid = env.instrument_ids[0]
    engine = env.engine(tmp_path, liquidate_at_end=liquidate)
    engine.market_data = _GoesSilent(  # type: ignore[assignment]
        env.market_data, iid, dates[last_index],
        dates[resume_index] if resume_index is not None else None,
    )
    targets = {d: full_exposure_target(d, env.regime_policy) for d in dates}
    return env, engine.run("silent", targets, dates, 100_000.0), dates


def _raw_close(env: Environment, day: dt.date) -> float:
    from data.models import PriceBasis

    (bar,) = env.market_data.get_equity_bars(env.instrument_ids[0], day, day,
                                             price_basis=PriceBasis.RAW)
    return float(bar.close)


def test_a_stock_that_stops_trading_is_exited_at_its_last_real_trade(tmp_path: Path) -> None:
    env, result, dates = _run(tmp_path, last_index=5)
    exits = result.trade_log[result.trade_log.get("exit_reason") == TAG]

    assert len(exits) == 1
    exit_row = exits.iloc[0]
    assert exit_row.last_trade_date == dates[5].isoformat()
    assert exit_row.fill_price == _raw_close(env, dates[5])
    assert exit_row.cost > 0
    # The position is gone and stays gone: the stock cannot be bought again
    # without a bar, so the book is flat for the rest of the run.
    assert result.positions_history[max(result.positions_history)] == {}


def test_it_waits_until_the_silence_is_long_enough_to_know(tmp_path: Path) -> None:
    """Knowable on the day: nothing is exited before five silent sessions."""
    env, result, dates = _run(tmp_path, last_index=5)
    exit_row = result.trade_log[result.trade_log.get("exit_reason") == TAG].iloc[0]
    silent = len(env.calendar.trading_days_between(dates[5], exit_row.execution_date)) - 1

    assert silent >= 5


def test_a_short_suspension_resumes_without_an_exit(tmp_path: Path) -> None:
    """Silent for three sessions, then trading again: a halt, not a stop."""
    _, result, _ = _run(tmp_path, last_index=5, resume_index=9)

    assert "exit_reason" not in result.trade_log or (
        result.trade_log["exit_reason"] != TAG
    ).all()


def test_the_fold_end_uses_the_last_trade_instead_of_aborting(tmp_path: Path) -> None:
    """Stopped two sessions before the fold ends -- too soon for the
    in-session rule -- the fold-end liquidation used to raise here."""
    env, result, dates = _run(tmp_path, last_index=17, liquidate=True)
    exits = result.trade_log[result.trade_log.get("exit_reason") == TAG]

    assert len(exits) == 1
    assert exits.iloc[0].fill_price == _raw_close(env, dates[17])
    assert result.positions_history[max(result.positions_history)] == {}
