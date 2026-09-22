from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from backtest.costs import TradeSide
from backtest.engine import OrderRecord
from data.market_data import parse_bar_rows
from data.models import PriceBasis
from data.nse_bhavcopy import VALUATION_SERIES, BhavcopyError, parse_bhavcopy
from portfolio.portfolio_constructor import TradeAction
from scripts.build_equity_bars import price_rows
from tests.unit._wf_support import Environment
from tests.unit.test_backtest_engine import full_exposure_target
from tests.unit.test_nse_bhavcopy import NEW_CSV, OLD_CSV, _zip
from universe.bhavcopy_universe import EligibilityRules, build_snapshots


@pytest.mark.parametrize("layout", [OLD_CSV, NEW_CSV])
@pytest.mark.parametrize("series", ["BE", "BZ"])
def test_restricted_prices_are_opt_in_and_never_eligible(layout: str, series: str) -> None:
    # Keep a normal EQ row so the default parser has a valid universe.
    header, normal, *_ = layout.splitlines()
    restricted = normal.replace("RELIANCE", "OTHER").replace(",EQ,", f",{series},")
    payload = _zip("\n".join([header, normal, restricted]) + "\n")
    assert [r.symbol for r in parse_bhavcopy(payload)] == ["RELIANCE"]
    rows = parse_bhavcopy(payload, allowed_series=VALUATION_SERIES)
    assert [r.series for r in rows] == ["EQ", series]
    rules = EligibilityRules(Decimal(0), lookback_sessions=1, min_sessions_traded=1)
    snapshot = build_snapshots({rows[0].session_date: rows}, rules)[0]
    assert snapshot.eligible == frozenset({"NSE:RELIANCE"})


def test_duplicate_series_prefers_eq_without_combining_prices_or_volume() -> None:
    eq = parse_bhavcopy(_zip(OLD_CSV))[0]
    be = replace(eq, series="BE", close=Decimal(2000), volume=5)
    assert price_rows((be, eq)) == (eq,)
    assert price_rows((eq, be)) == (eq,)
    with pytest.raises(BhavcopyError, match="Duplicate"):
        price_rows((be, be))
    with pytest.raises(BhavcopyError, match="Conflicting ISIN"):
        price_rows((eq, replace(be, isin="different")))


def test_series_survives_storage_parsing_and_adjustment() -> None:
    frame = pd.DataFrame(
        [
            dict(
                instrument_id="NSE:ADANITRANS",
                session_date="2021-08-23",
                open=1157,
                high=1179.6,
                low=1150,
                close=1179.6,
                volume=160423,
                price_basis="raw",
                data_source="nse_bhavcopy",
                trading_series="BE",
            )
        ]
    )
    (bar,) = parse_bar_rows(frame, "test")
    assert bar.trading_series == "BE"
    assert bar.adjusted(Decimal("0.5")).trading_series == "BE"
    assert bar.price_basis == PriceBasis.RAW


def test_series_transition_allows_exit_and_rejects_new_buy(tmp_path: Path) -> None:
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path)
    signal = env.dates[80]
    execution = env.dates[81]
    instrument = env.instrument_ids[0]
    env.market_data._bars[instrument] = [
        replace(b, trading_series="BE") if b.session_date == execution else b
        for b in env.market_data._bars[instrument]
    ]
    buy = OrderRecord(signal, execution, instrument, TradeAction.BUY, 0, 0.2, 0.2)
    assert engine._execute(buy, 100000, {}, 100000) is None
    sell = replace(
        buy, action=TradeAction.EXIT, current_weight=0.2, target_weight=0, delta_weight=-0.2
    )
    fill = engine._execute(sell, 0, {instrument: 10}, 100000)
    assert fill is not None and fill.side is TradeSide.SELL and fill.quantity == 10


def test_fold_liquidates_preexisting_holding_after_series_transition(tmp_path: Path) -> None:
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path)
    engine.liquidate_at_end = True
    days = env.dates[80:83]
    instrument = env.instrument_ids[0]
    last_execution = env.calendar.next_trading_day(days[-1])
    env.market_data._bars[instrument] = [
        replace(b, trading_series="BE") if b.session_date == last_execution else b
        for b in env.market_data._bars[instrument]
    ]
    result = engine.run(
        "transition", {d: full_exposure_target(d, env.regime_policy) for d in days}, days, 100000
    )
    sales = result.trade_log[result.trade_log.exit_reason == "fold_end_liquidation"]
    assert not sales.empty
    assert set(sales.execution_date) == {last_execution}
    assert result.cash_history.iloc[-1] == result.equity_curve.iloc[-1]


def test_missing_session_cannot_fill_using_a_future_bar(tmp_path: Path) -> None:
    env = Environment(n_days=120, n_stocks=1)
    engine = env.engine(tmp_path)
    signal, execution = env.dates[80:82]
    instrument = env.instrument_ids[0]
    env.market_data._bars[instrument] = [
        b for b in env.market_data._bars[instrument] if b.session_date != execution
    ]
    assert engine._next_open(instrument, execution) is None
    order = OrderRecord(signal, execution, instrument, TradeAction.BUY, 0, 0.2, 0.2)
    assert engine._execute(order, 100000, {}, 100000) is None
    assert engine._next_open(instrument, env.dates[82]) is not None


@pytest.mark.parametrize("problem", ["old_export", "duplicate"])
def test_preflight_rejects_incomplete_or_ambiguous_series_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    from scripts import preflight_backtest

    root = tmp_path / "raw" / "equity_bars"
    root.mkdir(parents=True)
    row = {"instrument_id": "NSE:TEST", "session_date": "2021-08-23"}
    if problem == "duplicate":
        row["trading_series"] = "BE"
    pd.DataFrame([row, row]).to_csv(root / "NSE_TEST.csv", index=False)
    monkeypatch.setattr(preflight_backtest, "DATA_CACHE", tmp_path)
    assert preflight_backtest.check_series_history().status == preflight_backtest.FAIL
