"""Pairing fills into closed round trips.

The properties that matter are arithmetic ones: a trip's net P&L must be
what the account actually kept, and the trips together must account for
every share bought. Both are asserted against hand-computed numbers rather
than against the implementation's own output.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from backtest.trade_analysis import (
    RoundTrip,
    by_instrument,
    round_trips,
    summarize,
)


def _log(rows: list[tuple[str, str, str, int, float, float]]) -> pd.DataFrame:
    """(execution_date, instrument, side, quantity, price, cost)."""
    return pd.DataFrame(
        [
            {
                "execution_date": dt.date.fromisoformat(day),
                "instrument_id": instrument,
                "side": side,
                "quantity": quantity,
                "fill_price": price,
                "cost": cost,
            }
            for day, instrument, side, quantity, price, cost in rows
        ]
    )


def test_a_buy_then_sell_is_one_round_trip_net_of_both_legs() -> None:
    """100 shares bought at 100 and sold at 110 gain 1,000 gross. Both legs
    cost 50, so the account keeps 900 -- and that, not the 1,000, is what
    decides whether the call was right."""
    closed, open_positions = round_trips(
        _log(
            [
                ("2024-01-02", "NSE:ACME", "buy", 100, 100.0, 50.0),
                ("2024-02-01", "NSE:ACME", "sell", 100, 110.0, 50.0),
            ]
        )
    )

    assert open_positions == []
    assert len(closed) == 1
    trip = closed[0]
    assert trip.gross_pnl == pytest.approx(1000.0)
    assert trip.cost == pytest.approx(100.0)
    assert trip.net_pnl == pytest.approx(900.0)
    assert trip.holding_days == 30
    assert trip.is_win


def test_a_gain_smaller_than_costs_is_not_a_win() -> None:
    """The distinction the whole module exists for. Price went up and the
    account went down; counting this as a correct call would report an
    accuracy the P&L does not support."""
    closed, _ = round_trips(
        _log(
            [
                ("2024-01-02", "NSE:ACME", "buy", 10, 100.0, 60.0),
                ("2024-01-20", "NSE:ACME", "sell", 10, 103.0, 60.0),
            ]
        )
    )

    assert closed[0].gross_pnl == pytest.approx(30.0)
    assert closed[0].net_pnl == pytest.approx(-90.0)
    assert not closed[0].is_win


def test_one_sell_closing_two_buys_splits_first_in_first_out() -> None:
    """Delivery equity settles per scrip and a sell removes the oldest
    shares, so the earlier lot must be the one closed first -- and its own
    entry price, not a blended average, decides its P&L."""
    closed, open_positions = round_trips(
        _log(
            [
                ("2024-01-02", "NSE:ACME", "buy", 50, 100.0, 0.0),
                ("2024-01-10", "NSE:ACME", "buy", 50, 120.0, 0.0),
                ("2024-02-01", "NSE:ACME", "sell", 100, 130.0, 0.0),
            ]
        )
    )

    assert open_positions == []
    assert [t.entry_price for t in closed] == [100.0, 120.0]
    assert [t.quantity for t in closed] == [50, 50]
    assert closed[0].net_pnl == pytest.approx(1500.0)
    assert closed[1].net_pnl == pytest.approx(500.0)


def test_a_partial_sell_leaves_the_rest_open() -> None:
    """Shares still held have no exit price. Marking them at the last close
    would mix an unrealised result into a realised one, so they come back
    separately and count toward no win rate."""
    closed, open_positions = round_trips(
        _log(
            [
                ("2024-01-02", "NSE:ACME", "buy", 100, 100.0, 0.0),
                ("2024-02-01", "NSE:ACME", "sell", 30, 110.0, 0.0),
            ]
        )
    )

    assert len(closed) == 1
    assert closed[0].quantity == 30
    assert len(open_positions) == 1
    assert open_positions[0].quantity == 70
    assert open_positions[0].entry_price == 100.0


def test_costs_are_charged_pro_rata_to_the_quantity_matched() -> None:
    """A 100-share buy costing 100 that is closed 30 at a time must carry
    30 of that cost to the first trip, not all of it -- otherwise the first
    exit looks ruinous and the later ones free."""
    closed, _ = round_trips(
        _log(
            [
                ("2024-01-02", "NSE:ACME", "buy", 100, 100.0, 100.0),
                ("2024-02-01", "NSE:ACME", "sell", 30, 100.0, 0.0),
                ("2024-03-01", "NSE:ACME", "sell", 70, 100.0, 0.0),
            ]
        )
    )

    assert closed[0].cost == pytest.approx(30.0)
    assert closed[1].cost == pytest.approx(70.0)
    assert sum(t.cost for t in closed) == pytest.approx(100.0)


def test_instruments_are_matched_independently() -> None:
    """Selling one name must never close another's lot."""
    closed, open_positions = round_trips(
        _log(
            [
                ("2024-01-02", "NSE:AAA", "buy", 10, 100.0, 0.0),
                ("2024-01-03", "NSE:BBB", "buy", 10, 200.0, 0.0),
                ("2024-02-01", "NSE:BBB", "sell", 10, 250.0, 0.0),
            ]
        )
    )

    assert [t.instrument_id for t in closed] == ["NSE:BBB"]
    assert [p.instrument_id for p in open_positions] == ["NSE:AAA"]


def test_an_empty_log_produces_nothing_rather_than_raising() -> None:
    closed, open_positions = round_trips(pd.DataFrame())
    assert closed == []
    assert open_positions == []


def test_a_log_missing_a_column_is_refused() -> None:
    """Silently treating an absent cost column as zero would report a
    strategy as more profitable than it was."""
    frame = _log([("2024-01-02", "NSE:ACME", "buy", 10, 100.0, 0.0)]).drop(columns=["cost"])
    with pytest.raises(ValueError, match="missing column"):
        round_trips(frame)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _trip(net: float, instrument: str = "NSE:ACME") -> RoundTrip:
    return RoundTrip(
        instrument_id=instrument,
        entry_date=dt.date(2024, 1, 1),
        exit_date=dt.date(2024, 1, 11),
        quantity=10,
        entry_price=100.0,
        exit_price=100.0 + net / 10,
        gross_pnl=net,
        cost=0.0,
        net_pnl=net,
    )


def test_summary_counts_accuracy_and_profit_consistently() -> None:
    summary = summarize([_trip(300.0), _trip(-100.0), _trip(200.0), _trip(-50.0)])

    assert summary.round_trips == 4
    assert summary.wins == 2
    assert summary.win_rate == pytest.approx(0.5)
    assert summary.net_pnl == pytest.approx(350.0)
    assert summary.gross_profit == pytest.approx(500.0)
    assert summary.gross_loss == pytest.approx(150.0)
    assert summary.profit_factor == pytest.approx(500.0 / 150.0)
    assert summary.avg_win == pytest.approx(250.0)
    assert summary.avg_loss == pytest.approx(-75.0)


def test_profit_factor_is_infinite_rather_than_a_sentinel_when_nothing_lost() -> None:
    """A zero or a -1 here would sort a flawless strategy below a mediocre
    one in any ranking that reads the number."""
    assert summarize([_trip(10.0)]).profit_factor == float("inf")


def test_summarizing_nothing_is_all_zeros_not_a_crash() -> None:
    summary = summarize([])
    assert summary.round_trips == 0
    assert summary.win_rate == 0.0


def test_per_instrument_ranks_by_net_contribution() -> None:
    frame = by_instrument(
        [_trip(500.0, "NSE:WIN"), _trip(-200.0, "NSE:LOSE"), _trip(100.0, "NSE:WIN")]
    )

    assert list(frame["instrument_id"]) == ["NSE:WIN", "NSE:LOSE"]
    assert frame.loc[0, "round_trips"] == 2
    assert frame.loc[0, "net_pnl"] == pytest.approx(600.0)
    assert frame.loc[1, "win_rate"] == pytest.approx(0.0)
