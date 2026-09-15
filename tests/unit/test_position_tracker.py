"""Unit tests for ``execution/position_tracker.py`` (Phase 14): weighted
average cost, realized/unrealized P&L, and the no-shorting invariant.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backtest.costs import TradeSide
from execution.position_tracker import PositionTracker, PositionTrackerError

_T0 = dt.datetime(2024, 6, 3, 9, 30, tzinfo=dt.UTC)
_T1 = dt.datetime(2024, 6, 3, 10, 0, tzinfo=dt.UTC)
_T2 = dt.datetime(2024, 6, 3, 11, 0, tzinfo=dt.UTC)


def test_a_single_buy_opens_a_position_at_the_fill_price() -> None:
    tracker = PositionTracker()
    position = tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    assert position.quantity == 10
    assert position.avg_price == 1500.0
    assert position.current_price == 1500.0
    assert position.unrealized_pnl == 0.0
    assert position.realized_pnl == 0.0


def test_two_buys_produce_a_quantity_weighted_average_price() -> None:
    tracker = PositionTracker()
    tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    position = tracker.apply_fill("NSE:INFY", 10, 1600.0, TradeSide.BUY, _T1)
    assert position.quantity == 20
    assert position.avg_price == pytest.approx(1550.0)


def test_a_partial_sell_realizes_pnl_on_the_sold_portion_only() -> None:
    tracker = PositionTracker()
    tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    position = tracker.apply_fill("NSE:INFY", 4, 1600.0, TradeSide.SELL, _T1)
    assert position.quantity == 6
    assert position.realized_pnl == pytest.approx((1600.0 - 1500.0) * 4)
    # avg cost basis of the remaining shares is unchanged by a sell
    assert position.avg_price == pytest.approx(1500.0)


def test_realized_pnl_survives_a_full_close_and_reopen() -> None:
    tracker = PositionTracker()
    tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    tracker.apply_fill("NSE:INFY", 10, 1600.0, TradeSide.SELL, _T1)
    assert tracker.held_quantity("NSE:INFY") == 0
    reopened = tracker.apply_fill("NSE:INFY", 5, 1700.0, TradeSide.BUY, _T2)
    assert reopened.quantity == 5
    assert reopened.avg_price == pytest.approx(1700.0)
    assert reopened.realized_pnl == pytest.approx((1600.0 - 1500.0) * 10)


def test_selling_more_than_held_is_rejected_as_no_shorting() -> None:
    tracker = PositionTracker()
    tracker.apply_fill("NSE:INFY", 5, 1500.0, TradeSide.BUY, _T0)
    with pytest.raises(PositionTrackerError, match="long-only"):
        tracker.apply_fill("NSE:INFY", 6, 1500.0, TradeSide.SELL, _T1)


def test_selling_with_no_position_at_all_is_rejected() -> None:
    tracker = PositionTracker()
    with pytest.raises(PositionTrackerError, match="long-only"):
        tracker.apply_fill("NSE:INFY", 1, 1500.0, TradeSide.SELL, _T0)


@pytest.mark.parametrize("bad_quantity", [0, -1])
def test_non_positive_fill_quantity_is_rejected(bad_quantity: int) -> None:
    tracker = PositionTracker()
    with pytest.raises(PositionTrackerError, match="fill_quantity"):
        tracker.apply_fill("NSE:INFY", bad_quantity, 1500.0, TradeSide.BUY, _T0)


@pytest.mark.parametrize("bad_price", [0.0, -1.0])
def test_non_positive_fill_price_is_rejected(bad_price: float) -> None:
    tracker = PositionTracker()
    with pytest.raises(PositionTrackerError, match="fill_price"):
        tracker.apply_fill("NSE:INFY", 1, bad_price, TradeSide.BUY, _T0)


def test_mark_to_market_updates_unrealized_pnl_without_a_fill() -> None:
    tracker = PositionTracker()
    tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    tracker.mark_to_market({"NSE:INFY": 1550.0}, _T1)
    position = tracker.current_positions()[0]
    assert position.current_price == 1550.0
    assert position.unrealized_pnl == pytest.approx((1550.0 - 1500.0) * 10)


def test_mark_to_market_ignores_instruments_with_no_open_position() -> None:
    tracker = PositionTracker()
    tracker.mark_to_market({"NSE:INFY": 1550.0}, _T0)
    assert tracker.current_positions() == []


def test_mark_to_market_rejects_a_non_positive_price() -> None:
    tracker = PositionTracker()
    tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    with pytest.raises(PositionTrackerError):
        tracker.mark_to_market({"NSE:INFY": 0.0}, _T1)


def test_current_positions_excludes_fully_closed_instruments() -> None:
    tracker = PositionTracker()
    tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    tracker.apply_fill("NSE:INFY", 10, 1600.0, TradeSide.SELL, _T1)
    assert tracker.current_positions() == []


def test_current_positions_is_sorted_by_instrument_id() -> None:
    tracker = PositionTracker()
    tracker.apply_fill("NSE:TCS", 5, 3500.0, TradeSide.BUY, _T0)
    tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    ids = [position.instrument_id for position in tracker.current_positions()]
    assert ids == sorted(ids)


def test_target_weight_defaults_to_zero_and_is_settable() -> None:
    tracker = PositionTracker()
    tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    assert tracker.current_positions()[0].target_weight == 0.0
    tracker.set_target_weight("NSE:INFY", 0.08)
    assert tracker.current_positions()[0].target_weight == pytest.approx(0.08)


def test_total_realized_and_unrealized_pnl_aggregate_across_instruments() -> None:
    tracker = PositionTracker()
    tracker.apply_fill("NSE:INFY", 10, 1500.0, TradeSide.BUY, _T0)
    tracker.apply_fill("NSE:INFY", 5, 1600.0, TradeSide.SELL, _T1)
    tracker.apply_fill("NSE:TCS", 4, 3500.0, TradeSide.BUY, _T0)
    tracker.mark_to_market({"NSE:INFY": 1550.0, "NSE:TCS": 3600.0}, _T2)

    expected_unrealized = (1550.0 - 1500.0) * 5 + (3600.0 - 3500.0) * 4
    expected_realized = (1600.0 - 1500.0) * 5
    assert tracker.total_unrealized_pnl() == pytest.approx(expected_unrealized)
    assert tracker.total_realized_pnl() == pytest.approx(expected_realized)


def test_held_quantity_is_zero_for_an_unknown_instrument() -> None:
    tracker = PositionTracker()
    assert tracker.held_quantity("NSE:UNKNOWN") == 0


def test_position_rejects_a_negative_quantity_directly() -> None:
    from execution.position_tracker import Position

    with pytest.raises(ValueError, match="long-only"):
        Position(
            instrument_id="NSE:INFY",
            quantity=-1,
            avg_price=100.0,
            current_price=100.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            target_weight=0.0,
            as_of=_T0,
        )
