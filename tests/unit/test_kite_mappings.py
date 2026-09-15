"""Unit tests for the Kite <-> generic vocabulary translation
(``broker/zerodha/kite_mappings.py``, Phase 15). Every status string and
enum value asserted here is quoted verbatim from Zerodha's published
Kite Connect v3 documentation -- see that module's docstring.
"""

from __future__ import annotations

import pytest

from backtest.costs import TradeSide
from broker.zerodha.kite_mappings import (
    join_instrument_id,
    order_state_from_kite,
    side_from_kite,
    side_to_kite,
    split_instrument_id,
)
from execution.order_manager import OrderState


def test_side_round_trips_through_kite_vocabulary() -> None:
    assert side_to_kite(TradeSide.BUY) == "BUY"
    assert side_to_kite(TradeSide.SELL) == "SELL"
    assert side_from_kite("BUY") is TradeSide.BUY
    assert side_from_kite("SELL") is TradeSide.SELL


def test_side_from_kite_rejects_an_unrecognized_value() -> None:
    with pytest.raises(ValueError, match="unrecognized Kite transaction_type"):
        side_from_kite("HOLD")


def test_instrument_id_split_and_join_round_trip() -> None:
    assert split_instrument_id("NSE:INFY") == ("NSE", "INFY")
    assert join_instrument_id("NSE", "INFY") == "NSE:INFY"


def test_instrument_id_split_rejects_a_malformed_value() -> None:
    with pytest.raises(ValueError, match="EXCHANGE:TRADINGSYMBOL"):
        split_instrument_id("INFY")


@pytest.mark.parametrize(
    ("kite_status", "filled", "quantity", "expected"),
    [
        ("COMPLETE", 100, 100, OrderState.FILLED),
        ("REJECTED", 0, 100, OrderState.REJECTED),
        ("CANCELLED", 40, 100, OrderState.CANCELLED),
        ("CANCEL PENDING", 0, 100, OrderState.CANCEL_REQUESTED),
        ("PUT ORDER REQ RECEIVED", 0, 100, OrderState.SUBMITTED),
        ("VALIDATION PENDING", 0, 100, OrderState.SUBMITTED),
        ("OPEN PENDING", 0, 100, OrderState.SUBMITTED),
        ("AMO REQ RECEIVED", 0, 100, OrderState.SUBMITTED),
        ("OPEN", 0, 100, OrderState.OPEN),
        ("MODIFIED", 0, 100, OrderState.OPEN),
        ("TRIGGER PENDING", 0, 100, OrderState.OPEN),
        ("OPEN", 40, 100, OrderState.PARTIALLY_FILLED),
        ("MODIFY PENDING", 40, 100, OrderState.PARTIALLY_FILLED),
        ("SOME-FUTURE-STATUS-NOT-YET-DOCUMENTED", 0, 100, OrderState.UNKNOWN),
    ],
)
def test_order_state_from_kite_maps_every_documented_status(
    kite_status: str, filled: int, quantity: int, expected: OrderState
) -> None:
    assert order_state_from_kite(kite_status, filled, quantity) is expected


def test_order_state_from_kite_treats_a_fully_filled_open_status_as_open_not_partial() -> None:
    """filled_quantity == quantity with status OPEN would be unusual in
    practice (Kite reports COMPLETE once fully filled) but must not be
    misreported as PARTIALLY_FILLED if it ever occurs."""
    assert order_state_from_kite("OPEN", 100, 100) is OrderState.OPEN
