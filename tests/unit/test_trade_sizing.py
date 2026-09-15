"""Unit tests for ``orchestration/trade_sizing.py`` (Phase 19): converting
weight-level ``RequiredTrade`` objects into whole-share orders, respecting
risk approval, circuit-breaker halts, and price/quantity edge cases.
"""

from __future__ import annotations

import pytest

from orchestration.trade_sizing import size_trades
from portfolio.portfolio_constructor import RequiredTrade, TradeAction
from risk.circuit_breaker import CircuitState
from risk.risk_manager import RiskCheck, RiskDecision, RiskViolation

_EQUITY = 1_000_000.0


def _approved(instrument_id: str, target_weight: float) -> RiskDecision:
    return RiskDecision(
        instrument_id=instrument_id,
        approved=True,
        target_weight=target_weight,
        circuit_state=CircuitState.NORMAL,
        violations=(),
    )


def _rejected(
    instrument_id: str, target_weight: float, message: str = "too concentrated"
) -> RiskDecision:
    return RiskDecision(
        instrument_id=instrument_id,
        approved=False,
        target_weight=target_weight,
        circuit_state=CircuitState.NORMAL,
        violations=(RiskViolation(RiskCheck.SINGLE_NAME_EXPOSURE, message, 0.1, 0.2),),
    )


def _buy(instrument_id: str, target_weight: float, current_weight: float = 0.0) -> RequiredTrade:
    return RequiredTrade(
        instrument_id=instrument_id,
        current_weight=current_weight,
        target_weight=target_weight,
        delta_weight=target_weight - current_weight,
        action=TradeAction.BUY,
    )


def _exit(instrument_id: str, current_weight: float) -> RequiredTrade:
    return RequiredTrade(
        instrument_id=instrument_id,
        current_weight=current_weight,
        target_weight=0.0,
        delta_weight=-current_weight,
        action=TradeAction.EXIT,
    )


def _hold(instrument_id: str, weight: float) -> RequiredTrade:
    return RequiredTrade(
        instrument_id=instrument_id,
        current_weight=weight,
        target_weight=weight,
        delta_weight=0.0,
        action=TradeAction.HOLD,
    )


def test_approved_buy_is_sized_to_whole_shares() -> None:
    trades = [_buy("NSE:INFY", 0.15)]
    decisions = {"NSE:INFY": _approved("NSE:INFY", 0.15)}
    prices = {"NSE:INFY": 1500.0}
    sized, skipped = size_trades(
        trades, decisions, prices, {}, _EQUITY, circuit_halted=False
    )
    assert skipped == []
    assert len(sized) == 1
    trade = sized[0]
    assert trade.side == "buy"
    assert trade.quantity == int((0.15 * _EQUITY) // 1500.0)


def test_unapproved_buy_is_skipped() -> None:
    trades = [_buy("NSE:INFY", 0.15)]
    decisions = {"NSE:INFY": _rejected("NSE:INFY", 0.15)}
    prices = {"NSE:INFY": 1500.0}
    sized, skipped = size_trades(trades, decisions, prices, {}, _EQUITY, circuit_halted=False)
    assert sized == []
    assert len(skipped) == 1
    assert "not risk-approved" in skipped[0]


def test_missing_decision_for_a_buy_is_skipped() -> None:
    trades = [_buy("NSE:INFY", 0.15)]
    prices = {"NSE:INFY": 1500.0}
    sized, skipped = size_trades(trades, {}, prices, {}, _EQUITY, circuit_halted=False)
    assert sized == []
    assert "no risk decision found" in skipped[0]


def test_exit_needs_no_risk_decision() -> None:
    trades = [_exit("NSE:INFY", 0.15)]
    prices = {"NSE:INFY": 1500.0}
    current_quantities = {"NSE:INFY": 100}
    sized, skipped = size_trades(
        trades, {}, prices, current_quantities, _EQUITY, circuit_halted=False
    )
    assert skipped == []
    assert len(sized) == 1
    assert sized[0].side == "sell"
    assert sized[0].quantity == 100


def test_hold_produces_no_trade() -> None:
    trades = [_hold("NSE:INFY", 0.15)]
    sized, skipped = size_trades(trades, {}, {}, {}, _EQUITY, circuit_halted=False)
    assert sized == []
    assert skipped == []


def test_circuit_halted_blocks_everything_including_exits() -> None:
    trades = [_buy("NSE:INFY", 0.15), _exit("NSE:TCS", 0.10)]
    decisions = {"NSE:INFY": _approved("NSE:INFY", 0.15)}
    prices = {"NSE:INFY": 1500.0, "NSE:TCS": 3500.0}
    sized, skipped = size_trades(
        trades, decisions, prices, {"NSE:TCS": 30}, _EQUITY, circuit_halted=True
    )
    assert sized == []
    assert len(skipped) == 2
    assert all("halted" in reason for reason in skipped)


def test_missing_price_is_skipped() -> None:
    trades = [_buy("NSE:INFY", 0.15)]
    decisions = {"NSE:INFY": _approved("NSE:INFY", 0.15)}
    sized, skipped = size_trades(trades, decisions, {}, {}, _EQUITY, circuit_halted=False)
    assert sized == []
    assert "no current price" in skipped[0]


def test_already_at_target_share_count_is_skipped() -> None:
    trades = [_buy("NSE:INFY", 0.15, current_weight=0.15)]
    decisions = {"NSE:INFY": _approved("NSE:INFY", 0.15)}
    prices = {"NSE:INFY": 1500.0}
    target_shares = int((0.15 * _EQUITY) // 1500.0)
    sized, skipped = size_trades(
        trades, decisions, prices, {"NSE:INFY": target_shares}, _EQUITY, circuit_halted=False
    )
    assert sized == []
    assert "already met" in skipped[0]


def test_a_reduction_produces_a_sell() -> None:
    trades = [
        RequiredTrade(
            instrument_id="NSE:INFY",
            current_weight=0.15,
            target_weight=0.05,
            delta_weight=-0.10,
            action=TradeAction.SELL,
        )
    ]
    decisions = {"NSE:INFY": _approved("NSE:INFY", 0.05)}
    prices = {"NSE:INFY": 1500.0}
    current_quantities = {"NSE:INFY": int((0.15 * _EQUITY) // 1500.0)}
    sized, skipped = size_trades(
        trades, decisions, prices, current_quantities, _EQUITY, circuit_halted=False
    )
    assert skipped == []
    assert sized[0].side == "sell"


def test_min_order_value_filters_dust_trades() -> None:
    trades = [_buy("NSE:INFY", 0.002)]  # floors to 1 share (~1500 INR notional)
    decisions = {"NSE:INFY": _approved("NSE:INFY", 0.002)}
    prices = {"NSE:INFY": 1500.0}
    sized, skipped = size_trades(
        trades, decisions, prices, {}, _EQUITY, circuit_halted=False, min_order_value_inr=5000.0
    )
    assert sized == []
    assert "below configured minimum" in skipped[0]


def test_equity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="equity"):
        size_trades([], {}, {}, {}, 0.0, circuit_halted=False)
