"""Converts weight-level ``portfolio.portfolio_constructor.RequiredTrade``
objects into whole-share order quantities for the orchestration layer's own
use (Phase 19, step 13: "calculate required trades").

This is deliberately the same "simple weight-based floor(notional/price)"
sizing ``backtest/engine.py`` already uses for its own fills (see that
module's docstring, and ``docs/ARCHITECTURE.md``'s dependency rules) --
**not** ``risk/position_sizer.py``'s future job. That module will
reconcile this weight-based formula against a stop-distance risk-based
formula and a liquidity-participation cap into one final canonical
quantity (Phase 7c, still unimplemented); this module only answers "how
many whole shares moves the currently-held quantity to the
already-risk-approved target weight," using the single-name/sector/
liquidity caps ``PortfolioConstructor`` already applied when it built the
target weight in the first place. No new strategy math is introduced here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from portfolio.portfolio_constructor import RequiredTrade, TradeAction
from risk.risk_manager import RiskDecision


@dataclass(frozen=True, slots=True)
class SizedTrade:
    instrument_id: str
    side: str
    """``"buy"`` or ``"sell"`` -- ``execution.order_manager.OrderManager.create``'s
    own ``side`` vocabulary."""

    quantity: int
    action: TradeAction
    price: float


def size_trades(
    trades: list[RequiredTrade],
    decisions_by_instrument: dict[str, RiskDecision],
    prices: dict[str, float],
    current_quantities: dict[str, int],
    equity: float,
    *,
    circuit_halted: bool,
    min_order_value_inr: float = 0.0,
) -> tuple[list[SizedTrade], list[str]]:
    """Returns ``(sized_trades, skip_reasons)``.

    A trade is skipped, with a human-readable reason recorded, when:

    - ``circuit_halted`` is ``True`` -- nothing trades, not even an exit.
      Mirrors ``execution.startup.StartupSequence``'s own "never auto-trade
      on disagreement" precedent, applied here to "never trade while
      halted".
    - it is a ``HOLD`` (no weight change).
    - it is a ``BUY``/``SELL`` (still held, target_weight > 0) without a
      matching *approved* ``RiskDecision`` -- ``RiskManager.evaluate`` only
      produces a decision per ``proposed.positions``, so an ``EXIT``
      (target_weight == 0, absent from ``proposed``) has no decision to
      check and is allowed through by weight-reduction alone; a
      BUY/SELL's absence of a decision is treated as unapproved, not as
      "not applicable".
    - the instrument's current price is unknown.
    - the resulting whole-share quantity is zero (already at target, or
      the weight delta rounds below one share).
    """
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")

    sized: list[SizedTrade] = []
    skipped: list[str] = []

    if circuit_halted:
        return sized, [
            f"{trade.instrument_id}: circuit breaker is halted; no orders submitted"
            for trade in trades
            if trade.action is not TradeAction.HOLD
        ]

    for trade in trades:
        instrument_id = trade.instrument_id
        if trade.action is TradeAction.HOLD:
            continue

        if trade.action is not TradeAction.EXIT:
            decision = decisions_by_instrument.get(instrument_id)
            if decision is None or not decision.approved:
                reason = (
                    "no risk decision found"
                    if decision is None
                    else "; ".join(v.message for v in decision.violations)
                )
                skipped.append(f"{instrument_id}: not risk-approved ({reason})")
                continue

        price = prices.get(instrument_id)
        if price is None or price <= 0:
            skipped.append(f"{instrument_id}: no current price available")
            continue

        current_quantity = current_quantities.get(instrument_id, 0)
        if trade.action is TradeAction.EXIT:
            target_quantity = 0
        else:
            target_quantity = math.floor((trade.target_weight * equity) / price)

        delta = target_quantity - current_quantity
        if delta == 0:
            skipped.append(f"{instrument_id}: target already met at current share count")
            continue

        quantity = abs(delta)
        if quantity * price < min_order_value_inr:
            skipped.append(
                f"{instrument_id}: order value below configured minimum "
                f"({quantity * price:.2f} < {min_order_value_inr:.2f})"
            )
            continue

        sized.append(
            SizedTrade(
                instrument_id=instrument_id,
                side="buy" if delta > 0 else "sell",
                quantity=quantity,
                action=trade.action,
                price=price,
            )
        )

    return sized, skipped
