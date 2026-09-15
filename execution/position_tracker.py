"""Tracks live positions from fills, and handles corporate-action identity
changes and forced exits (delisting/suspension) for held positions. See
docs/SPECIFICATION.md section 17 (``positions`` table) and
docs/ARCHITECTURE.md.

This is the one "portfolio state" model shape ``broker.adapters.paper_broker.PaperBroker``
(Phase 14) and a future live adapter both produce, so downstream code
(risk, reporting) never needs to know which one is running.

Corporate-action identity changes and forced exits remain Phase 11b scope
(they need ``data.interfaces.CorporateActionProvider`` and reconciliation
wiring this phase does not touch) -- still stubbed below.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from backtest.costs import TradeSide


class PositionTrackerError(RuntimeError):
    """A fill could not be applied -- e.g. a sell larger than the held
    quantity, which would imply shorting (forbidden, V1 long-only)."""


@dataclass(frozen=True, slots=True)
class Position:
    instrument_id: str
    quantity: int
    avg_price: float
    """Weighted-average cost basis of the currently held quantity --
    unaffected by a sell that only reduces quantity (the standard
    average-cost-basis convention), only recomputed on a buy."""

    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    """Cumulative realized P&L from every sell against this instrument so
    far, including sells that have since fully closed and reopened the
    position -- never reset just because quantity returned to zero."""

    target_weight: float
    as_of: dt.datetime

    def __post_init__(self) -> None:
        if self.quantity < 0:
            raise ValueError(f"quantity must be >= 0 (long-only), got {self.quantity}")


class PositionTracker:
    """Source of truth for live positions, reconciled against the broker at
    startup and on a recurring schedule (docs/SPECIFICATION.md section 15;
    the reconciliation loop itself is Phase 11b, not this module).
    """

    def __init__(self) -> None:
        self._quantity: dict[str, int] = {}
        self._avg_price: dict[str, float] = {}
        self._realized_pnl: dict[str, float] = {}
        self._target_weight: dict[str, float] = {}
        self._last_price: dict[str, float] = {}
        self._last_update: dict[str, dt.datetime] = {}

    def apply_fill(
        self,
        instrument_id: str,
        fill_quantity: int,
        fill_price: float,
        side: TradeSide,
        as_of: dt.datetime,
    ) -> Position:
        if fill_quantity <= 0:
            raise PositionTrackerError(f"fill_quantity must be > 0, got {fill_quantity}")
        if fill_price <= 0:
            raise PositionTrackerError(f"fill_price must be > 0, got {fill_price}")

        held = self._quantity.get(instrument_id, 0)
        avg_price = self._avg_price.get(instrument_id, 0.0)

        if side is TradeSide.BUY:
            new_quantity = held + fill_quantity
            new_avg_price = (held * avg_price + fill_quantity * fill_price) / new_quantity
            self._quantity[instrument_id] = new_quantity
            self._avg_price[instrument_id] = new_avg_price
        else:
            if fill_quantity > held:
                raise PositionTrackerError(
                    f"sell of {fill_quantity} exceeds held quantity {held} for "
                    f"{instrument_id}; this system is long-only and never shorts"
                )
            realized = (fill_price - avg_price) * fill_quantity
            prior_realized = self._realized_pnl.get(instrument_id, 0.0)
            self._realized_pnl[instrument_id] = prior_realized + realized
            remaining = held - fill_quantity
            self._quantity[instrument_id] = remaining
            if remaining == 0:
                self._avg_price.pop(instrument_id, None)

        self._last_price[instrument_id] = fill_price
        self._last_update[instrument_id] = as_of
        return self._snapshot(instrument_id, as_of)

    def mark_to_market(self, prices: dict[str, float], as_of: dt.datetime) -> None:
        """Update the last-known price used for unrealized P&L, independent
        of any fill -- e.g. from a fresh quote tick.
        """
        for instrument_id, price in prices.items():
            if price <= 0:
                raise PositionTrackerError(f"price for {instrument_id} must be > 0, got {price}")
            if self._quantity.get(instrument_id, 0) <= 0:
                continue
            self._last_price[instrument_id] = price
            self._last_update[instrument_id] = as_of

    def set_target_weight(self, instrument_id: str, target_weight: float) -> None:
        self._target_weight[instrument_id] = target_weight

    def _snapshot(self, instrument_id: str, as_of: dt.datetime) -> Position:
        quantity = self._quantity.get(instrument_id, 0)
        avg_price = self._avg_price.get(instrument_id, 0.0)
        current_price = self._last_price.get(instrument_id, avg_price)
        return Position(
            instrument_id=instrument_id,
            quantity=quantity,
            avg_price=avg_price,
            current_price=current_price,
            unrealized_pnl=(current_price - avg_price) * quantity,
            realized_pnl=self._realized_pnl.get(instrument_id, 0.0),
            target_weight=self._target_weight.get(instrument_id, 0.0),
            as_of=self._last_update.get(instrument_id, as_of),
        )

    def held_quantity(self, instrument_id: str) -> int:
        return self._quantity.get(instrument_id, 0)

    def current_positions(self) -> list[Position]:
        """Every instrument with a nonzero held quantity, sorted for
        deterministic output. An instrument fully closed out (quantity
        zero) drops out of this list -- its realized P&L stays on record
        internally, but the position itself no longer exists.
        """
        as_of = dt.datetime.now(dt.UTC)
        return [
            self._snapshot(instrument_id, self._last_update.get(instrument_id, as_of))
            for instrument_id in sorted(self._quantity)
            if self._quantity[instrument_id] > 0
        ]

    def total_unrealized_pnl(self) -> float:
        return sum(position.unrealized_pnl for position in self.current_positions())

    def total_realized_pnl(self) -> float:
        return sum(self._realized_pnl.values())

    def apply_corporate_action(
        self, instrument_id: str, successor_instrument_id: str | None, ratio: float
    ) -> Position:
        """Handle a corporate action affecting a held position: identity
        change (merger/demerger) or quantity adjustment (split/bonus).
        """
        raise NotImplementedError("Phase 11b: corporate-action handling is not implemented yet.")

    def force_exit(self, instrument_id: str, reason: str) -> None:
        """Mark a position for forced exit (e.g. delisting, trading
        suspension) as a first-class event, not a silent removal.
        """
        raise NotImplementedError("Phase 11b: forced exit is not implemented yet.")
