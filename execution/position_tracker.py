"""Tracks live positions from fills, and handles corporate-action identity
changes and forced exits (delisting/suspension) for held positions. See
docs/SPECIFICATION.md section 17 (``positions`` table) and
docs/ARCHITECTURE.md.

Not implemented yet (Phase 10/11).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    instrument_id: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    target_weight: float
    as_of: dt.datetime


class PositionTracker:
    """Source of truth for live positions, reconciled against the broker at
    startup and on a recurring schedule (docs/SPECIFICATION.md section 15).
    """

    def apply_fill(
        self, instrument_id: str, fill_quantity: int, fill_price: float, side: str
    ) -> Position:
        raise NotImplementedError("Phase 10/11: position tracking is not implemented yet.")

    def apply_corporate_action(
        self, instrument_id: str, successor_instrument_id: str | None, ratio: float
    ) -> Position:
        """Handle a corporate action affecting a held position: identity
        change (merger/demerger) or quantity adjustment (split/bonus).
        """
        raise NotImplementedError("Phase 10/11: position tracking is not implemented yet.")

    def force_exit(self, instrument_id: str, reason: str) -> None:
        """Mark a position for forced exit (e.g. delisting, trading
        suspension) as a first-class event, not a silent removal.
        """
        raise NotImplementedError("Phase 10/11: position tracking is not implemented yet.")

    def current_positions(self) -> list[Position]:
        raise NotImplementedError("Phase 10/11: position tracking is not implemented yet.")
