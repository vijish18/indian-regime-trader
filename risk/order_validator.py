"""Pre-trade order validation: the last gate before an order reaches
broker/execution. See docs/SPECIFICATION.md section 8 and section 12.2.

Not implemented yet (Phase 8/10).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    approved: bool
    rejection_reason: str | None


class OrderValidator:
    """Rejects orders for instrument tradability, stale quotes, excessive
    spread, inconsistent instrument metadata, tick size, freeze/quantity
    restrictions, and price bands (docs/SPECIFICATION.md section 12.2).
    """

    def validate(
        self,
        instrument_id: str,
        quantity: int,
        limit_price: float,
        is_tradable: bool,
        quote_is_stale: bool,
        spread_bps: float,
        max_spread_bps: float,
    ) -> ValidationResult:
        raise NotImplementedError("Phase 8/10: order validation is not implemented yet.")
