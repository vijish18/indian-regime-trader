"""The single canonical position-sizing function.

docs/SPECIFICATION.md gives two sizing formulas that are not reconciled in
the source document: a weight-based formula in portfolio construction
(section 7.2, ``selection_score / volatility``) and a stop-distance
risk-based formula here (section 8.1,
``floor((equity * max_risk_per_position) / |entry - stop|)``). This module is
where they are reconciled into one final order quantity, so no other module
needs to guess which one "wins". See docs/ARCHITECTURE.md, "Resolved
specification ambiguities" (B2/B3).

The ``stop_distance`` used here is a *sizing* input (an ATR/volatility-based
risk distance), not necessarily a resting protective stop order -- section
1.2 explicitly demotes live stop orders to a last-resort control. Sizing
still needs a risk-distance estimate even when no stop order will be placed.

Not implemented yet (Phase 7/8).
"""

from __future__ import annotations

from dataclasses import dataclass

from config.models import RiskConfig
from portfolio.portfolio_constructor import TargetPosition


@dataclass(frozen=True)
class SizedOrder:
    instrument_id: str
    quantity: int
    target_weight: float
    stop_distance: float
    binding_constraint: str  # which cap/formula ultimately determined quantity


class PositionSizer:
    """Converts a proposed target weight into a final, risk-bounded order
    quantity.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def weight_based_quantity(self, proposed: TargetPosition, equity: float, price: float) -> int:
        """Quantity implied purely by the proposed target weight."""
        raise NotImplementedError("Phase 7: position sizing is not implemented yet.")

    def risk_based_quantity(self, equity: float, entry_price: float, stop_distance: float) -> int:
        """``floor((equity * max_risk_per_position_pct) / stop_distance)``
        (docs/SPECIFICATION.md section 8.1), stressed for overnight gap risk
        rather than assuming a stop fills exactly at the stop price.
        """
        raise NotImplementedError("Phase 7: position sizing is not implemented yet.")

    def reconcile(
        self,
        proposed: TargetPosition,
        equity: float,
        price: float,
        stop_distance: float,
        available_cash: float,
        liquidity_participation_cap: int,
    ) -> SizedOrder:
        """Take the minimum of the weight-based and risk-based quantities,
        then cap further by single-name weight, sector exposure, available
        cash, and liquidity participation. Records which constraint bound
        the final quantity, for auditability.
        """
        raise NotImplementedError("Phase 7: position sizing is not implemented yet.")
