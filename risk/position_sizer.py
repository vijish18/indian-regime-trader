"""The single canonical position-sizing function.

docs/SPECIFICATION.md gives two sizing formulas that are not reconciled in
the source document: a weight-based formula in portfolio construction
(section 7.2, ``selection_score / volatility``) and a stop-distance
risk-based formula here (section 8.1,
``floor((equity * max_risk_per_position) / |entry - stop|)``). This module is
where they are reconciled into one final order quantity, so no other module
needs to guess which one "wins". See docs/ARCHITECTURE.md, "Resolved
specification ambiguities".

**The reconciliation rule is the minimum, and the direction matters.** The
two formulas answer different questions -- "what share of the portfolio
should this name be?" and "how much can I lose if this name moves against
me?" -- and a quantity that satisfies one can violate the other. Taking
the minimum means both are always respected; taking anything else means
one is sometimes not. Every subsequent cap (single-name, cash, liquidity)
narrows further, never widens, so the final quantity satisfies every
constraint simultaneously and :attr:`SizedOrder.binding_constraint`
records which one actually bit.

The ``stop_distance`` used here is a *sizing* input (an ATR/volatility-based
risk distance), not necessarily a resting protective stop order -- section
1.2 explicitly demotes live stop orders to a last-resort control. Sizing
still needs a risk-distance estimate even when no stop order will be placed.
That is also why the risk-based formula stresses the distance for overnight
gap risk: a stop is an intention, not a guarantee, and a cash-equity
position held overnight can reopen far through it.

This module does not decide *whether* to take a position. ``risk_manager``
(Phase 7b) approves or vetoes a proposal before it ever reaches here, and a
quantity of zero from this module is "the constraints leave no room", not
"rejected".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config.models import RiskConfig
from portfolio.portfolio_constructor import TargetPosition

OVERNIGHT_GAP_STRESS_MULTIPLE = 1.5
"""How far beyond the nominal stop distance the risk formula assumes price
can travel before the position is actually exited.

A stop is an intention, not a fill. An Indian cash-equity position held
overnight can reopen through its stop on a gap -- and this system trades
daily bars, so *every* position is held overnight. Sizing to the nominal
distance would systematically understate loss per position by exactly the
gap that shows up on the worst days.

1.5x is a deliberately round, conservative number rather than a fitted
one: it is a safety margin, and a margin tuned to historical gaps would
be calibrated to the gaps that have already happened. It errs toward
smaller positions, which is the correct direction for a cap whose purpose
is bounding loss.
"""


class PositionSizingError(ValueError):
    """An input that cannot produce a meaningful quantity.

    Raised rather than returning zero, because these represent a broken
    caller -- a negative price, a zero stop distance -- not a legitimately
    binding constraint. Returning zero would make a programming error look
    like a risk decision.
    """


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
        """Quantity implied purely by the proposed target weight.

        ``floor(equity * target_weight / price)``. Floor, not round: rounding
        up can push the position past the very weight cap that produced the
        target, and a half share cannot be bought anyway.
        """
        _require_positive("equity", equity)
        _require_positive("price", price)
        if not 0.0 <= proposed.target_weight <= 1.0:
            raise PositionSizingError(
                f"target_weight must be in [0, 1], got {proposed.target_weight}"
            )
        return math.floor(equity * proposed.target_weight / price)

    def risk_based_quantity(self, equity: float, entry_price: float, stop_distance: float) -> int:
        """``floor((equity * max_risk_per_position_pct) / stop_distance)``
        (docs/SPECIFICATION.md section 8.1), stressed for overnight gap risk
        rather than assuming a stop fills exactly at the stop price.

        The stress divides by a *wider* distance than quoted, so the
        resulting quantity is smaller. See
        :data:`OVERNIGHT_GAP_STRESS_MULTIPLE`.
        """
        _require_positive("equity", equity)
        _require_positive("entry_price", entry_price)
        if stop_distance <= 0:
            # Zero would divide; negative is meaningless. Either means the
            # caller's risk-distance estimate is broken, and sizing off a
            # broken estimate is worse than refusing.
            raise PositionSizingError(
                f"stop_distance must be positive, got {stop_distance}. It is a risk "
                "distance (e.g. an ATR multiple), not an optional hint."
            )
        budget = equity * self.config.max_risk_per_position_pct
        stressed_distance = stop_distance * OVERNIGHT_GAP_STRESS_MULTIPLE
        return math.floor(budget / stressed_distance)

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
        then cap further by single-name weight, available cash, and liquidity
        participation. Records which constraint bound the final quantity.

        Sector exposure is deliberately *not* applied here: it is a property
        of the portfolio, not of one order, so it cannot be evaluated from a
        single ``TargetPosition``. ``portfolio/portfolio_constructor.py``
        enforces it while building target weights, and
        ``risk/risk_manager.py`` re-checks it across the whole proposal.
        Pretending to enforce it here would give a false assurance, since
        this method cannot see the other positions.
        """
        _require_positive("equity", equity)
        _require_positive("price", price)
        if available_cash < 0:
            raise PositionSizingError(f"available_cash cannot be negative, got {available_cash}")
        if liquidity_participation_cap < 0:
            raise PositionSizingError(
                f"liquidity_participation_cap cannot be negative, "
                f"got {liquidity_participation_cap}"
            )

        # Every candidate is an upper bound on the final quantity, so the
        # answer is the smallest. Ordered so that ties resolve to the
        # earliest-listed, most fundamental constraint -- a tie between
        # "risk budget" and "ran out of cash" is more usefully reported as
        # the risk budget.
        candidates: list[tuple[str, int]] = [
            ("target_weight", self.weight_based_quantity(proposed, equity, price)),
            ("risk_per_position", self.risk_based_quantity(equity, price, stop_distance)),
            (
                "max_single_name_pct",
                math.floor(equity * self.config.max_single_name_pct / price),
            ),
            ("available_cash", math.floor(available_cash / price)),
            ("liquidity_participation", liquidity_participation_cap),
        ]

        binding, quantity = min(candidates, key=lambda item: item[1])
        # A negative cap would be a caller bug; a zero one is legitimate
        # (no cash, no liquidity) and means "do not trade this name now".
        quantity = max(0, quantity)

        return SizedOrder(
            instrument_id=proposed.instrument_id,
            quantity=quantity,
            target_weight=proposed.target_weight,
            stop_distance=stop_distance,
            binding_constraint=binding,
        )


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise PositionSizingError(f"{name} must be positive, got {value}")
