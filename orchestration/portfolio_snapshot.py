"""Converts ``execution.position_tracker.PositionTracker``'s current
holdings (share quantities and mark prices) into the weight-based
``portfolio.portfolio_constructor.TargetPortfolio`` shape that
``PortfolioConstructor.construct`` and ``portfolio_constructor.required_trades``
expect as "what is held right now". Purely descriptive: rank/score/
binding_constraint only matter for a *newly proposed* position, so a
currently-held position gets placeholder values for them rather than a
guess.
"""

from __future__ import annotations

import datetime as dt

from core.regime.allocation import AllocationRegime
from execution.position_tracker import Position
from portfolio.portfolio_constructor import (
    TargetPortfolio,
    TargetPosition,
    empty_portfolio,
)

CURRENT_HOLDING_BINDING_CONSTRAINT = "current_holding"
"""Distinct from ``PortfolioConstructor``'s five documented binding-constraint
values -- marks a position as descriptive of current state, not the output
of a weighting/capping step."""


def current_target_portfolio(
    positions: list[Position],
    equity: float,
    as_of: dt.date,
    regime: AllocationRegime,
    sector_map: dict[str, str] | None = None,
) -> TargetPortfolio:
    if equity <= 0:
        raise ValueError(f"equity must be positive, got {equity}")
    if not positions:
        return empty_portfolio(as_of, regime)

    sectors = sector_map or {}
    held = tuple(
        TargetPosition(
            instrument_id=position.instrument_id,
            symbol=position.instrument_id,
            target_weight=(position.quantity * position.current_price) / equity,
            sector=sectors.get(position.instrument_id, position.instrument_id),
            rank=0,
            score=0.0,
            binding_constraint=CURRENT_HOLDING_BINDING_CONSTRAINT,
        )
        for position in positions
        if position.current_price > 0
    )
    gross_exposure = sum(position.target_weight for position in held)
    return TargetPortfolio(
        as_of=as_of,
        positions=held,
        # Not clamped to >= 0: equity is total account equity (cash + holdings
        # value), so gross_exposure > 1.0 here would mean equity itself was
        # computed wrong -- TargetPortfolio's own validation should catch
        # that rather than this function silently hiding it.
        cash_weight=1.0 - gross_exposure,
        regime=regime,
        gross_exposure=gross_exposure,
    )
