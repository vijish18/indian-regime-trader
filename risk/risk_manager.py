"""The independent risk-management orchestrator with final veto authority.

Composes exposure.py, circuit_breaker.py, position_sizer.py, and
order_validator.py into one decision per proposed trade. Its output is
always the last word before execution -- portfolio construction's proposed
weights are inputs to this layer, never final targets
(docs/SPECIFICATION.md section 8, "NON-NEGOTIABLE").

Not implemented yet (Phase 7/8).
"""

from __future__ import annotations

from dataclasses import dataclass

from config.models import RiskConfig
from portfolio.portfolio_constructor import ProposedWeight
from risk.circuit_breaker import CircuitBreakerStatus
from risk.exposure import ExposureSnapshot
from risk.position_sizer import SizedOrder


@dataclass(frozen=True)
class RiskDecision:
    instrument_id: str
    approved: bool
    sized_order: SizedOrder | None
    modified_weight: float | None
    rejection_reason: str | None


class RiskManager:
    """Evaluates proposed portfolio weights against every configured risk
    control and returns an approved, sized, or rejected decision for each.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def evaluate(
        self,
        proposed_weights: list[ProposedWeight],
        current_exposure: ExposureSnapshot,
        circuit_status: CircuitBreakerStatus,
    ) -> list[RiskDecision]:
        """Evaluate every proposed weight and return one decision per
        instrument. A HALT circuit state must produce rejections for every
        entry, with no exceptions.
        """
        raise NotImplementedError("Phase 7/8: risk manager is not implemented yet.")
