"""Turns ranked stock candidates + a regime exposure target into proposed
target weights (docs/SPECIFICATION.md section 7.2).

This module proposes weights only. It does not compute final order
quantities -- that reconciliation (this weight-based proposal against the
stop-distance risk-based sizing formula in section 8.1) happens in
risk/position_sizer.py, which always has the final say. See
docs/ARCHITECTURE.md, "Resolved specification ambiguities" (B2/B3).

Not implemented yet (Phase 7).
"""

from __future__ import annotations

from dataclasses import dataclass

from config.models import PortfolioConfig
from core.regime.regime_policy import ExposureTarget
from universe.stock_selector import CandidateScore


@dataclass(frozen=True)
class ProposedWeight:
    instrument_id: str
    target_weight: float
    reason: str


class PortfolioConstructor:
    """Distributes a regime-determined gross-exposure budget across ranked
    candidates, volatility-adjusted, subject to single-name and sector caps.
    """

    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config

    def raw_weights(
        self, candidates: list[CandidateScore], volatilities: dict[str, float]
    ) -> dict[str, float]:
        """``selection_score / volatility`` for each candidate, before caps
        or exposure scaling.
        """
        raise NotImplementedError("Phase 7: portfolio construction is not implemented yet.")

    def apply_caps(
        self, raw_weights: dict[str, float], sector_map: dict[str, str]
    ) -> dict[str, float]:
        """Apply ``portfolio.max_single_name_pct`` and
        ``portfolio.max_sector_pct``.
        """
        raise NotImplementedError("Phase 7: portfolio construction is not implemented yet.")

    def scale_to_exposure_target(
        self, capped_weights: dict[str, float], exposure_target: ExposureTarget
    ) -> list[ProposedWeight]:
        """Scale capped weights so their sum falls within
        ``exposure_target``'s band.
        """
        raise NotImplementedError("Phase 7: portfolio construction is not implemented yet.")

    def construct(
        self,
        candidates: list[CandidateScore],
        volatilities: dict[str, float],
        sector_map: dict[str, str],
        exposure_target: ExposureTarget,
    ) -> list[ProposedWeight]:
        """Full pipeline: raw weights -> caps -> exposure scaling."""
        raise NotImplementedError("Phase 7: portfolio construction is not implemented yet.")
