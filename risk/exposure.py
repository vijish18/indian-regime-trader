"""Tracks live gross/net exposure, single-name, and sector concentration
against the configured risk limits. See docs/SPECIFICATION.md section 8.

Not implemented yet (Phase 7/8).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExposureSnapshot:
    gross_exposure_pct: float
    single_name_pct: dict[str, float]
    sector_pct: dict[str, float]
    position_count: int


class ExposureTracker:
    """Computes current exposure from live positions, for the risk manager
    to evaluate proposed trades against.
    """

    def current_exposure(self) -> ExposureSnapshot:
        raise NotImplementedError("Phase 7/8: exposure tracking is not implemented yet.")

    def exposure_after(self, proposed_trades: dict[str, float]) -> ExposureSnapshot:
        """Projected exposure if ``proposed_trades`` (instrument_id ->
        target weight) were applied, without mutating live state.
        """
        raise NotImplementedError("Phase 7/8: exposure tracking is not implemented yet.")
