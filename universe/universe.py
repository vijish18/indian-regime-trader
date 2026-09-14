"""Point-in-time tradable universe construction.

Never backtest today's NIFTY 50 membership into the past -- every snapshot
must reflect what was actually a member, liquid, and tradable on that date
(docs/SPECIFICATION.md section 2.1). This is the primary survivorship-bias
control in the system.

Not implemented yet (Phase 3, alongside data ingestion).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class UniverseSnapshot:
    as_of: dt.date
    instrument_ids: tuple[str, ...]
    exclusion_reasons: dict[str, str]  # instrument_id -> reason, for excluded names


class UniverseBuilder:
    """Builds and persists the point-in-time tradable universe."""

    def build_snapshot(self, as_of: dt.date) -> UniverseSnapshot:
        """Construct the eligible universe as of ``as_of``: NIFTY 50
        point-in-time members meeting the configured liquidity threshold,
        excluding persistently illiquid, suspended, or data-quality-flagged
        instruments (config: ``universe.min_avg_daily_value_inr``,
        ``universe.exclude_illiquid``).
        """
        raise NotImplementedError("Phase 3: universe construction is not implemented yet.")

    def get_snapshot(self, as_of: dt.date) -> UniverseSnapshot:
        """Return the exact, previously-stored snapshot used on ``as_of``,
        rather than recomputing it -- every rebalance must be reproducible
        against the snapshot actually used at the time.
        """
        raise NotImplementedError("Phase 3: universe construction is not implemented yet.")
