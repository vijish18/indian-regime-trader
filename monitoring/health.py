"""System health checks: broker connectivity, market-data freshness,
instrument-master freshness, model freshness, and process heartbeat. Feeds
the startup fail-closed sequence (docs/SPECIFICATION.md section 15.1/15.2)
and ongoing monitoring (section 16).

Not implemented yet (Phase 11/12).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum


class ComponentHealth(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class HealthReport:
    component: str
    status: ComponentHealth
    detail: str
    checked_at: dt.datetime


class HealthChecker:
    def check_broker(self) -> HealthReport:
        raise NotImplementedError("Phase 11/12: health checks are not implemented yet.")

    def check_market_data_freshness(self) -> HealthReport:
        raise NotImplementedError("Phase 11/12: health checks are not implemented yet.")

    def check_instrument_master_freshness(self) -> HealthReport:
        raise NotImplementedError("Phase 11/12: health checks are not implemented yet.")

    def check_model_freshness(self) -> HealthReport:
        raise NotImplementedError("Phase 11/12: health checks are not implemented yet.")

    def check_heartbeat(self) -> HealthReport:
        raise NotImplementedError("Phase 11/12: health checks are not implemented yet.")

    def run_startup_sequence(self) -> list[HealthReport]:
        """The ordered startup checks in docs/SPECIFICATION.md section 15.1.
        Any failing, fail-closed condition (section 15.2) must prevent new
        signal generation from being enabled.
        """
        raise NotImplementedError("Phase 11/12: startup health sequence is not implemented yet.")
