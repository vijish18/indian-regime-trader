"""System health checks: broker connectivity, market-data freshness,
instrument-master freshness, model freshness, and process heartbeat. Feeds
the startup fail-closed sequence (docs/SPECIFICATION.md section 15.1/15.2)
and ongoing monitoring (section 16).

Implemented in Phase 19 as the dedicated module ``orchestration.orchestrator.Orchestrator``
calls for its own "monitor health" step (step 19 of the daily workflow) --
this module has no orchestration logic of its own (no state machine, no
daily-workflow sequencing) and must not import from ``orchestration/``;
the dependency runs the other way, matching every other module
``Orchestrator`` wires together.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from broker.base import Broker
from core.regime.model_registry import ModelRegistry, NoApprovedModelError
from data.errors import CalendarCoverageError, DataNotAvailableError
from data.interfaces import MarketDataProvider, TradingCalendar
from universe.universe import UniverseProvider


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
    def __init__(
        self,
        broker: Broker,
        market_data: MarketDataProvider,
        universe_provider: UniverseProvider,
        model_registry: ModelRegistry,
        calendar: TradingCalendar,
        instrument_ids: Sequence[str],
        *,
        max_stale_sessions: int = 1,
        model_max_age_sessions: int = 21,
        heartbeat_interval_seconds: float = 300.0,
        last_heartbeat: Callable[[], dt.datetime | None] | None = None,
        clock: Callable[[], dt.datetime] | None = None,
    ) -> None:
        self.broker = broker
        self.market_data = market_data
        self.universe_provider = universe_provider
        self.model_registry = model_registry
        self.calendar = calendar
        self.instrument_ids = tuple(instrument_ids)
        self.max_stale_sessions = max_stale_sessions
        self.model_max_age_sessions = model_max_age_sessions
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._last_heartbeat = last_heartbeat or (lambda: None)
        self._clock: Callable[[], dt.datetime] = clock or (lambda: dt.datetime.now(dt.UTC))

    def check_broker(self) -> HealthReport:
        now = self._clock()
        status = self.broker.health_check()
        return HealthReport(
            component="broker",
            status=ComponentHealth.HEALTHY if status.healthy else ComponentHealth.UNHEALTHY,
            detail=status.detail,
            checked_at=now,
        )

    def check_market_data_freshness(self, as_of: dt.date) -> HealthReport:
        """HEALTHY if every configured instrument's available history
        reaches within ``max_stale_sessions`` trading sessions of
        ``as_of``; DEGRADED if only some do; UNHEALTHY if none do (in
        particular, if the required minimum coverage is entirely absent).
        """
        now = self._clock()
        if not self.instrument_ids:
            return HealthReport(
                component="market_data",
                status=ComponentHealth.UNHEALTHY,
                detail="no instruments configured to check freshness against",
                checked_at=now,
            )
        try:
            boundary = self.calendar.sessions_offset(as_of, -self.max_stale_sessions)
        except (CalendarCoverageError, ValueError) as exc:
            return HealthReport(
                component="market_data",
                status=ComponentHealth.UNHEALTHY,
                detail=f"could not establish a freshness boundary: {exc}",
                checked_at=now,
            )

        fresh, stale, missing = [], [], []
        for instrument_id in self.instrument_ids:
            available = self.market_data.available_range(instrument_id)
            if available is None:
                missing.append(instrument_id)
            elif available[1] >= boundary:
                fresh.append(instrument_id)
            else:
                stale.append(instrument_id)

        if not fresh:
            status = ComponentHealth.UNHEALTHY
        elif stale or missing:
            status = ComponentHealth.DEGRADED
        else:
            status = ComponentHealth.HEALTHY
        detail = f"{len(fresh)}/{len(self.instrument_ids)} instruments fresh through {boundary}"
        if stale:
            detail += f"; stale: {stale}"
        if missing:
            detail += f"; missing: {missing}"
        return HealthReport(
            component="market_data", status=status, detail=detail, checked_at=now
        )

    def check_instrument_master_freshness(self, as_of: dt.date) -> HealthReport:
        now = self._clock()
        try:
            snapshot = self.universe_provider.get_universe(as_of)
        except (DataNotAvailableError, CalendarCoverageError) as exc:
            return HealthReport(
                component="instrument_master",
                status=ComponentHealth.UNHEALTHY,
                detail=f"universe could not be constructed for {as_of}: {exc}",
                checked_at=now,
            )
        if not snapshot.instrument_ids:
            return HealthReport(
                component="instrument_master",
                status=ComponentHealth.UNHEALTHY,
                detail=f"universe for {as_of} is empty",
                checked_at=now,
            )
        return HealthReport(
            component="instrument_master",
            status=ComponentHealth.HEALTHY,
            detail=f"{len(snapshot.instrument_ids)} eligible instruments as of {as_of}",
            checked_at=now,
        )

    def check_model_freshness(self, as_of: dt.date) -> HealthReport:
        now = self._clock()
        try:
            artifact = self.model_registry.load_current_approved()
        except NoApprovedModelError as exc:
            return HealthReport(
                component="model",
                status=ComponentHealth.UNHEALTHY,
                detail=f"no approved model: {exc}",
                checked_at=now,
            )
        training_end = artifact.model.training_result.training_end
        try:
            boundary = self.calendar.sessions_offset(as_of, -self.model_max_age_sessions)
        except (CalendarCoverageError, ValueError) as exc:
            return HealthReport(
                component="model",
                status=ComponentHealth.DEGRADED,
                detail=f"could not verify model age against the calendar: {exc}",
                checked_at=now,
            )
        if training_end < boundary:
            return HealthReport(
                component="model",
                status=ComponentHealth.DEGRADED,
                detail=(
                    f"approved model {artifact.model_id!r} trained through {training_end}, "
                    f"exceeds the {self.model_max_age_sessions}-session freshness window"
                ),
                checked_at=now,
            )
        return HealthReport(
            component="model",
            status=ComponentHealth.HEALTHY,
            detail=f"approved model {artifact.model_id!r} trained through {training_end}",
            checked_at=now,
        )

    def check_heartbeat(self) -> HealthReport:
        now = self._clock()
        last = self._last_heartbeat()
        if last is None:
            return HealthReport(
                component="heartbeat",
                status=ComponentHealth.HEALTHY,
                detail="no heartbeat recorded yet",
                checked_at=now,
            )
        age_seconds = (now - last).total_seconds()
        if age_seconds > self.heartbeat_interval_seconds * 3:
            status = ComponentHealth.UNHEALTHY
        elif age_seconds > self.heartbeat_interval_seconds:
            status = ComponentHealth.DEGRADED
        else:
            status = ComponentHealth.HEALTHY
        return HealthReport(
            component="heartbeat",
            status=status,
            detail=f"last heartbeat {age_seconds:.0f}s ago",
            checked_at=now,
        )

    def run_startup_sequence(self, as_of: dt.date) -> list[HealthReport]:
        """The ordered startup checks in docs/SPECIFICATION.md section
        15.1. Any failing, fail-closed condition (section 15.2) must
        prevent new signal generation from being enabled -- this method
        only reports; the caller (``orchestration.orchestrator.Orchestrator``)
        decides what to do with an ``UNHEALTHY`` result.
        """
        return [
            self.check_broker(),
            self.check_market_data_freshness(as_of),
            self.check_instrument_master_freshness(as_of),
            self.check_model_freshness(as_of),
            self.check_heartbeat(),
        ]
