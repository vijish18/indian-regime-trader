"""The application-lifecycle state machine (Phase 19) -- distinct from, and
at a higher level than, ``execution.system_state.SystemState``, which is
narrowly scoped to ``execution.startup.StartupSequence``'s own internal
restart-recovery bookkeeping. ``OrchestratorState`` describes what the
*whole running process* is doing right now; a ``StartupSequence.run()``
call is one thing ``Orchestrator`` does on its way from ``STARTING`` to
``READY``, not something this enum tracks in parallel.
"""

from __future__ import annotations

from enum import StrEnum


class OrchestratorState(StrEnum):
    STARTING = "starting"
    """Process has just started; nothing has been verified yet."""

    HEALTH_CHECK = "health_check"
    """Verifying configuration, market calendar, data availability, and
    broker connectivity -- steps 1-5 of the daily workflow."""

    RECONCILING = "reconciling"
    """Comparing local state against the broker's -- step 6."""

    READY = "ready"
    """Startup fully verified and reconciled; not yet trading."""

    RUNNING = "running"
    """Actively executing the daily workflow and/or its ongoing monitoring
    loop -- steps 7-20."""

    DEGRADED = "degraded"
    """Something is wrong but not fatal: a periodic reconciliation found a
    quarantined instrument, or a non-critical health check failed. New
    order submission is paused; monitoring and periodic reconciliation
    continue, and the orchestrator can return to RUNNING once whatever
    caused this resolves on its own (e.g. the next reconciliation comes
    back clean)."""

    HALTED = "halted"
    """The circuit breaker is HALTED, or a health check failed in a way
    that requires operator attention. No orders are submitted -- not even
    exits. Recovery requires an explicit operator action
    (``risk.circuit_breaker.CircuitBreaker.manual_reset`` or
    ``execution.startup.StartupSequence.acknowledge_and_recover``), never
    automatic."""

    SHUTTING_DOWN = "shutting_down"
    """A shutdown was requested (SIGINT/SIGTERM or an explicit call).
    Draining in-flight work and persisting final state; positions are left
    exactly as they are unless the orchestrator was explicitly configured
    to close them on shutdown."""
