"""The container health check (Phase 23).

Run as ``python -m app.cli health``; exits 0 when healthy, 1 when not.
Docker restarts an ``unhealthy`` container, so what this file decides is
*precisely* what gets restarted, and the interesting question is not what
counts as broken but what must **not** count as broken.

**A halted system is healthy.** ``SystemState.HALTED`` means this system
met one of the ``orchestration.fail_closed`` conditions and correctly
refused to trade. It is doing exactly what it should. Restarting it would
throw away the process that knows why it halted, re-run startup into the
same condition, and halt again -- a crash loop whose only visible symptom
is a restart counter, which is the outcome the whole fail-closed design
exists to avoid. A halted container must stay up, stay queryable, and
wait for an operator. ``check_health`` therefore reports the halt
prominently in its detail line while still returning healthy, so a
dashboard or an alert rule can act on it without the supervisor acting on
it first.

**What is actually unhealthy** is a process that has stopped making
progress: no state file at all (startup never got far enough to write
one), a state file that will not parse (the store is corrupt, which is a
``FailClosedReason.DATABASE_FAILURE`` nothing can trade through), or a
heartbeat older than the bound below (the process is wedged -- deadlocked,
swapping, or stopped without exiting). Those are the cases where
replacing the process is the correct and only remedy.

This runs as a separate short-lived process, so it deliberately shares no
state with the service beyond the one file on disk. A health check that
asked the service's own in-memory objects whether it was healthy would
return "yes" right up until the moment the service stopped being able to
answer at all.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.service import state_file_path
from execution.system_state import SystemState, SystemStateStore, SystemStateStoreError

DEFAULT_MAX_HEARTBEAT_AGE_SECONDS = 300.0
"""Five minutes: comfortably more than the default 60-second heartbeat
interval in ``config/settings.yaml``, so an ordinary slow iteration or a
clock adjustment never reads as a wedged process. Raising the heartbeat
interval above this in settings means raising this too, via
``--max-heartbeat-age-seconds`` in the compose HEALTHCHECK."""


@dataclass(frozen=True, slots=True)
class HealthResult:
    healthy: bool
    detail: str
    system_state: SystemState | None
    heartbeat_age_seconds: float | None


def check_health(
    *,
    state_path: Path | None = None,
    max_heartbeat_age_seconds: float = DEFAULT_MAX_HEARTBEAT_AGE_SECONDS,
    clock: Callable[[], dt.datetime] | None = None,
) -> HealthResult:
    now = (clock or (lambda: dt.datetime.now(dt.UTC)))()
    path = state_path if state_path is not None else state_file_path()

    if not path.is_file():
        return HealthResult(
            healthy=False,
            detail=(
                f"no persisted state at {path}: the service has not completed startup, "
                "or the state volume is not mounted where it expects it"
            ),
            system_state=None,
            heartbeat_age_seconds=None,
        )

    try:
        persisted = SystemStateStore(path).load()
    except SystemStateStoreError as exc:
        return HealthResult(
            healthy=False,
            detail=f"persisted state is unreadable ({exc}); this is a database failure",
            system_state=None,
            heartbeat_age_seconds=None,
        )

    if persisted is None:  # pragma: no cover - is_file() above already excludes this
        return HealthResult(
            healthy=False,
            detail=f"persisted state vanished between the existence check and the read: {path}",
            system_state=None,
            heartbeat_age_seconds=None,
        )

    updated_at = persisted.updated_at
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=dt.UTC)
    age = (now - updated_at).total_seconds()

    if age > max_heartbeat_age_seconds:
        return HealthResult(
            healthy=False,
            detail=(
                f"heartbeat is {age:.0f}s old (limit {max_heartbeat_age_seconds:.0f}s); "
                "the process is not making progress"
            ),
            system_state=persisted.system_state,
            heartbeat_age_seconds=age,
        )

    if persisted.system_state is SystemState.HALTED:
        return HealthResult(
            healthy=True,
            detail=(
                "HALTED and alive. The system has failed closed and is placing no orders. "
                "This is reported healthy on purpose so the supervisor does not restart "
                "away the process that knows why -- it requires an operator, not a restart. "
                "See docs/INCIDENT_RESPONSE.md."
            ),
            system_state=persisted.system_state,
            heartbeat_age_seconds=age,
        )

    return HealthResult(
        healthy=True,
        detail=f"{persisted.system_state.value}, heartbeat {age:.0f}s old",
        system_state=persisted.system_state,
        heartbeat_age_seconds=age,
    )


def render_health(result: HealthResult) -> str:
    return f"{'HEALTHY' if result.healthy else 'UNHEALTHY'}: {result.detail}"
