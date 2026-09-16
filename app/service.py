"""The long-running process a production deployment supervises (Phase 23).

``docker compose`` needs something to keep alive: a health check needs a
process to answer it, and ``restart: unless-stopped`` needs a process
whose exit means something. That is this module.

**What it honestly does.** It performs every startup obligation this
repository can currently discharge without a composition root:
configuration loads and validates, logging is configured (including the
durable audit trail), live mode is refused, the persisted state store is
proven writable and parseable, and a heartbeat is written on an interval
so an external health check can tell a wedged process from a working one.

**What it does not do.** It does not run a trading day. Doing so requires
a market-data provider, a populated holiday calendar, an approved model
artifact and a constructed broker -- none of which exist until this
deployment is provisioned with real data and credentials (the same gap
``main.py`` names, and which ``tests/unit/test_orchestrator.py``'s
``Harness`` shows the shape of). The service says so in its logs on every
start rather than presenting an idle process as a trading one. When that
wiring lands, the loop below becomes a call to
``orchestration.orchestrator.Orchestrator.run_forever``, which already
implements the daily cycle and the fail-closed handling.

**Two different failure behaviors, deliberately.**

*Startup* failures exit non-zero. Nothing has been started, there is no
state worth preserving in memory, and a restart is a legitimate retry for
the transient cause these usually have in a container (a volume not yet
mounted, a secret not yet injected). The reason is logged at CRITICAL
before the exit, so it survives in the container log that Docker keeps.

*In-session* failures halt and stay alive -- see
``orchestration.fail_closed`` for why. A process that dies cannot be
asked why it died, and under an automatic restart policy it becomes a
crash loop that buries the reason behind a restart counter.
"""

from __future__ import annotations

import datetime as dt
import os
import signal
import sys
import types
from collections.abc import Callable
from pathlib import Path

from config.loader import ConfigError, load_environment, load_settings
from config.models import Settings
from execution.system_state import (
    APP_VERSION,
    STATE_SCHEMA_VERSION,
    PersistedState,
    SystemState,
    SystemStateStore,
    SystemStateStoreError,
)
from monitoring.logger import AuditLogError, configure_logging, get_logger
from orchestration.fail_closed import FailClosedReason

logger = get_logger("app.service")

DEFAULT_STATE_FILE = Path("state/system_state.json")
"""Relative to the working directory, matching the ``data_cache`` paths in
``config/settings.yaml``. The container's WORKDIR is ``/app`` and the state
volume mounts at ``/app/state``."""

STATE_FILE_ENV_VAR = "STATE_FILE"

UNWIRED_STRATEGY_VERSION = "unknown"
"""What this service records as the strategy version on a *first* run,
matching ``execution.startup.StartupSequence``'s own default. It never
overwrites a version a real orchestrated run already persisted -- see
``_write_heartbeat``."""

EXIT_OK = 0
EXIT_STARTUP_FAILED = 1
EXIT_LIVE_MODE_REFUSED = 3
"""A distinct code so a deploy pipeline can tell "this deployment tried to
go live and was refused" from an ordinary misconfiguration."""


class ServiceStartupError(RuntimeError):
    """A startup obligation could not be discharged. Carries the
    fail-closed condition responsible, so the log line names the same
    vocabulary the orchestrator uses."""

    def __init__(self, reason: FailClosedReason, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


def state_file_path(environ: dict[str, str] | None = None) -> Path:
    """Where persisted state lives, per deployment.

    An environment variable rather than a ``settings.yaml`` key because it
    is a property of the host's filesystem layout, not of the strategy --
    the same reasoning that puts ``DATABASE_URL`` in the environment.
    """
    env = os.environ if environ is None else environ
    configured = env.get(STATE_FILE_ENV_VAR, "").strip()
    return Path(configured) if configured else DEFAULT_STATE_FILE


def _load_configuration() -> Settings:
    try:
        load_environment()
        settings = load_settings()
    except ConfigError as exc:
        raise ServiceStartupError(FailClosedReason.CONFIGURATION_FAILURE, str(exc)) from exc

    try:
        configure_logging(settings.logging)
    except AuditLogError as exc:
        raise ServiceStartupError(FailClosedReason.CONFIGURATION_FAILURE, str(exc)) from exc

    return settings


def _refuse_live_mode(settings: Settings) -> None:
    """This deployment does not turn live trading on.

    Phase 22 put four independent gates in front of a live-capable broker
    (``broker.factory.build_broker``), and deploying is not one of them.
    A container image is copied between hosts, promoted between
    environments and restarted by a supervisor; none of those events
    involve a human deciding to risk real money, so none of them may be
    the thing that causes real money to be risked. Going live is a
    deliberate, separately-authorized change, documented in
    ``docs/PRE_LIVE_CHECKLIST.md``.
    """
    if settings.execution.mode == "live":
        raise ServiceStartupError(
            FailClosedReason.CONFIGURATION_FAILURE,
            "execution.mode is 'live', and this service refuses to start in live mode. "
            "Deployment is not one of the gates that authorizes live trading: see "
            "docs/PRE_LIVE_CHECKLIST.md and run `python -m app.cli preflight` first.",
        )


def _verify_state_store(path: Path) -> SystemStateStore:
    store = SystemStateStore(path)
    try:
        store.verify_accessible()
    except SystemStateStoreError as exc:
        raise ServiceStartupError(FailClosedReason.DATABASE_FAILURE, str(exc)) from exc
    return store


def _write_heartbeat(
    store: SystemStateStore,
    state: SystemState,
    *,
    now: dt.datetime,
    strategy_version: str,
) -> None:
    """Refresh the persisted state, preserving everything a previous run
    recorded.

    Read-modify-write rather than write-fresh: the portfolio snapshot and
    the last broker event id in this file are what Phase 18's
    reconciliation reads after a restart. A heartbeat that flattened them
    would quietly destroy the record of what this system had already done.
    """
    previous = store.load()
    store.save(
        PersistedState(
            schema_version=STATE_SCHEMA_VERSION,
            app_version=APP_VERSION,
            system_state=state,
            model_version=previous.model_version if previous else None,
            strategy_version=previous.strategy_version if previous else strategy_version,
            portfolio_snapshot=previous.portfolio_snapshot if previous else None,
            last_market_data_timestamp=previous.last_market_data_timestamp if previous else None,
            last_broker_event_id=previous.last_broker_event_id if previous else None,
            last_broker_event_timestamp=(
                previous.last_broker_event_timestamp if previous else None
            ),
            updated_at=now,
        )
    )


class _ShutdownRequest:
    """Set by SIGTERM/SIGINT; read by the loop.

    A flag rather than an exception, because a signal can arrive in the
    middle of writing the state file and unwinding from there could leave
    it half-written.
    """

    def __init__(self) -> None:
        self.requested = False
        self.signal_name = ""

    def request(self, signum: int, _frame: types.FrameType | None) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name


SHUTDOWN_POLL_SECONDS = 1.0
"""How finely the heartbeat wait is chopped up so a shutdown signal is
noticed promptly. See :func:`_wait_for_shutdown`."""


def _wait_for_shutdown(
    shutdown: _ShutdownRequest, seconds: float, sleep: Callable[[float], None]
) -> None:
    """Wait up to ``seconds``, returning early once shutdown is requested.

    A plain ``sleep(seconds)`` looks equivalent and is not. Since PEP 475,
    ``time.sleep`` *resumes* after a signal handler returns rather than
    raising, so a SIGTERM arriving one second into a 60-second heartbeat
    interval sets the flag and then sleeps for the remaining 59. Docker's
    ``stop_grace_period`` is 30 seconds, so the container would be
    SIGKILLed every single time and the shutdown path would never run --
    on a *trading* process, whose shutdown path is the part that stops
    cleanly rather than mid-write.

    This was not caught by any unit test, because a unit test passes its
    own ``sleep_fn`` and never blocks. It was caught by
    ``tests/integration/test_docker_smoke.py`` actually stopping a real
    container and looking for the shutdown log line, which is the reason
    that test builds an image instead of trusting this one.
    """
    remaining = seconds
    while remaining > 0 and not shutdown.requested:
        slice_seconds = min(SHUTDOWN_POLL_SECONDS, remaining)
        sleep(slice_seconds)
        remaining -= slice_seconds


def run_service(
    *,
    max_iterations: int | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    clock: Callable[[], dt.datetime] | None = None,
    install_signal_handlers: bool = True,
) -> int:
    """Start up, then heartbeat until asked to stop. Returns an exit code.

    ``max_iterations`` and ``sleep_fn`` exist so the smoke tests can drive
    this to completion without waiting in real time.
    """
    import time

    now = clock or (lambda: dt.datetime.now(dt.UTC))
    sleep = sleep_fn or time.sleep

    try:
        settings = _load_configuration()
        _refuse_live_mode(settings)
        store = _verify_state_store(state_file_path())
    except ServiceStartupError as exc:
        # configure_logging may not have run, so this must be visible
        # regardless of logging state: stderr is collected by the
        # container runtime exactly as stdout is.
        message = f"FAIL CLOSED [{exc.reason.value}]: {exc.detail}"
        print(message, file=sys.stderr, flush=True)
        logger.critical(
            "refusing to start: %s",
            exc.detail,
            extra={
                "extra_fields": {
                    "event": "service_startup_refused",
                    "reason": exc.reason.value,
                }
            },
        )
        if exc.reason is FailClosedReason.CONFIGURATION_FAILURE and "live mode" in exc.detail:
            return EXIT_LIVE_MODE_REFUSED
        return EXIT_STARTUP_FAILED

    interval = float(settings.monitoring.heartbeat_interval_seconds)
    logger.info(
        "service started",
        extra={
            "extra_fields": {
                "event": "service_started",
                "app_version": APP_VERSION,
                "execution_mode": settings.execution.mode,
                "state_file": str(store.state_path),
                "heartbeat_interval_seconds": interval,
                "audit_log_path": settings.logging.audit_log_path,
            }
        },
    )
    logger.warning(
        "no composition root is wired, so no trading day will run: this process is "
        "keeping configuration, logging and persisted state live only. See "
        "app/service.py's module docstring.",
        extra={"extra_fields": {"event": "service_not_trading"}},
    )

    shutdown = _ShutdownRequest()
    if install_signal_handlers:
        signal.signal(signal.SIGINT, shutdown.request)
        signal.signal(signal.SIGTERM, shutdown.request)

    state = SystemState.READY
    iterations = 0
    while not shutdown.requested:
        if max_iterations is not None and iterations >= max_iterations:
            break
        iterations += 1
        try:
            _write_heartbeat(
                store, state, now=now(), strategy_version=UNWIRED_STRATEGY_VERSION
            )
        except SystemStateStoreError:
            # Fail closed in place: halt, keep the process alive so it can
            # still be asked what went wrong, and keep trying -- the cause
            # is often a full or remounted volume, which an operator can
            # fix underneath a running container.
            state = SystemState.HALTED
            logger.critical(
                "could not write persisted state; halting",
                exc_info=True,
                extra={
                    "extra_fields": {
                        "event": "service_fail_closed",
                        "reason": FailClosedReason.DATABASE_FAILURE.value,
                    }
                },
            )
        if shutdown.requested:
            break
        _wait_for_shutdown(shutdown, interval, sleep)

    logger.info(
        "shutting down",
        extra={
            "extra_fields": {
                "event": "service_shutdown",
                "signal": shutdown.signal_name or None,
                "iterations": iterations,
                "final_state": state.value,
            }
        },
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised via `python -m app.cli serve`
    raise SystemExit(run_service())
