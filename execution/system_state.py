"""Persisted system state across restarts (Phase 18). See
``execution/startup.py`` for the sequence that reads and writes this.

Mirrors ``risk.circuit_breaker.CircuitBreaker``'s own JSON-file
persistence exactly (single file, ``to_dict``/``from_dict``, an
explicit, separately-versioned schema) -- this codebase's only existing
precedent for "state that must survive a restart," reused rather than
reinvented.

**Deliberately does not persist circuit-breaker state itself.** That
already has its own dedicated, tested persistence
(``CircuitBreaker.state_path``); this module tracks it only by
reference -- ``execution.startup.StartupReport.circuit_breaker_state``
reads it directly from the real ``CircuitBreaker`` -- never by copying
its content into a second file that could drift out of sync with the
original.

**"Verify database" (Phase 18 step 3) means this store.** No real
database exists in this codebase yet (``storage/database.py`` is still
an unimplemented Phase 12 stub); ``SystemStateStore`` -- a single
JSON file on disk, exactly like ``CircuitBreaker``'s own state file --
is what actually exists to verify and persist against today. A future
phase that adds a real database would extend this module's *contract*
(load/save/verify_accessible), not invent a parallel one.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

STATE_SCHEMA_VERSION = 1
"""Bumped whenever :class:`PersistedState`'s own shape changes
incompatibly -- distinct from the *application* version below, which
changes on every release regardless of whether the persisted-state shape
did. A schema mismatch is a hard failure (see
``execution.startup.StartupSequence.run``): silently reading a
differently-shaped file as if it matched this version could
misinterpret it rather than refuse to."""


APP_VERSION_ENV_VAR = "APP_VERSION"


def _application_version() -> str:
    """The application version, or an explicit ``"unknown"`` sentinel --
    never a guess.

    Two sources, in order. Normally it is the installed package version
    from ``pyproject.toml``, the single source of truth. But the
    production image deliberately does **not** install this project (see
    the ``Dockerfile``: it installs the dependency set and then removes
    the project, so the only copy of the source in the image is the one
    at ``/app`` that actually runs). That leaves no package metadata to
    read, and Phase 23's first real container run duly recorded
    ``app_version: "unknown"`` into the state file and the audit log.

    That matters more than it looks. ``app_version`` is how an incident
    review answers "which build wrote this state?" -- and a restart
    reading a state file written by a different build is exactly the
    situation ``StartupSequence`` exists to be careful about. So the
    image sets ``APP_VERSION`` explicitly and it wins here, with
    ``tests/unit/test_deployment.py`` asserting the Dockerfile's value
    matches ``pyproject.toml`` so the two cannot drift.
    """
    from_environment = os.environ.get(APP_VERSION_ENV_VAR, "").strip()
    if from_environment:
        return from_environment
    try:
        return importlib.metadata.version("indian-regime-trader")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


APP_VERSION = _application_version()


class SystemState(StrEnum):
    STARTING = "starting"
    VERIFYING = "verifying"
    RECONCILING = "reconciling"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    READY = "ready"
    HALTED = "halted"


class SystemStateStoreError(RuntimeError):
    """The state store could not be read or written -- corrupted file,
    unwritable directory. Fail closed: never treated as "empty" or
    silently ignored, since either could hide the very discrepancy this
    phase exists to catch."""


@dataclass(frozen=True, slots=True)
class PortfolioPositionSnapshot:
    instrument_id: str
    quantity: int
    avg_price: float


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    as_of: dt.datetime
    cash: float
    positions: tuple[PortfolioPositionSnapshot, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "cash": self.cash,
            "positions": [asdict(p) for p in self.positions],
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            as_of=dt.datetime.fromisoformat(payload["as_of"]),
            cash=payload["cash"],
            positions=tuple(
                PortfolioPositionSnapshot(**p) for p in payload["positions"]
            ),
        )


@dataclass(frozen=True, slots=True)
class PersistedState:
    schema_version: int
    app_version: str
    system_state: SystemState
    model_version: str | None
    strategy_version: str
    portfolio_snapshot: PortfolioSnapshot | None
    last_market_data_timestamp: dt.datetime | None
    last_broker_event_id: str | None
    last_broker_event_timestamp: dt.datetime | None
    updated_at: dt.datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "app_version": self.app_version,
            "system_state": self.system_state.value,
            "model_version": self.model_version,
            "strategy_version": self.strategy_version,
            "portfolio_snapshot": (
                self.portfolio_snapshot.to_dict() if self.portfolio_snapshot else None
            ),
            "last_market_data_timestamp": (
                self.last_market_data_timestamp.isoformat()
                if self.last_market_data_timestamp
                else None
            ),
            "last_broker_event_id": self.last_broker_event_id,
            "last_broker_event_timestamp": (
                self.last_broker_event_timestamp.isoformat()
                if self.last_broker_event_timestamp
                else None
            ),
            "updated_at": self.updated_at.isoformat(),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> PersistedState:
        return PersistedState(
            schema_version=payload["schema_version"],
            app_version=payload["app_version"],
            system_state=SystemState(payload["system_state"]),
            model_version=payload["model_version"],
            strategy_version=payload["strategy_version"],
            portfolio_snapshot=(
                PortfolioSnapshot.from_dict(payload["portfolio_snapshot"])
                if payload["portfolio_snapshot"]
                else None
            ),
            last_market_data_timestamp=(
                dt.datetime.fromisoformat(payload["last_market_data_timestamp"])
                if payload["last_market_data_timestamp"]
                else None
            ),
            last_broker_event_id=payload["last_broker_event_id"],
            last_broker_event_timestamp=(
                dt.datetime.fromisoformat(payload["last_broker_event_timestamp"])
                if payload["last_broker_event_timestamp"]
                else None
            ),
            updated_at=dt.datetime.fromisoformat(payload["updated_at"]),
        )


class SystemStateStore:
    """Single-JSON-file persistence for :class:`PersistedState`, in the
    same style as ``risk.circuit_breaker.CircuitBreaker``'s own state
    file."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    def load(self) -> PersistedState | None:
        """``None`` if this is a first-ever run (no file yet). Raises
        :class:`SystemStateStoreError` if the file exists but cannot be
        parsed -- a corrupted state file is never treated as an empty
        one, since that would silently discard exactly the discrepancy
        history this phase exists to preserve.
        """
        if not self.state_path.is_file():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raise SystemStateStoreError(
                f"state file is corrupted or unreadable: {self.state_path}: {exc}"
            ) from exc
        try:
            return PersistedState.from_dict(payload)
        except (KeyError, ValueError, TypeError) as exc:
            raise SystemStateStoreError(
                f"state file does not match the expected shape: {self.state_path}: {exc}"
            ) from exc

    def save(self, state: PersistedState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    def verify_accessible(self) -> None:
        """"Verify database" (Phase 18 step 3): confirms the store's
        directory is writable and, if a state file already exists, that
        it actually parses. Fails closed
        (:class:`SystemStateStoreError`) rather than silently treating an
        inaccessible or corrupted store as empty.
        """
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Creating the directory can fail on its own -- a file already
            # occupies the path, a volume is mounted read-only, a parent
            # is missing on a filesystem that will not create it. Phase 23
            # found this escaping as a raw OSError, which callers that
            # correctly catch SystemStateStoreError would not have seen:
            # the store's failures must all arrive as the store's own
            # error type, or "fail closed" quietly has a hole in it.
            raise SystemStateStoreError(
                f"state store directory could not be created: {self.state_path.parent}: {exc}"
            ) from exc
        probe = self.state_path.parent / f".probe-{uuid.uuid4().hex}"
        try:
            probe.write_text("ok", encoding="utf-8")
        except OSError as exc:
            raise SystemStateStoreError(
                f"state store directory is not writable: {self.state_path.parent}: {exc}"
            ) from exc
        finally:
            probe.unlink(missing_ok=True)
        self.load()  # raises SystemStateStoreError itself if corrupted
