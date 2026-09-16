"""Structured logging setup.

This is the one piece of ``monitoring`` implemented in Phase 1 (the spec's
own Phase 1 scope includes "logging, environment handling"). Every other
module in this package -- alerts, health checks, the dashboard -- is a stub
for later phases.

Every log record includes a UTC timestamp and the component name, so
messages from different layers can be correlated without guessing (see
docs/SPECIFICATION.md section 23, "Explicit timestamps/timezones").
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.models import LoggingConfig


class AuditLogError(RuntimeError):
    """The audit log could not be opened.

    Raised rather than swallowed: ``logging_config.audit_log_path`` is set
    only when an operator has asked for a durable audit trail, and a system
    that silently runs without the audit trail its operator believes it has
    is worse than one that refuses to start (Phase 23).
    """


class JsonFormatter(logging.Formatter):
    """Renders each log record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


def _build_formatter(config: LoggingConfig) -> logging.Formatter:
    if config.format == "json":
        return JsonFormatter()
    return logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")


def configure_logging(config: LoggingConfig) -> None:
    """Configure the root logger once, at process startup.

    Idempotent: calling this more than once replaces prior handlers rather
    than stacking duplicate output.

    Two destinations, for two different readers (Phase 23):

    * **stdout** -- always. This is what a container runtime collects, and
      what ``docker compose logs`` shows. Rotation of *this* stream is the
      runtime's job (``deploy/docker-compose.yml`` sets the ``json-file``
      driver's ``max-size``/``max-file``), because this process does not
      own the file it is being written to.
    * **a rotating file** -- only when ``config.audit_log_path`` is set.
      This is the durable audit trail: it lives on a mounted volume and so
      outlives the container that wrote it, where the stdout stream does
      not survive ``docker system prune``. Rotation here *is* this
      process's job, since it opened the file itself.

    Both carry the same records. Splitting a separate "audit" severity out
    was considered and rejected: the records an incident actually turns on
    are ordinary INFO ones (which order, at what price, on whose decision),
    and a filter deciding in advance which of those matter is a filter that
    will be wrong during the one incident that matters.
    """
    root = logging.getLogger()
    root.setLevel(config.level)
    for existing in list(root.handlers):
        existing.close()
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_build_formatter(config))
    root.addHandler(handler)

    if config.audit_log_path is None:
        return

    path = Path(config.audit_log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        audit_handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=config.audit_log_max_bytes,
            backupCount=config.audit_log_backup_count,
            encoding="utf-8",
            delay=False,
        )
    except OSError as exc:
        raise AuditLogError(
            f"could not open the audit log at {path}: {exc}. "
            "In a container this usually means the volume mounted at that "
            "path is not writable by the non-root user the image runs as."
        ) from exc

    audit_handler.setFormatter(_build_formatter(config))
    root.addHandler(audit_handler)


def get_logger(component: str) -> logging.Logger:
    """Return a named logger for one component (e.g. ``"risk.risk_manager"``)."""
    return logging.getLogger(component)
