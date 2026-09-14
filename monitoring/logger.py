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
import sys
from datetime import UTC, datetime
from typing import Any

from config.models import LoggingConfig


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


def configure_logging(config: LoggingConfig) -> None:
    """Configure the root logger once, at process startup.

    Idempotent: calling this more than once replaces prior handlers rather
    than stacking duplicate output.
    """
    root = logging.getLogger()
    root.setLevel(config.level)
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    if config.format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root.addHandler(handler)


def get_logger(component: str) -> logging.Logger:
    """Return a named logger for one component (e.g. ``"risk.risk_manager"``)."""
    return logging.getLogger(component)
