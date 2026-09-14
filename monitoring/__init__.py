"""Structured logging, alerting, health checks, and dashboard. See
docs/SPECIFICATION.md section 16.
"""

from monitoring.logger import configure_logging

__all__ = ["configure_logging"]
