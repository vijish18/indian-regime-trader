"""Alert dispatch for the conditions in docs/SPECIFICATION.md section 16:
broker disconnect, stale market data, reconciliation mismatch, drawdown
thresholds, execution-quality deviation, order rejections, HMM
instability/staleness, and system heartbeat loss.

Not implemented yet (Phase 12).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    source: str
    severity: AlertSeverity
    message: str


class AlertManager:
    def __init__(self, channels: list[str]) -> None:
        self.channels = channels

    def send(self, alert: Alert) -> None:
        raise NotImplementedError("Phase 12: alerting is not implemented yet.")
