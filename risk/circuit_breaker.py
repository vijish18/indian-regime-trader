"""Daily/weekly loss and peak-drawdown circuit breakers. See
docs/SPECIFICATION.md section 8: warning -> reduce -> halt escalation.

Not implemented yet (Phase 7/8).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CircuitState(StrEnum):
    NORMAL = "normal"
    WARNING = "warning"
    REDUCE = "reduce"
    HALT = "halt"


@dataclass(frozen=True)
class CircuitBreakerStatus:
    state: CircuitState
    daily_pnl_pct: float
    weekly_pnl_pct: float
    peak_drawdown_pct: float
    reason: str | None


class CircuitBreaker:
    """Evaluates account-level P&L/drawdown against the configured
    thresholds (``risk.daily_loss_*``, ``risk.weekly_loss_*``,
    ``risk.peak_drawdown_halt_pct``) and returns the current state.

    A HALT state must make new order creation impossible, not merely
    discouraged (docs/SPECIFICATION.md section 19, kill-switch tests).
    """

    def evaluate(
        self, daily_pnl_pct: float, weekly_pnl_pct: float, peak_drawdown_pct: float
    ) -> CircuitBreakerStatus:
        raise NotImplementedError("Phase 7/8: circuit breaker is not implemented yet.")
