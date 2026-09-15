"""Account-level circuit breaker: NORMAL / REDUCED_RISK / HALTED.

Independent of the HMM by construction -- ``evaluate()`` never sees a
``RegimeState`` or ``AllocationTarget``, only measured P&L/drawdown numbers
and two operational health flags (system health, broker connectivity). A
market-regime classification can never talk this breaker out of tripping,
and a tripped breaker can never be reasoned away by "the regime looks fine."

Three checks force HALTED outright, bypassing the loss tiers entirely:
broker disconnection, system unhealthiness, and any drawdown tier's "halt"
threshold (daily, rolling-window, or peak-to-trough -- see
``config.models.RiskConfig``). A "reduce" threshold trips REDUCED_RISK.
Crossing only a "warning" threshold (``daily_loss_warning_pct``) is logged
but leaves the state unchanged -- a warning is an observability signal, not
an operational state.

**HALTED is sticky.** Once persisted, ``evaluate()`` returns the persisted
HALTED status unchanged on every subsequent call, no matter what the current
numbers say -- there is no automatic recovery path out of a halt. The only
way out is :meth:`CircuitBreaker.manual_reset`, an explicit, separately
logged action. REDUCED_RISK has no such stickiness: it clears back to NORMAL
on its own as soon as the triggering metric recovers, since it was never a
"critical" halt in the first place.

**Persistence.** State is written to a single JSON file
(``state_path``) after every transition, so a HALTED state survives an
application restart -- a fresh ``CircuitBreaker`` pointed at the same path
picks up exactly where the last process left off, per
docs/SPECIFICATION.md section 19's kill-switch requirement that a halt must
"make new order creation impossible, not merely discouraged."
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from config.models import RiskConfig
from monitoring.logger import get_logger
from risk.portfolio_risk_state import PortfolioRiskState

logger = get_logger("risk.circuit_breaker")


class CircuitState(StrEnum):
    NORMAL = "normal"
    REDUCED_RISK = "reduced_risk"
    HALTED = "halted"


class CircuitBreakerError(RuntimeError):
    """Raised by an operation that only makes sense in a state the breaker
    is not currently in (e.g. manually resetting a breaker that is not
    halted)."""


@dataclass(frozen=True, slots=True)
class CircuitBreakerStatus:
    state: CircuitState
    as_of: dt.datetime
    reason: str | None
    triggered_by: str | None
    """Which check tripped this state, e.g. ``"daily_loss_halt"``,
    ``"broker_connectivity"``. ``None`` in NORMAL with nothing triggered."""

    requires_manual_recovery: bool
    halted_at: dt.datetime | None

    def __post_init__(self) -> None:
        if self.state is CircuitState.HALTED:
            if self.halted_at is None:
                raise ValueError("a HALTED status must record halted_at")
            if not self.requires_manual_recovery:
                raise ValueError("a HALTED status must require manual recovery")
        else:
            if self.halted_at is not None:
                raise ValueError("halted_at must be None outside HALTED")
            if self.requires_manual_recovery:
                raise ValueError("only HALTED requires manual recovery")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["as_of"] = self.as_of.isoformat()
        payload["halted_at"] = self.halted_at.isoformat() if self.halted_at else None
        return payload

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> CircuitBreakerStatus:
        return CircuitBreakerStatus(
            state=CircuitState(payload["state"]),
            as_of=dt.datetime.fromisoformat(payload["as_of"]),
            reason=payload["reason"],
            triggered_by=payload["triggered_by"],
            requires_manual_recovery=payload["requires_manual_recovery"],
            halted_at=dt.datetime.fromisoformat(payload["halted_at"])
            if payload["halted_at"]
            else None,
        )


def _baseline_status(as_of: dt.datetime) -> CircuitBreakerStatus:
    """The state of a breaker that has never evaluated anything and has no
    persisted history -- NORMAL, nothing triggered."""
    return CircuitBreakerStatus(
        state=CircuitState.NORMAL,
        as_of=as_of,
        reason=None,
        triggered_by=None,
        requires_manual_recovery=False,
        halted_at=None,
    )


class CircuitBreaker:
    """Evaluates account-level P&L/drawdown and operational health against
    ``RiskConfig``'s thresholds, persisting every transition so a halt
    survives a restart.
    """

    def __init__(self, config: RiskConfig, state_path: Path) -> None:
        self.config = config
        self.state_path = state_path

    # -- persistence ------------------------------------------------------

    def current_status(self) -> CircuitBreakerStatus:
        """The persisted status, or a fresh NORMAL baseline if this breaker
        has never evaluated or persisted anything yet."""
        if not self.state_path.is_file():
            return _baseline_status(dt.datetime.now(dt.UTC))
        return CircuitBreakerStatus.from_dict(
            json.loads(self.state_path.read_text(encoding="utf-8"))
        )

    def _persist(self, status: CircuitBreakerStatus) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(status.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, state: PortfolioRiskState) -> CircuitBreakerStatus:
        """Compute the current circuit state from ``state`` and persist any
        transition. A persisted HALTED status is returned unchanged --
        evaluation never auto-recovers a halt.
        """
        persisted = self.current_status()
        if persisted.state is CircuitState.HALTED:
            return persisted

        candidate = self._derive_status(state)
        if candidate.state != persisted.state:
            self._log_transition(persisted.state, candidate)
            self._persist(candidate)
        elif candidate.triggered_by == "daily_loss_warning":
            # A same-state warning is still a loggable event, just not a
            # transition worth persisting.
            logger.warning(
                "risk warning threshold crossed: %s",
                candidate.reason,
                extra={"extra_fields": {"event": "risk_warning", "reason": candidate.reason}},
            )
        return candidate

    def _derive_status(self, state: PortfolioRiskState) -> CircuitBreakerStatus:
        cfg = self.config
        now = state.as_of

        if not state.broker_connected:
            return self._halted(
                now, "broker_connectivity", state.broker_detail or "broker disconnected"
            )
        if not state.system_healthy:
            return self._halted(now, "system_health", state.system_detail or "system unhealthy")

        for trigger, observed, halt_limit in (
            ("daily_loss_halt", state.daily_pnl_pct, cfg.daily_loss_halt_pct),
            ("rolling_loss_halt", state.rolling_pnl_pct, cfg.rolling_loss_halt_pct),
            (
                "peak_to_trough_drawdown_halt",
                state.peak_to_trough_drawdown_pct,
                cfg.peak_to_trough_drawdown_halt_pct,
            ),
        ):
            if observed >= halt_limit:
                return self._halted(
                    now,
                    trigger,
                    f"{trigger}: observed {observed:.4f} >= limit {halt_limit:.4f}",
                )

        for trigger, observed, reduce_limit in (
            ("daily_loss_reduce", state.daily_pnl_pct, cfg.daily_loss_reduce_pct),
            ("rolling_loss_reduce", state.rolling_pnl_pct, cfg.rolling_loss_reduce_pct),
        ):
            if observed >= reduce_limit:
                return CircuitBreakerStatus(
                    state=CircuitState.REDUCED_RISK,
                    as_of=now,
                    reason=f"{trigger}: observed {observed:.4f} >= limit {reduce_limit:.4f}",
                    triggered_by=trigger,
                    requires_manual_recovery=False,
                    halted_at=None,
                )

        if state.daily_pnl_pct >= cfg.daily_loss_warning_pct:
            return CircuitBreakerStatus(
                state=CircuitState.NORMAL,
                as_of=now,
                reason=(
                    f"daily_loss_warning: observed {state.daily_pnl_pct:.4f} "
                    f">= limit {cfg.daily_loss_warning_pct:.4f}"
                ),
                triggered_by="daily_loss_warning",
                requires_manual_recovery=False,
                halted_at=None,
            )

        return _baseline_status(now)

    @staticmethod
    def _halted(as_of: dt.datetime, trigger: str, reason: str) -> CircuitBreakerStatus:
        return CircuitBreakerStatus(
            state=CircuitState.HALTED,
            as_of=as_of,
            reason=reason,
            triggered_by=trigger,
            requires_manual_recovery=True,
            halted_at=as_of,
        )

    def _log_transition(self, previous: CircuitState, new: CircuitBreakerStatus) -> None:
        level = logger.critical if new.state is CircuitState.HALTED else logger.warning
        level(
            "circuit breaker transition: %s -> %s (%s)",
            previous.value,
            new.state.value,
            new.reason,
            extra={
                "extra_fields": {
                    "event": "circuit_breaker_transition",
                    "previous_state": previous.value,
                    "new_state": new.state.value,
                    "triggered_by": new.triggered_by,
                    "reason": new.reason,
                }
            },
        )

    # -- manual recovery ------------------------------------------------------

    def manual_reset(self, operator: str, reason: str) -> CircuitBreakerStatus:
        """Explicitly clear a HALTED breaker back to NORMAL.

        Raises :class:`CircuitBreakerError` if the breaker is not currently
        halted -- there is nothing to reset, and silently no-op'ing here
        would hide a caller's mistaken assumption about the current state.
        This is the *only* way a HALTED breaker ever leaves that state; no
        code path in :meth:`evaluate` performs this transition.
        """
        persisted = self.current_status()
        if persisted.state is not CircuitState.HALTED:
            raise CircuitBreakerError(
                f"cannot manually reset a breaker that is not halted (currently {persisted.state})"
            )
        if not operator or not reason:
            raise CircuitBreakerError("manual_reset requires a non-empty operator and reason")

        now = dt.datetime.now(dt.UTC)
        new_status = CircuitBreakerStatus(
            state=CircuitState.NORMAL,
            as_of=now,
            reason=f"manually reset by {operator}: {reason}",
            triggered_by=None,
            requires_manual_recovery=False,
            halted_at=None,
        )
        logger.critical(
            "circuit breaker manually reset: halted -> normal (operator=%s, reason=%s)",
            operator,
            reason,
            extra={
                "extra_fields": {
                    "event": "circuit_breaker_manual_reset",
                    "operator": operator,
                    "reason": reason,
                }
            },
        )
        self._persist(new_status)
        return new_status

    # -- kill switch ------------------------------------------------------

    def force_halt(self, operator: str, reason: str) -> CircuitBreakerStatus:
        """The kill switch: immediately halt trading regardless of the
        portfolio's current risk state, bypassing :meth:`evaluate`'s
        threshold logic entirely.

        For an operator-invoked emergency stop -- "something looks wrong,
        stop trading right now, investigate after" -- never for an
        automatic response to a breached limit (that is exactly what
        :meth:`evaluate` already does). See ``live.kill_switch.KillSwitch``,
        which exists to give this exact call its own clear, discoverable
        name at the call site.

        Persists even if the breaker is already ``HALTED``, updating the
        recorded operator/reason/timestamp -- a kill switch engaged twice
        should still show its second invocation, not silently no-op.
        """
        if not operator or not reason:
            raise CircuitBreakerError("force_halt requires a non-empty operator and reason")
        previous = self.current_status()
        now = dt.datetime.now(dt.UTC)
        new_status = self._halted(
            now, "kill_switch", f"kill switch engaged by {operator}: {reason}"
        )
        self._log_transition(previous.state, new_status)
        self._persist(new_status)
        return new_status
