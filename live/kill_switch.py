"""The kill switch (Phase 22, pre-live checklist item 13): an operator's
explicit, immediate "stop trading now" action.

This is a thin, deliberately-named wrapper around
``risk.circuit_breaker.CircuitBreaker.force_halt`` -- the actual halt
mechanism already exists and is already respected everywhere trading
decisions get made (``risk.risk_manager.RiskManager.evaluate`` refuses
every position the instant the breaker is ``HALTED``,
``orchestration.orchestrator.Orchestrator`` treats a halted breaker as a
hard stop on both order submission and its own lifecycle state). Giving
that call its own name and module is what makes "kill switch tested" a
checkable, discoverable thing rather than a fact a reader has to already
know to find inside ``CircuitBreaker``.
"""

from __future__ import annotations

from risk.circuit_breaker import CircuitBreaker, CircuitBreakerStatus, CircuitState


class KillSwitch:
    def __init__(self, circuit_breaker: CircuitBreaker) -> None:
        self.circuit_breaker = circuit_breaker

    def engage(self, operator: str, reason: str) -> CircuitBreakerStatus:
        """Halt trading immediately, regardless of current risk state.
        Raises ``risk.circuit_breaker.CircuitBreakerError`` if ``operator``
        or ``reason`` is empty -- an anonymous, unexplained kill switch
        engagement is not permitted."""
        return self.circuit_breaker.force_halt(operator, reason)

    def is_engaged(self) -> bool:
        return self.circuit_breaker.current_status().state is CircuitState.HALTED

    def release(self, operator: str, reason: str) -> CircuitBreakerStatus:
        """Clear the halt. A thin alias for
        ``CircuitBreaker.manual_reset`` -- kept here so a caller that
        reasoned about the system in terms of "the kill switch" does not
        have to switch vocabulary to release it."""
        return self.circuit_breaker.manual_reset(operator, reason)
