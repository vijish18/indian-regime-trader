"""Stress-testing and failure-simulation scenarios. See
docs/SPECIFICATION.md section 11.

The system should fail closed for risk-critical uncertainty in every
scenario here: stop new entries, reduce risk, or require manual
intervention -- never guess.

Not implemented yet (Phase 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StressScenario(StrEnum):
    GAP_DOWN = "gap_down"
    CIRCUIT_LIMIT = "circuit_limit"
    SPREAD_WIDENING = "spread_widening"
    MARKET_HALT = "market_halt"
    BROKER_OUTAGE = "broker_outage"
    WEBSOCKET_DISCONNECT = "websocket_disconnect"
    DUPLICATE_ORDER_RESPONSE = "duplicate_order_response"
    PARTIAL_FILL_THEN_RESTART = "partial_fill_then_restart"
    STALE_INSTRUMENT_METADATA = "stale_instrument_metadata"
    CORPORATE_ACTION_MID_FLIGHT = "corporate_action_mid_flight"
    WRONG_REGIME_CLASSIFICATION = "wrong_regime_classification"
    MISSING_FEATURE_DATA = "missing_feature_data"


@dataclass(frozen=True)
class StressTestResult:
    scenario: StressScenario
    system_failed_closed: bool
    detail: str


class StressTestSuite:
    """Runs each scenario against the backtest/paper-trading stack and
    asserts the system's response is fail-closed, not undefined behavior.
    """

    def run_scenario(self, scenario: StressScenario, **params: object) -> StressTestResult:
        raise NotImplementedError("Phase 9: stress testing is not implemented yet.")

    def run_all(self) -> list[StressTestResult]:
        raise NotImplementedError("Phase 9: stress testing is not implemented yet.")
