"""The conditions under which this system refuses to trade (Phase 23).

Every one of these is a *fail-closed* condition: when it is true, the
correct behavior is to place no further orders and say plainly why --
never to proceed on a guess, and never to treat "I could not check" as
"the check passed".

    unknown broker state          -> do not place more orders
    stale market data             -> do not trade
    risk engine failure           -> do not trade
    database failure              -> do not trade
    configuration failure         -> do not trade
    market-calendar uncertainty   -> do not trade

**Why this is an enum and not just a log message.** Before Phase 23,
three of these six conditions raised out of
``orchestration.orchestrator.Orchestrator.run_daily_cycle`` uncaught. In
a development shell that reads as fail-closed -- the process dies, so it
certainly places no orders. Under a production supervisor that restarts
the container automatically, it reads very differently: the process
crash-loops, nothing is ever persisted about *why*, the health endpoint
cannot answer because there is no process left to answer it, and the
operator sees a restart counter instead of a reason.

So ``run_daily_cycle`` now catches all six, records which one tripped on
the returned ``DailyCycleReport``, halts, and stays alive to be asked
about it. Naming them makes that set enumerable, testable as a whole
(``tests/unit/test_fail_closed.py`` asserts every member is reachable),
and reportable on a dashboard.
"""

from __future__ import annotations

from enum import StrEnum


class FailClosedReason(StrEnum):
    UNKNOWN_BROKER_STATE = "unknown broker state"
    """An order's true state at the broker could not be established, or
    the broker reports state this system has no record of. Placing a
    further order while an earlier one's outcome is unknown risks
    doubling a position that may already exist."""

    STALE_MARKET_DATA = "stale market data"
    """The data this system would decide on is older than the configured
    freshness bound. A decision made on stale data is not a conservative
    decision -- it is a decision about a market that no longer exists."""

    RISK_ENGINE_FAILURE = "risk engine failure"
    """The independent risk layer could not produce a decision. Its veto
    is non-negotiable (docs/SPECIFICATION.md section 8), so an absent
    verdict must read as "rejected", never as "nothing objected"."""

    DATABASE_FAILURE = "database failure"
    """Persisted state could not be read or written. Without it, a
    restart cannot know what this system already did, which is precisely
    the state Phase 18's reconciliation exists to avoid trading through."""

    CONFIGURATION_FAILURE = "configuration failure"
    """Configuration failed to load or validate. Every limit this system
    respects is configured, so unreadable configuration means no limits
    are known to be in force."""

    MARKET_CALENDAR_UNCERTAINTY = "market-calendar uncertainty"
    """Whether the exchange is open on this date could not be determined
    (``data.calendar`` refuses to guess for an uncovered year). Trading
    into an unknown session state risks orders on a closed exchange or a
    special session with different rules."""
