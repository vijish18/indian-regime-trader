"""Alert detection, rate limiting and dispatch (Phase 20), for the
conditions in docs/SPECIFICATION.md section 16.

Three things live here, in dependency order:

- the vocabulary (:class:`AlertType`, :class:`AlertSeverity`, :class:`Alert`);
- :func:`evaluate_alerts`, a *pure function* of one or two
  ``monitoring.snapshot.MonitoringSnapshot`` objects -- given the same
  snapshots it always returns the same alerts, queries nothing, and holds
  no state, which is what makes each of the ten conditions testable by
  construction rather than by simulation;
- :class:`AlertManager`, which owns the two stateful concerns: how often
  the same alert may be repeated (:class:`RateLimiter`) and where an
  alert goes once it survives that.

**Why some rules need two snapshots.** A count that is merely nonzero is
not news: three orders were rejected this morning and are still rejected
this afternoon. What deserves an alert is the *transition* -- a new
rejection, a cash balance that moved by more than the fills explain. Those
rules therefore compare the current snapshot against the previous one,
which is still pure: the caller (or :meth:`AlertManager.evaluate`) holds
the memory, the rule does not.

**Why rate limiting is not optional.** Most of these conditions persist
until someone acts on them -- a disconnected broker stays disconnected
through every iteration of the monitoring loop. Without a cooldown the
alerts that matter are buried under the ones already known, which is the
failure mode alerting exists to prevent. The cooldown is
``config.models.MonitoringConfig.alert_cooldown_seconds``, and a
suppressed alert is counted, not discarded: the next one that gets through
reports how many it stands for.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import StrEnum

from monitoring.health import ComponentHealth
from monitoring.snapshot import MonitoringSnapshot
from risk.circuit_breaker import CircuitState

logger = logging.getLogger(__name__)

SUPPORTED_CHANNELS: frozenset[str] = frozenset({"log"})
"""Delivery channels this system can actually deliver to today. Alerts are
*always* logged (that is the audit trail, not a channel); anything listed
in ``MonitoringConfig.alert_channels`` is additional delivery on top. An
unrecognized channel name is refused at construction rather than silently
dropped -- a channel an operator believes is configured but that quietly
delivers nothing is worse than no alerting at all."""


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(StrEnum):
    BROKER_DISCONNECT = "broker_disconnect"
    MARKET_DATA_DISCONNECT = "market_data_disconnect"
    STALE_DATA = "stale_data"
    ORDER_REJECTION = "order_rejection"
    UNKNOWN_ORDER_STATE = "unknown_order_state"
    RECONCILIATION_MISMATCH = "reconciliation_mismatch"
    RISK_HALT = "risk_halt"
    EXCESSIVE_DRAWDOWN = "excessive_drawdown"
    UNEXPECTED_POSITION = "unexpected_position"
    UNEXPECTED_CASH_BALANCE = "unexpected_cash_balance"


class AlertConfigurationError(RuntimeError):
    """An alert channel was configured that this system cannot deliver to."""


@dataclass(frozen=True)
class Alert:
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    source: str
    subject: str = ""
    """What the alert is about, when the type alone is not specific enough
    -- an instrument id, usually. Part of the rate-limit key, so a mismatch
    on one instrument never suppresses the alert about another."""

    raised_at: dt.datetime | None = None
    suppressed_since_last: int = 0
    """How many alerts identical to this one were rate-limited away since
    the last one was delivered."""

    @property
    def key(self) -> str:
        return f"{self.alert_type.value}:{self.subject}" if self.subject else self.alert_type.value


# --------------------------------------------------------------------------
# Detection: pure, given one or two snapshots
# --------------------------------------------------------------------------


def evaluate_alerts(
    current: MonitoringSnapshot,
    previous: MonitoringSnapshot | None = None,
    *,
    drawdown_alert_pct: float,
    cash_tolerance_pct: float = 0.001,
) -> list[Alert]:
    """Every alert condition that is true of ``current``.

    Args:
        current: the snapshot to evaluate.
        previous: the one before it, if there is one. Rules that detect a
            *change* (a new rejection, an unexplained cash move) are
            skipped when it is ``None``, except where firing on first sight
            is the safer default -- see the order-rejection rule.
        drawdown_alert_pct: drawdown, as a positive fraction, at or beyond
            which to alert. Normally set below
            ``RiskConfig.peak_to_trough_drawdown_halt_pct`` so the alert
            arrives before the circuit breaker halts rather than with it.
        cash_tolerance_pct: how far the broker's cash may drift from what
            the observed fills explain, as a fraction of equity, before it
            is treated as unexpected. Must exceed realistic cost drag: the
            fill cash flow this compares against is gross of brokerage,
            taxes and slippage.
    """
    alerts: list[Alert] = []
    alerts.extend(_connectivity_alerts(current))
    alerts.extend(_execution_alerts(current, previous))
    alerts.extend(_reconciliation_alerts(current))
    alerts.extend(_risk_alerts(current, drawdown_alert_pct))
    alerts.extend(_cash_alerts(current, previous, cash_tolerance_pct))
    return alerts


def _connectivity_alerts(current: MonitoringSnapshot) -> list[Alert]:
    alerts: list[Alert] = []
    if not current.system.broker_connected:
        alerts.append(
            Alert(
                AlertType.BROKER_DISCONNECT,
                AlertSeverity.CRITICAL,
                f"broker is not connected: {current.system.broker_detail}",
                source="broker",
            )
        )

    # A feed that cannot be reached at all and a feed that is merely behind
    # are different problems with different responses, so they are
    # different alerts -- HealthChecker already draws exactly that line.
    if current.system.data_feed is ComponentHealth.UNHEALTHY:
        alerts.append(
            Alert(
                AlertType.MARKET_DATA_DISCONNECT,
                AlertSeverity.CRITICAL,
                f"market data is unavailable: {current.system.data_feed_detail}",
                source="market_data",
            )
        )
    elif current.system.data_feed is ComponentHealth.DEGRADED:
        alerts.append(
            Alert(
                AlertType.STALE_DATA,
                AlertSeverity.WARNING,
                f"market data is stale: {current.system.data_feed_detail}",
                source="market_data",
            )
        )
    return alerts


def _execution_alerts(
    current: MonitoringSnapshot, previous: MonitoringSnapshot | None
) -> list[Alert]:
    alerts: list[Alert] = []
    rejected_now = current.execution.rejected_orders
    # On the first snapshot there is nothing to compare against, and a
    # process that restarted into an already-rejected order should not be
    # silent about it -- so first sight of any rejection alerts, and after
    # that only an increase does.
    rejected_before = previous.execution.rejected_orders if previous else 0
    if rejected_now > rejected_before:
        alerts.append(
            Alert(
                AlertType.ORDER_REJECTION,
                AlertSeverity.WARNING,
                f"{rejected_now - rejected_before} order(s) rejected "
                f"({rejected_now} rejected in total)",
                source="execution",
            )
        )

    if current.execution.orders_in_unknown_state > 0:
        alerts.append(
            Alert(
                AlertType.UNKNOWN_ORDER_STATE,
                AlertSeverity.CRITICAL,
                f"{current.execution.orders_in_unknown_state} order(s) in an unknown state; "
                "resolve against the broker before trading further",
                source="execution",
            )
        )
    return alerts


def _reconciliation_alerts(current: MonitoringSnapshot) -> list[Alert]:
    alerts: list[Alert] = []
    for mismatch in current.execution.reconciliation_mismatches:
        if mismatch.local_quantity == 0:
            # The broker holds something this system has no record of at
            # all, which is a stronger signal than a disagreement about
            # size: something outside this system traded the account.
            alerts.append(
                Alert(
                    AlertType.UNEXPECTED_POSITION,
                    AlertSeverity.CRITICAL,
                    f"broker reports {mismatch.broker_quantity} of {mismatch.instrument_id} "
                    "that this system has no record of",
                    source="reconciliation",
                    subject=mismatch.instrument_id,
                )
            )
        else:
            alerts.append(
                Alert(
                    AlertType.RECONCILIATION_MISMATCH,
                    AlertSeverity.CRITICAL,
                    f"{mismatch.instrument_id}: local={mismatch.local_quantity} "
                    f"broker={mismatch.broker_quantity} ({mismatch.detail})",
                    source="reconciliation",
                    subject=mismatch.instrument_id,
                )
            )
    return alerts


def _risk_alerts(current: MonitoringSnapshot, drawdown_alert_pct: float) -> list[Alert]:
    alerts: list[Alert] = []
    if current.risk.risk_mode is CircuitState.HALTED:
        alerts.append(
            Alert(
                AlertType.RISK_HALT,
                AlertSeverity.CRITICAL,
                f"circuit breaker HALTED: {current.risk.circuit_reason or 'no reason recorded'}",
                source="risk",
            )
        )
    if current.portfolio.drawdown_pct >= drawdown_alert_pct:
        alerts.append(
            Alert(
                AlertType.EXCESSIVE_DRAWDOWN,
                AlertSeverity.CRITICAL,
                f"drawdown {current.portfolio.drawdown_pct:.2%} has reached the "
                f"{drawdown_alert_pct:.2%} alert threshold",
                source="risk",
            )
        )
    return alerts


def _cash_alerts(
    current: MonitoringSnapshot,
    previous: MonitoringSnapshot | None,
    cash_tolerance_pct: float,
) -> list[Alert]:
    """Cash should move between two snapshots by exactly the net cash flow
    of the fills observed between them. Anything else -- a dividend, a
    fee sweep, a manual transfer, a fill this system never saw -- is by
    definition unexpected, and worth saying so even when it is benign.
    """
    if previous is None:
        return []
    expected_change = (
        current.portfolio.cumulative_fill_cash_flow
        - previous.portfolio.cumulative_fill_cash_flow
    )
    actual_change = current.portfolio.cash - previous.portfolio.cash
    discrepancy = actual_change - expected_change
    tolerance = abs(cash_tolerance_pct * current.portfolio.equity)
    if abs(discrepancy) <= tolerance:
        return []
    return [
        Alert(
            AlertType.UNEXPECTED_CASH_BALANCE,
            AlertSeverity.CRITICAL,
            f"cash moved by {actual_change:,.2f} but observed fills explain "
            f"{expected_change:,.2f} (unexplained: {discrepancy:,.2f}, "
            f"tolerance: {tolerance:,.2f})",
            source="portfolio",
        )
    ]


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


class RateLimiter:
    """Allows one alert per key per ``cooldown_seconds``, counting what it
    suppresses in between so nothing is silently lost.
    """

    def __init__(
        self, cooldown_seconds: float, clock: Callable[[], dt.datetime] | None = None
    ) -> None:
        if cooldown_seconds < 0:
            raise ValueError(f"cooldown_seconds must be >= 0, got {cooldown_seconds}")
        self.cooldown_seconds = cooldown_seconds
        self._clock: Callable[[], dt.datetime] = clock or (lambda: dt.datetime.now(dt.UTC))
        self._last_sent: dict[str, dt.datetime] = {}
        self._suppressed: dict[str, int] = {}

    def allow(self, key: str) -> tuple[bool, int]:
        """``(allowed, suppressed_since_last)``. When allowed, the
        suppressed counter for ``key`` resets."""
        now = self._clock()
        last = self._last_sent.get(key)
        if last is not None and (now - last).total_seconds() < self.cooldown_seconds:
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
            return False, self._suppressed[key]
        self._last_sent[key] = now
        suppressed = self._suppressed.pop(key, 0)
        return True, suppressed

    def suppressed_count(self, key: str) -> int:
        return self._suppressed.get(key, 0)


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


class AlertManager:
    def __init__(
        self,
        channels: list[str],
        *,
        cooldown_seconds: float,
        drawdown_alert_pct: float,
        cash_tolerance_pct: float = 0.001,
        sink: Callable[[Alert], None] | None = None,
        history_limit: int = 500,
        clock: Callable[[], dt.datetime] | None = None,
    ) -> None:
        unsupported = sorted(set(channels) - SUPPORTED_CHANNELS)
        if unsupported:
            raise AlertConfigurationError(
                f"unsupported alert channel(s): {unsupported}; this system can deliver to "
                f"{sorted(SUPPORTED_CHANNELS)} only. Remove them from "
                "monitoring.alert_channels or implement the channel before configuring it."
            )
        self.channels = list(channels)
        self.drawdown_alert_pct = drawdown_alert_pct
        self.cash_tolerance_pct = cash_tolerance_pct
        self._clock: Callable[[], dt.datetime] = clock or (lambda: dt.datetime.now(dt.UTC))
        self._rate_limiter = RateLimiter(cooldown_seconds, clock=self._clock)
        self._sink = sink
        self._history: deque[Alert] = deque(maxlen=history_limit)
        self._previous_snapshot: MonitoringSnapshot | None = None

    @property
    def history(self) -> tuple[Alert, ...]:
        """Alerts actually delivered, oldest first, most recent
        ``history_limit`` only."""
        return tuple(self._history)

    def send(self, alert: Alert) -> bool:
        """Deliver one alert unless it is inside its cooldown. Returns
        whether it was delivered."""
        allowed, suppressed = self._rate_limiter.allow(alert.key)
        if not allowed:
            return False
        delivered = replace(
            alert, raised_at=alert.raised_at or self._clock(), suppressed_since_last=suppressed
        )
        self._dispatch(delivered)
        self._history.append(delivered)
        return True

    def evaluate(self, snapshot: MonitoringSnapshot) -> list[Alert]:
        """Evaluate every rule against ``snapshot`` (and the one before it),
        deliver whatever is not rate-limited, and return what was actually
        delivered.
        """
        alerts = evaluate_alerts(
            snapshot,
            self._previous_snapshot,
            drawdown_alert_pct=self.drawdown_alert_pct,
            cash_tolerance_pct=self.cash_tolerance_pct,
        )
        self._previous_snapshot = snapshot
        return [alert for alert in alerts if self.send(alert)]

    def send_all(self, alerts: Iterable[Alert]) -> list[Alert]:
        return [alert for alert in alerts if self.send(alert)]

    def _dispatch(self, alert: Alert) -> None:
        message = alert.message
        if alert.suppressed_since_last:
            message = f"{message} [+{alert.suppressed_since_last} suppressed since last alert]"
        logger.log(
            _LOG_LEVELS[alert.severity],
            "ALERT %s: %s",
            alert.alert_type.value,
            message,
            extra={
                "extra_fields": {
                    "event": "alert",
                    "alert_type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "source": alert.source,
                    "subject": alert.subject,
                    "suppressed_since_last": alert.suppressed_since_last,
                }
            },
        )
        if self._sink is not None:
            self._sink(alert)


_LOG_LEVELS: dict[AlertSeverity, int] = {
    AlertSeverity.INFO: logging.INFO,
    AlertSeverity.WARNING: logging.WARNING,
    AlertSeverity.CRITICAL: logging.CRITICAL,
}
