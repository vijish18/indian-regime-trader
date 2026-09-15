"""India API/algo operational controls (Phase 16). See ``docs/COMPLIANCE.md``
for the research this module implements and, just as importantly, what
remains unverified before this system may actually go live.

Two responsibilities, kept separate:

- :class:`ComplianceGate` validates a :class:`config.models.ComplianceConfig`
  once, at construction time -- it refuses to exist at all if required
  configuration is missing or still a placeholder ("the execution engine
  must refuse to operate", not degrade gracefully), and separately
  validates individual orders, order modifications, and session state
  against that configuration.
- :class:`ComplianceGuardedBroker` wraps any :class:`broker.base.Broker`
  and calls through a :class:`ComplianceGate` before every order-affecting
  operation, so the gate applies uniformly to whichever adapter it wraps
  without that adapter having to remember to call it itself.

**Never automatically substitutes a prohibited order type, validity, or
missing identifier for an allowed one.** Every failure here raises
:class:`ComplianceError` -- a hard failure -- rather than silently
adjusting the order and proceeding; this is the single behavioral
requirement this whole module exists to guarantee.

This is a third, independent gate on top of the two ``broker/factory.py``
and ``broker/zerodha/kite_broker.py`` already establish (Phase 15):
``execution.mode == "live"`` and ``enable_live_trading=True``. None of the
three alone is sufficient to place a real order.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from typing import NoReturn

from backtest.costs import TradeSide
from broker.base import (
    Broker,
    BrokerAccount,
    BrokerCapabilities,
    BrokerFill,
    BrokerOrder,
    BrokerPosition,
    BrokerQuote,
    HealthStatus,
)
from broker.errors import BrokerError, BrokerRequestError
from config.models import ComplianceConfig
from execution.order_manager import OrderState

logger = logging.getLogger(__name__)

_UNSET_STATIC_IP = "0.0.0.0"
"""The not-yet-configured placeholder used in ``settings.yaml``'s default
(paper-mode) configuration -- see ``config.models.ComplianceConfig.static_ip_primary``."""

_UNSET_ALGO_IDENTIFIER = "UNSET"
"""See ``config.models.ComplianceConfig.algo_identifier``."""


class ComplianceError(BrokerError):
    """A compliance requirement was not met -- a hard failure. Nothing in
    this codebase catches this and silently works around it; the caller
    must fix the underlying configuration or order."""


class ComplianceGate:
    """Validates ``ComplianceConfig`` once at construction, and every
    order/session against it thereafter. Holds no reference to any
    ``Broker`` -- :class:`ComplianceGuardedBroker` is what wires this to
    one; this class stays independently testable without a broker at all.
    """

    def __init__(
        self, config: ComplianceConfig, clock: Callable[[], dt.datetime] | None = None
    ) -> None:
        self.config = config
        self._clock: Callable[[], dt.datetime] = clock or (lambda: dt.datetime.now(dt.UTC))
        self._recent_order_timestamps: deque[dt.datetime] = deque()
        self._validate_configuration()

    # -- construction-time gate: "refuse to operate" -----------------------

    def _validate_configuration(self) -> None:
        if not self.config.broker_authorization_confirmed:
            self._fail(
                "missing broker authorization: broker_authorization_confirmed is False "
                "-- the broker has not been confirmed to have authorized this account "
                "for API/algo trading (docs/COMPLIANCE.md section 13). Refusing to "
                "operate a live-capable broker."
            )
        if self.config.static_ip_primary == _UNSET_STATIC_IP:
            self._fail(
                "missing static IP configuration: static_ip_primary is still the "
                f"placeholder {_UNSET_STATIC_IP!r} (docs/COMPLIANCE.md section 2). "
                "Refusing to operate a live-capable broker."
            )
        if self.config.algo_identifier == _UNSET_ALGO_IDENTIFIER:
            self._fail(
                "missing algo identifier: algo_identifier is still the placeholder "
                f"{_UNSET_ALGO_IDENTIFIER!r} (docs/COMPLIANCE.md section 4). Refusing "
                "to operate a live-capable broker."
            )
        logger.info(
            "compliance gate activated",
            extra={
                "extra_fields": {
                    "event": "compliance_gate_activated",
                    "compliance_version": self.config.compliance_version,
                    "algo_identifier": self.config.algo_identifier,
                    "max_orders_per_second": self.config.max_orders_per_second,
                    "allowed_order_types": list(self.config.allowed_order_types),
                    "allowed_validities": list(self.config.allowed_validities),
                }
            },
        )

    # -- per-call checks -----------------------------------------------

    def check_order_type(self, order_type: str) -> None:
        if order_type not in self.config.allowed_order_types:
            self._fail(
                f"unsupported order type: {order_type!r} is not in allowed_order_types "
                f"{self.config.allowed_order_types} (docs/COMPLIANCE.md section 5). "
                "Refusing to place this order -- never substituting a permitted type."
            )

    def check_validity(self, validity: str) -> None:
        if validity not in self.config.allowed_validities:
            self._fail(
                f"unsupported order validity: {validity!r} is not in allowed_validities "
                f"{self.config.allowed_validities} (docs/COMPLIANCE.md section 5). "
                "Refusing to place this order -- never substituting a permitted validity."
            )

    def check_order(self, order: BrokerOrder) -> BrokerOrder:
        """Validates ``order`` and returns the order to actually submit --
        identical to ``order`` except ``tag`` is overwritten with the
        configured algo identifier (docs/COMPLIANCE.md section 4), since
        this system, not any one adapter, is responsible for making sure
        every live order carries it.
        """
        self.check_order_type(order.order_type)
        self.check_validity(order.validity)
        return dataclasses.replace(order, tag=self.config.algo_identifier)

    def check_session(self, status: HealthStatus) -> None:
        if not status.session_active:
            self._fail(
                f"expired authentication: broker session is not active ({status.detail}). "
                "Refusing to place an order against an invalid session."
            )
        if status.login_time is None:
            self._fail(
                "expired authentication: broker session has no known login_time, so its "
                "age cannot be verified. Refusing to place an order."
            )
        age_hours = (self._clock() - status.login_time).total_seconds() / 3600.0
        if age_hours > self.config.session_max_age_hours:
            self._fail(
                f"expired authentication: session is {age_hours:.2f}h old, exceeding "
                f"session_max_age_hours={self.config.session_max_age_hours} "
                "(docs/COMPLIANCE.md section 9 -- a Kite access token is valid for one "
                "trading day only). Refusing to place an order."
            )

    def check_rate_limit(self) -> None:
        """A sliding one-second window over this gate's own record of
        recent order submissions -- a client-side throttle in front of
        the broker's own server-side 10 OPS limit (docs/COMPLIANCE.md
        section 8), not a replacement for it.
        """
        now = self._clock()
        window_start = now - dt.timedelta(seconds=1)
        while self._recent_order_timestamps and self._recent_order_timestamps[0] < window_start:
            self._recent_order_timestamps.popleft()
        if len(self._recent_order_timestamps) >= self.config.max_orders_per_second:
            self._fail(
                f"order-per-second limit exceeded: {len(self._recent_order_timestamps)} "
                f"order(s) already sent within the last second, "
                f"max_orders_per_second={self.config.max_orders_per_second} "
                "(docs/COMPLIANCE.md section 8). Refusing to place another order."
            )

    def record_order_submission(self) -> None:
        self._recent_order_timestamps.append(self._clock())

    def _fail(self, reason: str) -> NoReturn:
        logger.critical(
            "compliance hard failure",
            extra={"extra_fields": {"event": "compliance_hard_failure", "reason": reason}},
        )
        raise ComplianceError(reason)


class ComplianceGuardedBroker(Broker):
    """Wraps any ``Broker`` and enforces a ``ComplianceGate`` before every
    order-affecting operation. Read-only calls (account, positions,
    orders, trades, quotes, health, capabilities, authenticate,
    streaming) pass straight through -- observing a real account carries
    none of the risk placing a real order does.

    ``close_position``/``close_all_positions`` are reimplemented here
    (mirroring the identical construction both ``PaperBroker`` and
    ``KiteBroker`` already do internally) rather than delegated to the
    inner broker's own methods, specifically so the resulting sell order
    is tagged with the configured algo identifier the same as any other
    order -- delegating straight to ``inner.close_position`` would bypass
    this wrapper's ``place_order`` entirely, since the inner adapter
    calls its own internal method, not this wrapper's.
    """

    def __init__(self, inner: Broker, gate: ComplianceGate) -> None:
        self._inner = inner
        self._gate = gate

    @property
    def inner(self) -> Broker:
        """The wrapped broker. Not part of ``Broker`` -- an introspection
        convenience (e.g. for a caller or test that needs to confirm what
        this wrapper is actually guarding)."""
        return self._inner

    @property
    def gate(self) -> ComplianceGate:
        return self._gate

    def capabilities(self) -> BrokerCapabilities:
        return self._inner.capabilities()

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        self._inner.authenticate(credentials)

    def get_account(self) -> BrokerAccount:
        return self._inner.get_account()

    def get_positions(self) -> list[BrokerPosition]:
        return self._inner.get_positions()

    def get_open_orders(self) -> list[BrokerOrder]:
        return self._inner.get_open_orders()

    def get_order(self, order_id: str) -> BrokerOrder:
        return self._inner.get_order(order_id)

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        return self._inner.get_trades(order_id)

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        return self._inner.get_quotes(instrument_ids)

    def subscribe_market_data(
        self, instrument_ids: list[str], on_tick: Callable[[BrokerQuote], None]
    ) -> Callable[[], None]:
        return self._inner.subscribe_market_data(instrument_ids, on_tick)

    def health_check(self) -> HealthStatus:
        return self._inner.health_check()

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        checked_order = self._gate.check_order(order)
        self._gate.check_session(self._inner.health_check())
        self._gate.check_rate_limit()
        result = self._inner.place_order(checked_order)
        self._gate.record_order_submission()
        return result

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        if "order_type" in changes:
            self._gate.check_order_type(str(changes["order_type"]))
        if "validity" in changes:
            self._gate.check_validity(str(changes["validity"]))
        self._gate.check_session(self._inner.health_check())
        self._gate.check_rate_limit()
        result = self._inner.modify_order(order_id, changes)
        self._gate.record_order_submission()
        return result

    def cancel_order(self, order_id: str) -> BrokerOrder:
        self._gate.check_session(self._inner.health_check())
        self._gate.check_rate_limit()
        result = self._inner.cancel_order(order_id)
        self._gate.record_order_submission()
        return result

    def close_position(self, instrument_id: str) -> BrokerOrder:
        position = next(
            (p for p in self._inner.get_positions() if p.instrument_id == instrument_id), None
        )
        if position is None or position.quantity <= 0:
            raise BrokerRequestError(f"no open long position in {instrument_id} to close")
        quotes = self._inner.get_quotes([instrument_id])
        if not quotes:
            raise BrokerRequestError(f"no live quote for {instrument_id}")
        order = BrokerOrder(
            client_order_id=str(uuid.uuid4()),
            broker_order_id=None,
            instrument_id=instrument_id,
            side=TradeSide.SELL.value,
            quantity=position.quantity,
            order_type="LIMIT",
            limit_price=quotes[0].bid,
            status=OrderState.CREATED.value,
            product=position.product,
        )
        return self.place_order(order)

    def close_all_positions(self) -> list[BrokerOrder]:
        return [self.close_position(p.instrument_id) for p in self._inner.get_positions()]
