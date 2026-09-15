"""Failure injection for the end-to-end validation (Phase 21).

Every fault here is injected at the boundary the real failure would
actually arrive at -- a broker call raising, a feed going stale, a quote's
displayed depth being thinner than an order -- and is left to run through
the system's *real* handling: ``OrderManager.submit``'s catch-and-mark-
``UNKNOWN``, ``PaperBroker``'s own rejection/partial-fill logic,
``FillTracker``'s dedup, ``StartupSequence``'s reconciliation gate. This
module manufactures the trigger; it asserts nothing and repairs nothing --
that is ``validation.scenario``'s job, checking
``validation.invariants`` after each one.

``FaultInjectingBroker`` mirrors the narrow, deliberate broker doubles
already established in ``tests/unit/test_order_reconciler.py``
(``_LostResponseBroker``) and ``tests/unit/test_startup.py``: wrap a real
broker, make exactly the calls a test needs to fail actually fail, pass
everything else straight through.
"""

from __future__ import annotations

import datetime as dt
from collections import deque
from collections.abc import Callable, Mapping

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
from data.errors import DataNotAvailableError
from data.models import Quote
from validation.paper_feed import ReplayQuoteFeed


class FaultInjectingBroker(Broker):
    """Wraps a real broker. Each method checks a per-method queue of
    exceptions to raise before delegating -- ``queue_fault(method, exc)``
    schedules one, consumed the next time that method is called. Passes
    through unchanged when nothing is queued, which is most calls, most of
    the time.
    """

    def __init__(self, inner: Broker) -> None:
        self._inner = inner
        self._faults: dict[str, deque[Exception]] = {}
        self.duplicate_next_trades = False
        self.call_counts: dict[str, int] = {}

    def queue_fault(self, method: str, exc: Exception) -> None:
        self._faults.setdefault(method, deque()).append(exc)

    def _maybe_raise(self, method: str) -> None:
        self.call_counts[method] = self.call_counts.get(method, 0) + 1
        queue = self._faults.get(method)
        if queue:
            raise queue.popleft()

    def capabilities(self) -> BrokerCapabilities:
        return self._inner.capabilities()

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        self._inner.authenticate(credentials)

    def get_account(self) -> BrokerAccount:
        self._maybe_raise("get_account")
        return self._inner.get_account()

    def get_positions(self) -> list[BrokerPosition]:
        self._maybe_raise("get_positions")
        return self._inner.get_positions()

    def get_open_orders(self) -> list[BrokerOrder]:
        self._maybe_raise("get_open_orders")
        return self._inner.get_open_orders()

    def get_order(self, order_id: str) -> BrokerOrder:
        self._maybe_raise("get_order")
        return self._inner.get_order(order_id)

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        self._maybe_raise("get_trades")
        trades = self._inner.get_trades(order_id)
        if self.duplicate_next_trades:
            self.duplicate_next_trades = False
            trades = [*trades, *trades]
        return trades

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        self._maybe_raise("get_quotes")
        return self._inner.get_quotes(instrument_ids)

    def subscribe_market_data(
        self, instrument_ids: list[str], on_tick: Callable[[BrokerQuote], None]
    ) -> Callable[[], None]:
        self._maybe_raise("subscribe_market_data")
        return self._inner.subscribe_market_data(instrument_ids, on_tick)

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        self._maybe_raise("place_order")
        return self._inner.place_order(order)

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        self._maybe_raise("modify_order")
        return self._inner.modify_order(order_id, changes)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        self._maybe_raise("cancel_order")
        return self._inner.cancel_order(order_id)

    def close_position(self, instrument_id: str) -> BrokerOrder:
        self._maybe_raise("close_position")
        return self._inner.close_position(instrument_id)

    def close_all_positions(self) -> list[BrokerOrder]:
        self._maybe_raise("close_all_positions")
        return self._inner.close_all_positions()

    def health_check(self) -> HealthStatus:
        self._maybe_raise("health_check")
        return self._inner.health_check()


class DisconnectableFeed(ReplayQuoteFeed):
    """A quote feed that can be switched into "disconnected" (every quote
    raises, simulating a lost streaming connection) or "delayed" (quotes
    keep coming, but stamped with a timestamp in the past, simulating a
    feed that has fallen behind) on command.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._disconnected = False
        self._delay: dt.timedelta = dt.timedelta(0)

    def disconnect(self) -> None:
        self._disconnected = True

    def reconnect(self) -> None:
        self._disconnected = False

    def delay_by(self, delay: dt.timedelta) -> None:
        self._delay = delay

    def clear_delay(self) -> None:
        self._delay = dt.timedelta(0)

    def get_quote(self, instrument_id: str) -> Quote:
        if self._disconnected:
            raise DataNotAvailableError(
                f"market-data connection lost; no quote for {instrument_id}"
            )
        quote = super().get_quote(instrument_id)
        if self._delay:
            quote = Quote(
                instrument_id=quote.instrument_id,
                bid=quote.bid,
                ask=quote.ask,
                last_price=quote.last_price,
                as_of=quote.as_of - self._delay,
                bid_quantity=quote.bid_quantity,
                ask_quantity=quote.ask_quantity,
            )
        return quote


def force_rejection(order: BrokerOrder, broker: Broker) -> BrokerOrder:
    """Submits ``order`` and returns the broker's response. Real brokers
    (``PaperBroker`` included) reject by *returning* an order with
    ``status="rejected"`` and a ``reject_reason``, not by raising -- so the
    caller sets ``order.quantity``/``order.limit_price`` to something the
    broker's own validation will refuse (insufficient cash, a crossed
    price) and checks ``response.status`` rather than catching anything
    here. This exercises the real rejection path (``PaperBroker._reject``),
    not a fabricated one.
    """
    return broker.place_order(order)
