"""Broker-neutral interface. See docs/SPECIFICATION.md section 12.

Do not add a generic "market order fallback" to any implementation of this
interface -- order types and algo-order restrictions must match the specific
broker/exchange framework in force (docs/SPECIFICATION.md section 12.2 and
section 13; NSE's retail-algo framework does not permit market orders for
algo-originated flow).

Strategy, risk, and portfolio-construction code must depend only on
:class:`Broker` -- never on a concrete adapter -- so
``broker.adapters.paper_broker.PaperBroker`` (Phase 14) and a future
``LiveBroker`` are interchangeable beneath it without any code above this
interface changing. ``order_id`` throughout this interface means the
client-generated ``client_order_id`` (docs/SPECIFICATION.md section 12.1's
"unique client-side trade IDs"), not a broker-assigned identifier, since
that is the one ID this system controls and can always resolve a status
query by.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Account:
    equity: float
    cash: float
    buying_power: float
    as_of: dt.datetime


@dataclass(frozen=True)
class BrokerPosition:
    instrument_id: str
    quantity: int
    avg_price: float


@dataclass(frozen=True)
class BrokerOrder:
    client_order_id: str
    broker_order_id: str | None
    instrument_id: str
    side: str
    quantity: int
    order_type: str
    limit_price: float | None
    status: str
    """A raw, broker-reported status string. Every adapter must map its own
    vendor vocabulary onto ``execution.order_manager.OrderState``'s values
    exactly -- ``OrderManager`` is the layer that turns this into a typed
    state, so no caller above it should pattern-match on adapter-specific
    strings."""

    filled_quantity: int = 0
    """Cumulative quantity filled so far -- required to treat a partial
    fill as a first-class event (docs/SPECIFICATION.md section 12.2)
    rather than only ever knowing "submitted" vs. "fully filled"."""

    avg_fill_price: float | None = None
    """Quantity-weighted average price across every fill this order has
    received so far, ``None`` before the first fill."""

    reject_reason: str | None = None
    """Set only when ``status`` reports a rejection -- never free text a
    caller has to parse out of ``status`` itself."""


@dataclass(frozen=True)
class BrokerQuote:
    instrument_id: str
    bid: float
    ask: float
    last_price: float
    as_of: dt.datetime


@dataclass(frozen=True)
class HealthStatus:
    healthy: bool
    detail: str
    checked_at: dt.datetime


class Broker(ABC):
    """Abstract broker interface. One concrete adapter is implemented first
    (paper, Phase 14); additional adapters can be added later without
    changing any code above this interface.
    """

    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]: ...

    @abstractmethod
    def get_open_orders(self) -> list[BrokerOrder]: ...

    @abstractmethod
    def get_order(self, order_id: str) -> BrokerOrder:
        """The current status of one order by client-side ID, regardless of
        whether it is still open -- the query
        ``execution.order_manager.OrderManager.handle_ambiguous_response``
        must make before ever retrying a submission whose response was
        lost or delayed, per this module's docstring and
        docs/SPECIFICATION.md section 12.1. Raises if the ID is unknown to
        this broker.
        """

    @abstractmethod
    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]: ...

    @abstractmethod
    def place_order(self, order: BrokerOrder) -> BrokerOrder: ...

    @abstractmethod
    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> BrokerOrder: ...

    @abstractmethod
    def close_position(self, instrument_id: str) -> BrokerOrder: ...

    @abstractmethod
    def close_all_positions(self) -> list[BrokerOrder]: ...

    @abstractmethod
    def health_check(self) -> HealthStatus: ...
