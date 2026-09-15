"""Broker-neutral interface. See docs/SPECIFICATION.md section 12.

Do not add a generic "market order fallback" to any implementation of this
interface -- order types and algo-order restrictions must match the specific
broker/exchange framework in force (docs/SPECIFICATION.md section 12.2 and
section 13; NSE's retail-algo framework does not permit market orders for
algo-originated flow).

Strategy, risk, and portfolio-construction code must depend only on
:class:`Broker` -- never on a concrete adapter -- so
``broker.adapters.paper_broker.PaperBroker`` (Phase 14) and
``broker.zerodha.kite_broker.KiteBroker`` (Phase 15) are interchangeable
beneath it without any code above this interface changing. ``order_id``
throughout this interface means the client-generated ``client_order_id``
(docs/SPECIFICATION.md section 12.1's "unique client-side trade IDs"), not
a broker-assigned identifier, since that is the one ID this system
controls and can always resolve a status query by -- an adapter whose
underlying broker only supports lookup by its own broker-assigned ID (this
is true of Kite) is responsible for keeping its own
``client_order_id -> broker_order_id`` mapping to bridge the two.

Every required capability from docs/SPECIFICATION.md section 12 maps onto
exactly one method or field below:

- authentication -> :meth:`Broker.authenticate`
- account information, funds -> :meth:`Broker.get_account` (:class:`BrokerAccount`)
- positions -> :meth:`Broker.get_positions`
- orders, order status -> :meth:`Broker.get_open_orders`, :meth:`Broker.get_order`
- order placement -> :meth:`Broker.place_order`
- order modification where supported -> :meth:`Broker.modify_order`
- cancellation -> :meth:`Broker.cancel_order`
- trade/fill history -> :meth:`Broker.get_trades` (:class:`BrokerFill`)
- market data subscription where available -> :meth:`Broker.subscribe_market_data`
- connection health, broker time/session information -> :meth:`Broker.health_check`
  (:class:`HealthStatus` carries both -- whether the connection is
  currently healthy, and when this session last logged in)

"Where supported"/"where available" are not silent partial implementations:
an adapter that lacks a capability still implements the method (so calling
code never hits ``AttributeError``) but raises
:class:`broker.errors.BrokerCapabilityError` -- ``PaperBroker`` has no live
streaming feed and raises it from :meth:`Broker.subscribe_market_data`
accordingly. :class:`BrokerCapabilities` lets a caller check ahead of time
rather than discover this by catching the exception.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BrokerCapabilities:
    """What a given adapter actually supports -- queryable ahead of time,
    rather than something a caller discovers only by catching
    ``BrokerCapabilityError``.
    """

    broker_name: str
    supports_order_modification: bool
    supports_market_data_streaming: bool
    supported_exchanges: frozenset[str]
    supported_products: frozenset[str]
    supported_order_types: frozenset[str]
    supported_varieties: frozenset[str]


@dataclass(frozen=True)
class BrokerAccount:
    """"Account information" and "funds" combined -- in every Indian
    broker API this system has reviewed, both come from one endpoint.
    """

    account_id: str
    equity: float
    cash: float
    buying_power: float
    as_of: dt.datetime


@dataclass(frozen=True)
class BrokerPosition:
    instrument_id: str
    quantity: int
    avg_price: float
    product: str = "CNC"
    """The margin product this position is held under (CNC/NRML/MIS/...).
    Defaults to ``"CNC"`` (cash-and-carry delivery), this system's only
    V1 product -- a real broker can hold the same instrument under
    multiple products simultaneously, which this field distinguishes."""


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

    product: str = "CNC"
    """See :attr:`BrokerPosition.product` -- defaults to this system's
    only V1 product, cash-and-carry delivery."""

    variety: str = "regular"
    """Which order-placement route this is (a Kite Connect concept: a
    "regular" order is the plain case; "amo"/"co"/"iceberg"/"auction" are
    distinct routes with their own rules). Defaults to the plain case;
    this system does not place any of the special varieties in V1."""

    validity: str = "DAY"
    """How long the order rests if unfilled (a Kite Connect concept:
    ``DAY``/``IOC``/``TTL``). Defaults to ``DAY``; see
    ``config.models.ComplianceConfig.allowed_validities`` and
    ``docs/COMPLIANCE.md`` section 5 for why ``IOC`` is excluded from this
    system's algo-order flow."""

    tag: str | None = None
    """An optional, adapter-defined label attached to the order --
    ``None`` lets the adapter fall back to its own default (e.g.
    ``KiteBroker`` uses a truncated ``client_order_id``).
    ``broker.compliance.ComplianceGuardedBroker`` overwrites this with
    ``ComplianceConfig.algo_identifier`` before every live order, per
    NSE's algo-tagging requirement (``docs/COMPLIANCE.md`` section 4) --
    this field is broker-neutral so that injection happens above any one
    adapter, not inside ``KiteBroker`` specifically."""


@dataclass(frozen=True)
class BrokerFill:
    """One trade -- a fill against an order, not the order itself. An
    order can have many fills (partial fills, each a separate trade)."""

    trade_id: str
    order_id: str
    """The client-side ``client_order_id`` this fill belongs to (see this
    module's docstring on what ``order_id`` means throughout)."""

    instrument_id: str
    side: str
    quantity: int
    price: float
    product: str
    as_of: dt.datetime


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
    session_active: bool = True
    """Whether the broker session used for this check is still valid --
    ``True`` unconditionally for an adapter with no real session concept
    (``PaperBroker``); reflects the actual session state for a live
    adapter (a call that raises ``BrokerSessionExpiredError`` should be
    reported here as ``session_active=False``, not just as unhealthy)."""

    login_time: dt.datetime | None = None
    """When the current session was established, if known -- this
    adapter's answer to "broker time/session information": most Indian
    broker APIs (including Kite Connect) have no dedicated server-time
    endpoint, so session state is the honest, documented substitute
    rather than a fabricated clock reading."""


class Broker(ABC):
    """Abstract broker interface. Two concrete adapters exist:
    ``broker.adapters.paper_broker.PaperBroker`` (Phase 14, simulation
    only) and ``broker.zerodha.kite_broker.KiteBroker`` (Phase 15, built
    strictly from Zerodha's published Kite Connect v3 API documentation).
    Additional adapters can be added later without changing any code
    above this interface.
    """

    @abstractmethod
    def capabilities(self) -> BrokerCapabilities: ...

    @abstractmethod
    def authenticate(self, credentials: Mapping[str, str]) -> None:
        """Establish a session. Which keys ``credentials`` must carry is
        adapter-specific (documented on each adapter's own
        ``authenticate`` override) since login mechanics genuinely differ
        broker to broker -- this method is the one uniform call site,
        not a claim that every broker's login flow is the same shape.
        Raises :class:`broker.errors.BrokerAuthenticationError` on
        failure.
        """

    @abstractmethod
    def get_account(self) -> BrokerAccount: ...

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
    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        """Every fill, or (if ``order_id`` is given) only the fills
        belonging to that one order -- "trade/fill history"."""

    @abstractmethod
    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]: ...

    @abstractmethod
    def subscribe_market_data(
        self, instrument_ids: list[str], on_tick: Callable[[BrokerQuote], None]
    ) -> Callable[[], None]:
        """Stream live quotes for ``instrument_ids``, calling ``on_tick``
        for each update, until the returned ``unsubscribe`` callable is
        invoked. Raises :class:`broker.errors.BrokerCapabilityError` if
        this adapter has no live streaming feed configured -- check
        ``capabilities().supports_market_data_streaming`` first.
        """

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
