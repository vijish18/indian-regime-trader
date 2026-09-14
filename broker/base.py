"""Broker-neutral interface. See docs/SPECIFICATION.md section 12.

Do not add a generic "market order fallback" to any implementation of this
interface -- order types and algo-order restrictions must match the specific
broker/exchange framework in force (docs/SPECIFICATION.md section 12.2 and
section 13; NSE's retail-algo framework does not permit market orders for
algo-originated flow).

Not implemented yet (Phase 10). This is the abstract contract only.
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
    (paper, in Phase 10/11); additional adapters can be added later without
    changing any code above this interface.
    """

    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_positions(self) -> list[BrokerPosition]: ...

    @abstractmethod
    def get_open_orders(self) -> list[BrokerOrder]: ...

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
