"""Zerodha Kite Connect v3 REST adapter.

Every endpoint path, request parameter, response field, header, and
status value below is taken from Zerodha's published documentation
(https://kite.trade/docs/connect/v3/) as fetched and verified while
building this module -- nothing here is guessed. Where the docs leave a
gap (e.g. no dedicated "server time" endpoint, no client-side-ID lookup
for an order), that gap is documented at the point it matters rather
than papered over with an invented endpoint.

## Base URL, auth, and the response envelope

- REST base URL: ``https://api.kite.trade``
- Login URL: ``https://kite.zerodha.com/connect/login?v=3&api_key=...`` --
  a human completes login in a browser; Kite redirects back with a
  one-time ``request_token``. This adapter never automates that browser
  step (there is no documented API for it) -- :meth:`KiteBroker.authenticate`
  takes the resulting ``request_token`` as input.
- Session exchange: ``POST /session/token`` with ``api_key``,
  ``request_token``, and ``checksum`` (``sha256(api_key + request_token +
  api_secret)``), returning ``access_token`` among other fields.
- Every subsequent request carries ``Authorization: token
  {api_key}:{access_token}`` and ``X-Kite-Version: 3``.
- Every response is a JSON envelope: ``{"status": "success", "data":
  ...}`` or ``{"status": "error", "message": ..., "error_type": ...}``.

## The client_order_id bridge

This system's :class:`broker.base.Broker` interface promises callers can
query an order by the *client-generated* ``client_order_id`` (see
``broker/base.py``'s module docstring). Kite's own API has no such
concept -- ``POST /orders/:variety`` returns only Kite's own
broker-assigned ``order_id``, and every other order endpoint is
addressed by that same broker-assigned ID. This adapter bridges the two
by keeping an in-memory ``client_order_id -> kite_order_id`` map,
populated at ``place_order`` time, and also tags every placed order with
a truncated ``client_order_id`` (Kite's ``tag`` field, max 20
alphanumeric characters) as a human-visible breadcrumb in the Kite order
book UI -- not the actual lookup mechanism, which is the in-memory map.

**This mapping is not persisted.** An order this process did not place
itself (a previous run, or a manual order placed outside this system)
has no known ``client_order_id``; :meth:`get_order`/:meth:`get_trades`
fall back to reporting Kite's own ``order_id`` as the identifier in that
case. Closing this gap with real reconciliation against the broker's own
state at startup is Phase 11b's job, not this adapter's.

## Live trading is disabled by default

``enable_live_trading`` defaults to ``False``. Every order-placing method
(``place_order``, ``modify_order``, ``cancel_order``, ``close_position``,
``close_all_positions``) checks this flag before making any network call
and raises :class:`broker.errors.BrokerCapabilityError` if it is not
``True`` -- a second, adapter-level gate on top of
``broker/factory.py``'s own refusal to construct a live-capable broker
unless ``execution.mode`` is explicitly ``"live"``. Read-only calls
(account, positions, orders, quotes) are not gated -- authenticating and
observing a real account does not risk placing a real order.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import uuid
import zoneinfo
from collections.abc import Callable, Mapping
from typing import Protocol

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
from broker.errors import (
    BrokerAuthenticationError,
    BrokerCapabilityError,
    BrokerError,
    BrokerRateLimitError,
    BrokerRequestError,
    BrokerSessionExpiredError,
)
from broker.zerodha.kite_mappings import (
    KITE_VALIDITIES,
    SUPPORTED_EXCHANGES,
    SUPPORTED_ORDER_TYPES,
    SUPPORTED_PRODUCTS,
    SUPPORTED_VARIETIES,
    join_instrument_id,
    order_state_from_kite,
    side_from_kite,
    side_to_kite,
    split_instrument_id,
)
from broker.zerodha.kite_transport import HttpResponse, HttpTransport, UrllibHttpTransport
from execution.order_manager import OrderState

_BASE_URL = "https://api.kite.trade"
_LOGIN_URL = "https://kite.zerodha.com/connect/login"
_KITE_VERSION = "3"
_IST = zoneinfo.ZoneInfo("Asia/Kolkata")

_OPEN_STATES: frozenset[OrderState] = frozenset(
    {
        OrderState.SUBMITTED,
        OrderState.OPEN,
        OrderState.PARTIALLY_FILLED,
        OrderState.CANCEL_REQUESTED,
    }
)


def _parse_kite_timestamp(value: str) -> dt.datetime:
    """Kite timestamps are ``"yyyy-mm-dd hh:mm:ss"`` in IST
    (docs/connect/v3/response-structure/, verified)."""
    naive = dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return naive.replace(tzinfo=_IST)


def _as_dict(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise BrokerError(f"expected a JSON object for {context}, got {type(value).__name__}")
    return value


def _as_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise BrokerError(f"expected a JSON array for {context}, got {type(value).__name__}")
    return value


def _as_str_set(value: object) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    return frozenset(str(item) for item in value)


def _as_int(value: object) -> int:
    if not isinstance(value, int | float | str):
        raise BrokerError(f"expected a number for an int field, got {type(value).__name__}")
    return int(value)


def _as_float(value: object) -> float:
    if not isinstance(value, int | float | str):
        raise BrokerError(f"expected a number for a float field, got {type(value).__name__}")
    return float(value)


def _as_optional_float(value: object) -> float | None:
    if value is None or value == 0:
        return None
    return _as_float(value)


class KiteTickerProtocol(Protocol):
    """Structural interface for the WebSocket ticker this adapter delegates
    streaming to -- see ``kite_ticker.py`` for the real implementation.
    Typed as a ``Protocol`` (not a hard import) so ``kite_broker.py`` and
    ``kite_ticker.py`` do not need to import each other."""

    def subscribe(
        self,
        api_key: str,
        access_token: str,
        instrument_ids: list[str],
        on_tick: Callable[[BrokerQuote], None],
    ) -> Callable[[], None]: ...


class KiteBroker(Broker):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        transport: HttpTransport | None = None,
        enable_live_trading: bool = False,
        clock: Callable[[], dt.datetime] | None = None,
        ticker: KiteTickerProtocol | None = None,
    ) -> None:
        if not api_key or not api_secret:
            raise BrokerAuthenticationError("api_key and api_secret must both be non-empty")
        self.api_key = api_key
        self._api_secret = api_secret
        self._transport: HttpTransport = transport or UrllibHttpTransport()
        self._enable_live_trading = enable_live_trading
        self._clock: Callable[[], dt.datetime] = clock or (lambda: dt.datetime.now(dt.UTC))
        self._ticker = ticker

        self._access_token: str | None = None
        self._user_id: str | None = None
        self._login_time: dt.datetime | None = None
        self._enabled_exchanges: frozenset[str] = SUPPORTED_EXCHANGES
        self._enabled_products: frozenset[str] = SUPPORTED_PRODUCTS
        self._enabled_order_types: frozenset[str] = SUPPORTED_ORDER_TYPES

        self._client_to_kite_order_id: dict[str, str] = {}
        self._kite_to_client_order_id: dict[str, str] = {}
        self._order_variety: dict[str, str] = {}

    @property
    def live_trading_enabled(self) -> bool:
        return self._enable_live_trading

    def login_url(self) -> str:
        """The browser URL a human must complete login at; Kite then
        redirects back with a ``request_token`` for :meth:`authenticate`.
        """
        return f"{_LOGIN_URL}?v=3&api_key={self.api_key}"

    # -- Broker interface --------------------------------------------------

    def capabilities(self) -> BrokerCapabilities:
        return BrokerCapabilities(
            broker_name="zerodha",
            supports_order_modification=True,
            supports_market_data_streaming=self._ticker is not None,
            supported_exchanges=self._enabled_exchanges,
            supported_products=self._enabled_products,
            supported_order_types=self._enabled_order_types,
            supported_varieties=SUPPORTED_VARIETIES,
        )

    def authenticate(self, credentials: Mapping[str, str]) -> None:
        """``credentials`` must carry ``{"request_token": "..."}`` -- the
        one-time token Kite's login redirect returns after a human
        completes login at :meth:`login_url`. Exchanges it for a session
        access token via ``POST /session/token``.
        """
        request_token = credentials.get("request_token")
        if not request_token:
            raise BrokerAuthenticationError(
                "KiteBroker.authenticate requires credentials={'request_token': '...'}"
            )
        checksum = hashlib.sha256(
            f"{self.api_key}{request_token}{self._api_secret}".encode()
        ).hexdigest()
        session_params: dict[str, object] = {
            "api_key": self.api_key,
            "request_token": request_token,
            "checksum": checksum,
        }
        payload = _as_dict(
            self._request("POST", "/session/token", params=session_params, authenticated=False),
            "session/token",
        )
        try:
            self._access_token = str(payload["access_token"])
            self._user_id = str(payload["user_id"])
            self._login_time = _parse_kite_timestamp(str(payload["login_time"]))
        except KeyError as exc:
            raise BrokerAuthenticationError(
                f"session/token response is missing expected field: {exc}"
            ) from exc

        try:
            profile = _as_dict(self._request("GET", "/user/profile"), "user/profile")
            self._enabled_exchanges = SUPPORTED_EXCHANGES & _as_str_set(profile.get("exchanges"))
            self._enabled_products = SUPPORTED_PRODUCTS & _as_str_set(profile.get("products"))
            enabled_order_types = _as_str_set(profile.get("order_types"))
            self._enabled_order_types = SUPPORTED_ORDER_TYPES & enabled_order_types
        except BrokerError:
            # The session itself is valid even if this follow-up profile
            # fetch fails; capabilities() simply keeps this adapter's own
            # default supported set until a profile fetch succeeds.
            pass

    def get_account(self) -> BrokerAccount:
        payload = _as_dict(self._request("GET", "/user/margins/equity"), "user/margins")
        available = _as_dict(payload.get("available") or {}, "margins.available")
        net = _as_float(payload.get("net", 0.0))
        cash = _as_float(available.get("cash", 0.0))
        return BrokerAccount(
            account_id=self._user_id or "",
            equity=net,
            cash=cash,
            buying_power=net,
            as_of=self._clock(),
        )

    def get_positions(self) -> list[BrokerPosition]:
        payload = _as_dict(self._request("GET", "/portfolio/positions"), "positions")
        rows = _as_list(payload.get("net") or [], "positions.net")
        positions = []
        for raw in rows:
            row = _as_dict(raw, "position row")
            quantity = _as_int(row["quantity"])
            if quantity == 0:
                continue
            instrument_id = join_instrument_id(str(row["exchange"]), str(row["tradingsymbol"]))
            positions.append(
                BrokerPosition(
                    instrument_id=instrument_id,
                    quantity=quantity,
                    avg_price=_as_float(row["average_price"]),
                    product=str(row["product"]),
                )
            )
        return positions

    def get_open_orders(self) -> list[BrokerOrder]:
        rows = _as_list(self._request("GET", "/orders"), "orders")
        orders = [self._order_from_kite(_as_dict(row, "order row")) for row in rows]
        return [order for order in orders if OrderState(order.status) in _OPEN_STATES]

    def get_order(self, order_id: str) -> BrokerOrder:
        kite_order_id = self._client_to_kite_order_id.get(order_id, order_id)
        rows = _as_list(self._request("GET", f"/orders/{kite_order_id}"), "order history")
        if not rows:
            raise BrokerRequestError(f"no such order: {order_id}")
        return self._order_from_kite(_as_dict(rows[-1], "order history row"))

    def get_trades(self, order_id: str | None = None) -> list[BrokerFill]:
        if order_id is None:
            rows = _as_list(self._request("GET", "/trades"), "trades")
        else:
            kite_order_id = self._client_to_kite_order_id.get(order_id, order_id)
            rows = _as_list(self._request("GET", f"/orders/{kite_order_id}/trades"), "order trades")
        return [self._fill_from_kite(_as_dict(row, "trade row")) for row in rows]

    def get_quotes(self, instrument_ids: list[str]) -> list[BrokerQuote]:
        if not instrument_ids:
            return []
        payload = _as_dict(
            self._request("GET", "/quote", params={"i": instrument_ids}), "quote"
        )
        now = self._clock()
        quotes = []
        for instrument_id in instrument_ids:
            raw = payload.get(instrument_id)
            if raw is None:
                continue
            row = _as_dict(raw, f"quote for {instrument_id}")
            depth = _as_dict(row.get("depth") or {}, "quote.depth")
            buy_levels = _as_list(depth.get("buy") or [], "quote.depth.buy")
            sell_levels = _as_list(depth.get("sell") or [], "quote.depth.sell")
            last_price = _as_float(row["last_price"])
            bid = (
                _as_float(_as_dict(buy_levels[0], "depth level")["price"])
                if buy_levels
                else last_price
            )
            ask = (
                _as_float(_as_dict(sell_levels[0], "depth level")["price"])
                if sell_levels
                else last_price
            )
            timestamp = row.get("timestamp")
            as_of = _parse_kite_timestamp(str(timestamp)) if timestamp else now
            quotes.append(
                BrokerQuote(
                    instrument_id=instrument_id,
                    bid=bid,
                    ask=ask,
                    last_price=last_price,
                    as_of=as_of,
                )
            )
        return quotes

    def subscribe_market_data(
        self, instrument_ids: list[str], on_tick: Callable[[BrokerQuote], None]
    ) -> Callable[[], None]:
        if self._ticker is None:
            raise BrokerCapabilityError(
                "no WebSocket transport configured -- construct KiteBroker with a "
                "ticker (broker.zerodha.kite_ticker.KiteTicker) to enable live "
                "market-data streaming"
            )
        if self._access_token is None:
            raise BrokerAuthenticationError("not authenticated -- call authenticate() first")
        return self._ticker.subscribe(self.api_key, self._access_token, instrument_ids, on_tick)

    def place_order(self, order: BrokerOrder) -> BrokerOrder:
        self._require_live_trading_enabled("place_order")
        if order.client_order_id in self._client_to_kite_order_id:
            # Idempotency: Kite has no client-order-ID deduplication of its
            # own, so this adapter is the one place that guarantees a
            # resubmitted request (same client_order_id) never reaches the
            # exchange a second time.
            return self.get_order(order.client_order_id)

        exchange, tradingsymbol = split_instrument_id(order.instrument_id)
        if order.order_type not in SUPPORTED_ORDER_TYPES:
            raise BrokerCapabilityError(
                f"order_type {order.order_type!r} not supported by this adapter "
                f"(only {sorted(SUPPORTED_ORDER_TYPES)})"
            )
        if order.variety not in SUPPORTED_VARIETIES:
            raise BrokerCapabilityError(
                f"variety {order.variety!r} not supported by this adapter "
                f"(only {sorted(SUPPORTED_VARIETIES)})"
            )
        if order.product not in SUPPORTED_PRODUCTS:
            raise BrokerCapabilityError(
                f"product {order.product!r} not supported by this adapter "
                f"(only {sorted(SUPPORTED_PRODUCTS)})"
            )
        if order.validity not in KITE_VALIDITIES:
            raise BrokerCapabilityError(
                f"validity {order.validity!r} is not a recognized Kite validity "
                f"(one of {sorted(KITE_VALIDITIES)})"
            )
        if order.limit_price is None or order.limit_price <= 0:
            raise BrokerRequestError("a LIMIT order requires a positive limit_price")

        tag = order.tag if order.tag else order.client_order_id.replace("-", "")[:20]
        params: dict[str, object] = {
            "tradingsymbol": tradingsymbol,
            "exchange": exchange,
            "transaction_type": side_to_kite(TradeSide(order.side)),
            "order_type": order.order_type,
            "quantity": order.quantity,
            "product": order.product,
            "price": order.limit_price,
            "validity": order.validity,
            "tag": tag[:20],
        }
        payload = _as_dict(
            self._request("POST", f"/orders/{order.variety}", params=params), "place order"
        )
        kite_order_id = str(payload["order_id"])
        self._client_to_kite_order_id[order.client_order_id] = kite_order_id
        self._kite_to_client_order_id[kite_order_id] = order.client_order_id
        self._order_variety[kite_order_id] = order.variety
        return self.get_order(order.client_order_id)

    def modify_order(self, order_id: str, changes: dict[str, object]) -> BrokerOrder:
        self._require_live_trading_enabled("modify_order")
        kite_order_id = self._client_to_kite_order_id.get(order_id, order_id)
        variety = self._order_variety.get(kite_order_id, "regular")

        params: dict[str, object] = {}
        field_map = {
            "limit_price": "price",
            "quantity": "quantity",
            "trigger_price": "trigger_price",
            "disclosed_quantity": "disclosed_quantity",
            "order_type": "order_type",
            "validity": "validity",
        }
        for generic_key, kite_key in field_map.items():
            if generic_key in changes:
                params[kite_key] = changes[generic_key]

        self._request("PUT", f"/orders/{variety}/{kite_order_id}", params=params)
        return self.get_order(order_id)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        self._require_live_trading_enabled("cancel_order")
        kite_order_id = self._client_to_kite_order_id.get(order_id, order_id)
        variety = self._order_variety.get(kite_order_id, "regular")
        self._request("DELETE", f"/orders/{variety}/{kite_order_id}")
        return self.get_order(order_id)

    def close_position(self, instrument_id: str) -> BrokerOrder:
        self._require_live_trading_enabled("close_position")
        position = next(
            (p for p in self.get_positions() if p.instrument_id == instrument_id), None
        )
        if position is None or position.quantity <= 0:
            raise BrokerRequestError(f"no open long position in {instrument_id} to close")
        quotes = self.get_quotes([instrument_id])
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
        return [self.close_position(position.instrument_id) for position in self.get_positions()]

    def health_check(self) -> HealthStatus:
        now = self._clock()
        if self._access_token is None:
            return HealthStatus(
                healthy=False,
                detail="not authenticated",
                checked_at=now,
                session_active=False,
                login_time=None,
            )
        try:
            self._request("GET", "/user/profile")
        except BrokerSessionExpiredError as exc:
            return HealthStatus(
                healthy=False,
                detail=str(exc),
                checked_at=now,
                session_active=False,
                login_time=self._login_time,
            )
        except BrokerError as exc:
            return HealthStatus(
                healthy=False,
                detail=str(exc),
                checked_at=now,
                session_active=True,
                login_time=self._login_time,
            )
        return HealthStatus(
            healthy=True,
            detail="session active",
            checked_at=now,
            session_active=True,
            login_time=self._login_time,
        )

    # -- internals -----------------------------------------------------

    def _require_live_trading_enabled(self, operation: str) -> None:
        if not self._enable_live_trading:
            raise BrokerCapabilityError(
                f"live trading is disabled on this KiteBroker instance; {operation} refused. "
                "Construct KiteBroker with enable_live_trading=True to place real orders -- "
                "this system's default mode is paper trading and must stay that way unless "
                "explicitly overridden."
            )

    def _order_from_kite(self, row: dict[str, object]) -> BrokerOrder:
        kite_order_id = str(row["order_id"])
        quantity = _as_int(row["quantity"])
        filled_quantity = _as_int(row.get("filled_quantity", 0))
        kite_status = str(row["status"])
        state = order_state_from_kite(kite_status, filled_quantity, quantity)
        client_order_id = self._kite_to_client_order_id.get(kite_order_id, kite_order_id)
        status_message = row.get("status_message")
        return BrokerOrder(
            client_order_id=client_order_id,
            broker_order_id=kite_order_id,
            instrument_id=join_instrument_id(str(row["exchange"]), str(row["tradingsymbol"])),
            side=side_from_kite(str(row["transaction_type"])).value,
            quantity=quantity,
            order_type=str(row["order_type"]),
            limit_price=_as_optional_float(row.get("price")),
            status=state.value,
            filled_quantity=filled_quantity,
            avg_fill_price=_as_optional_float(row.get("average_price")),
            reject_reason=(
                str(status_message) if state is OrderState.REJECTED and status_message else None
            ),
            product=str(row["product"]),
            variety=str(row["variety"]),
        )

    def _fill_from_kite(self, row: dict[str, object]) -> BrokerFill:
        kite_order_id = str(row["order_id"])
        client_order_id = self._kite_to_client_order_id.get(kite_order_id, kite_order_id)
        return BrokerFill(
            trade_id=str(row["trade_id"]),
            order_id=client_order_id,
            instrument_id=join_instrument_id(str(row["exchange"]), str(row["tradingsymbol"])),
            side=side_from_kite(str(row["transaction_type"])).value,
            quantity=_as_int(row["quantity"]),
            price=_as_float(row["average_price"]),
            product=str(row["product"]),
            as_of=_parse_kite_timestamp(str(row["fill_timestamp"])),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"X-Kite-Version": _KITE_VERSION}
        if self._access_token is not None:
            headers["Authorization"] = f"token {self.api_key}:{self._access_token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
        authenticated: bool = True,
    ) -> object:
        if authenticated and self._access_token is None:
            raise BrokerAuthenticationError(
                "not authenticated -- call authenticate() with a request_token first"
            )
        response = self._transport.request(
            method, f"{_BASE_URL}{path}", headers=self._headers(), params=params
        )
        return self._parse_response(response)

    def _parse_response(self, response: HttpResponse) -> object:
        try:
            envelope: object = response.json()
        except (ValueError, UnicodeDecodeError):
            envelope = None

        if response.status_code == 429:
            raise BrokerRateLimitError(
                self._error_message(envelope) or "rate limit exceeded",
                error_type=self._error_type(envelope),
            )
        if response.status_code == 403:
            self._access_token = None
            raise BrokerSessionExpiredError(
                self._error_message(envelope) or "session expired or invalid"
            )
        if envelope is None:
            raise BrokerError(f"Kite returned a non-JSON response (HTTP {response.status_code})")
        if not isinstance(envelope, dict):
            raise BrokerError(
                f"expected a JSON object response envelope, got {type(envelope).__name__}"
            )

        if envelope.get("status") == "success":
            return envelope.get("data")

        message = self._error_message(envelope) or "unknown error"
        error_type = self._error_type(envelope)
        if error_type == "TokenException":
            self._access_token = None
            raise BrokerSessionExpiredError(message)
        raise BrokerRequestError(message, error_type=error_type)

    @staticmethod
    def _error_message(envelope: object) -> str | None:
        if isinstance(envelope, dict):
            value = envelope.get("message")
            return str(value) if value is not None else None
        return None

    @staticmethod
    def _error_type(envelope: object) -> str | None:
        if isinstance(envelope, dict):
            value = envelope.get("error_type")
            return str(value) if value is not None else None
        return None
