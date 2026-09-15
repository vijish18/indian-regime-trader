"""Unit and integration tests for the Zerodha Kite Connect adapter
(``broker/zerodha/kite_broker.py``, Phase 15) -- entirely against a
scripted, in-memory fake HTTP transport. No test here makes a real
network call; live trading stays disabled by default throughout.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Callable

import pytest

from broker.base import BrokerOrder, BrokerQuote
from broker.errors import (
    BrokerAuthenticationError,
    BrokerCapabilityError,
    BrokerError,
    BrokerRateLimitError,
    BrokerRequestError,
    BrokerSessionExpiredError,
)
from broker.zerodha.kite_broker import KiteBroker
from broker.zerodha.kite_transport import HttpResponse
from execution.order_manager import OrderManager, OrderState


def _ok(data: object) -> HttpResponse:
    return HttpResponse(
        status_code=200, body=json.dumps({"status": "success", "data": data}).encode()
    )


def _err(status_code: int, message: str, error_type: str) -> HttpResponse:
    body = json.dumps({"status": "error", "message": message, "error_type": error_type})
    return HttpResponse(status_code=status_code, body=body.encode())


class _FakeHttpTransport:
    """Routes by ``(method, path)`` (query string stripped). Each call to
    a route consumes the next scripted response for it, in the order
    :meth:`script` was called; once a route's queue is exhausted, the
    last response actually returned for it is reused indefinitely (so a
    route scripted only once can still serve any number of calls).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object] | None, dict[str, str]]] = []
        self._responses: dict[tuple[str, str], list[HttpResponse]] = {}
        self._last_response: dict[tuple[str, str], HttpResponse] = {}

    def script(self, method: str, path: str, response: HttpResponse) -> None:
        self._responses.setdefault((method, path), []).append(response)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, object] | None = None,
    ) -> HttpResponse:
        path = url.removeprefix("https://api.kite.trade").split("?", 1)[0]
        self.calls.append((method, path, params, dict(headers)))
        key = (method, path)
        queue = self._responses.get(key)
        if queue:
            response = queue.pop(0)
            self._last_response[key] = response
            return response
        if key in self._last_response:
            return self._last_response[key]
        raise AssertionError(f"no scripted response for {method} {path}")


def _broker(transport: _FakeHttpTransport, **kwargs: object) -> KiteBroker:
    return KiteBroker(api_key="testkey", api_secret="testsecret", transport=transport, **kwargs)  # type: ignore[arg-type]


def _authenticate(
    broker: KiteBroker,
    transport: _FakeHttpTransport,
    *,
    request_token: str = "req-token-1",
    user_id: str = "AB1234",
    exchanges: list[str] | None = None,
    products: list[str] | None = None,
    order_types: list[str] | None = None,
) -> None:
    transport.script(
        "POST",
        "/session/token",
        _ok(
            {
                "access_token": "acc-token-1",
                "user_id": user_id,
                "user_name": "Test User",
                "login_time": "2024-06-03 09:00:00",
            }
        ),
    )
    transport.script(
        "GET",
        "/user/profile",
        _ok(
            {
                "user_id": user_id,
                "exchanges": exchanges if exchanges is not None else ["NSE", "BSE"],
                "products": products if products is not None else ["CNC"],
                "order_types": order_types if order_types is not None else ["LIMIT"],
            }
        ),
    )
    broker.authenticate({"request_token": request_token})


# --------------------------------------------------------------------------
# Construction and login_url
# --------------------------------------------------------------------------


def test_construction_rejects_empty_credentials() -> None:
    with pytest.raises(BrokerAuthenticationError):
        KiteBroker(api_key="", api_secret="secret")
    with pytest.raises(BrokerAuthenticationError):
        KiteBroker(api_key="key", api_secret="")


def test_login_url_carries_the_api_key() -> None:
    broker = _broker(_FakeHttpTransport())
    assert broker.login_url() == "https://kite.zerodha.com/connect/login?v=3&api_key=testkey"


def test_live_trading_disabled_by_default() -> None:
    broker = _broker(_FakeHttpTransport())
    assert broker.live_trading_enabled is False


# --------------------------------------------------------------------------
# authenticate()
# --------------------------------------------------------------------------


def test_authenticate_requires_a_request_token() -> None:
    broker = _broker(_FakeHttpTransport())
    with pytest.raises(BrokerAuthenticationError, match="request_token"):
        broker.authenticate({})


def test_authenticate_sends_the_documented_checksum() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport, request_token="the-request-token")

    method, path, params, _headers = transport.calls[0]
    assert (method, path) == ("POST", "/session/token")
    assert params is not None
    expected_checksum = hashlib.sha256(
        b"testkey" + b"the-request-token" + b"testsecret"
    ).hexdigest()
    assert params["checksum"] == expected_checksum
    assert params["api_key"] == "testkey"
    assert params["request_token"] == "the-request-token"


def test_authenticate_populates_session_state_and_unlocks_authenticated_calls() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport, user_id="ZZ9999")

    transport.script(
        "GET", "/user/margins/equity", _ok({"net": 100000.0, "available": {"cash": 90000.0}})
    )
    account = broker.get_account()
    assert account.account_id == "ZZ9999"

    _method, _path, _params, headers = transport.calls[-1]
    assert headers["Authorization"] == "token testkey:acc-token-1"
    assert headers["X-Kite-Version"] == "3"


def test_authenticate_raises_if_response_is_missing_a_required_field() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    transport.script("POST", "/session/token", _ok({"user_id": "AB1234"}))  # no access_token
    with pytest.raises(BrokerAuthenticationError, match="access_token"):
        broker.authenticate({"request_token": "req-token-1"})


def test_authenticate_survives_a_failed_profile_follow_up_call() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    transport.script(
        "POST",
        "/session/token",
        _ok(
            {
                "access_token": "acc-token-1",
                "user_id": "AB1234",
                "login_time": "2024-06-03 09:00:00",
            }
        ),
    )
    transport.script("GET", "/user/profile", _err(500, "internal error", "GeneralException"))

    broker.authenticate({"request_token": "req-token-1"})  # must not raise
    caps = broker.capabilities()
    assert caps.supported_exchanges  # fell back to this adapter's own default set


# --------------------------------------------------------------------------
# capabilities()
# --------------------------------------------------------------------------


def test_capabilities_reflect_the_authenticated_accounts_profile() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(
        broker,
        transport,
        exchanges=["NSE", "BSE", "NFO"],
        products=["CNC", "MIS"],
        order_types=["MARKET", "LIMIT"],
    )
    caps = broker.capabilities()
    assert caps.broker_name == "zerodha"
    assert caps.supported_exchanges == {"NSE", "BSE"}
    assert caps.supported_products == {"CNC"}
    assert caps.supported_order_types == {"LIMIT"}
    assert caps.supported_varieties == {"regular"}
    assert caps.supports_order_modification is True
    assert caps.supports_market_data_streaming is False


def test_capabilities_before_authentication_uses_the_adapters_own_defaults() -> None:
    broker = _broker(_FakeHttpTransport())
    caps = broker.capabilities()
    assert caps.supported_exchanges == {"NSE", "BSE"}
    assert caps.supported_products == {"CNC"}


# --------------------------------------------------------------------------
# Unauthenticated calls fail closed
# --------------------------------------------------------------------------


def test_an_authenticated_endpoint_before_login_raises() -> None:
    broker = _broker(_FakeHttpTransport())
    with pytest.raises(BrokerAuthenticationError, match="not authenticated"):
        broker.get_account()


# --------------------------------------------------------------------------
# get_account / get_positions
# --------------------------------------------------------------------------


def test_get_account_maps_margins_response() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script(
        "GET", "/user/margins/equity", _ok({"net": 123456.78, "available": {"cash": 100000.0}})
    )
    account = broker.get_account()
    assert account.equity == pytest.approx(123456.78)
    assert account.cash == pytest.approx(100000.0)
    assert account.buying_power == pytest.approx(123456.78)


def test_get_positions_maps_net_positions_and_drops_zero_quantity_rows() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script(
        "GET",
        "/portfolio/positions",
        _ok(
            {
                "net": [
                    {
                        "tradingsymbol": "INFY",
                        "exchange": "NSE",
                        "quantity": 10,
                        "average_price": 1500.0,
                        "product": "CNC",
                    },
                    {
                        "tradingsymbol": "TCS",
                        "exchange": "NSE",
                        "quantity": 0,
                        "average_price": 0.0,
                        "product": "CNC",
                    },
                ]
            }
        ),
    )
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].instrument_id == "NSE:INFY"
    assert positions[0].quantity == 10
    assert positions[0].avg_price == pytest.approx(1500.0)
    assert positions[0].product == "CNC"


# --------------------------------------------------------------------------
# get_open_orders / get_order
# --------------------------------------------------------------------------


def _order_row(
    order_id: str,
    status: str,
    *,
    quantity: int = 10,
    filled_quantity: int = 0,
    tradingsymbol: str = "INFY",
    exchange: str = "NSE",
    transaction_type: str = "BUY",
    order_type: str = "LIMIT",
    price: float = 1500.0,
    average_price: float = 0.0,
    product: str = "CNC",
    variety: str = "regular",
    status_message: str | None = None,
) -> dict[str, object]:
    return {
        "order_id": order_id,
        "tradingsymbol": tradingsymbol,
        "exchange": exchange,
        "quantity": quantity,
        "filled_quantity": filled_quantity,
        "status": status,
        "transaction_type": transaction_type,
        "order_type": order_type,
        "price": price,
        "average_price": average_price,
        "product": product,
        "variety": variety,
        "status_message": status_message,
    }


def test_get_open_orders_filters_to_non_terminal_statuses() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script(
        "GET",
        "/orders",
        _ok(
            [
                _order_row("K1", "COMPLETE", filled_quantity=10),
                _order_row("K2", "OPEN"),
                _order_row("K3", "PUT ORDER REQ RECEIVED"),
                _order_row("K4", "CANCELLED"),
                _order_row("K5", "REJECTED"),
            ]
        ),
    )
    open_orders = broker.get_open_orders()
    statuses = {order.broker_order_id: OrderState(order.status) for order in open_orders}
    assert statuses == {"K2": OrderState.OPEN, "K3": OrderState.SUBMITTED}


def test_get_order_falls_back_to_treating_an_unmapped_id_as_a_kite_order_id() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script("GET", "/orders/K999", _ok([_order_row("K999", "OPEN")]))
    order = broker.get_order("K999")  # never placed through this instance
    assert order.broker_order_id == "K999"
    assert order.client_order_id == "K999"  # best-effort fallback, documented


def test_get_order_uses_the_latest_history_entry() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script(
        "GET",
        "/orders/K1",
        _ok(
            [
                _order_row("K1", "OPEN", filled_quantity=0),
                _order_row("K1", "OPEN", filled_quantity=4),
                _order_row("K1", "COMPLETE", filled_quantity=10),
            ]
        ),
    )
    order = broker.get_order("K1")
    assert order.status == OrderState.FILLED.value
    assert order.filled_quantity == 10


def test_get_order_raises_when_history_is_empty() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script("GET", "/orders/K1", _ok([]))
    with pytest.raises(BrokerRequestError, match="no such order"):
        broker.get_order("K1")


def test_order_rejection_carries_the_status_message_as_reject_reason() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script(
        "GET",
        "/orders/K1",
        _ok([_order_row("K1", "REJECTED", status_message="RMS:Insufficient funds")]),
    )
    order = broker.get_order("K1")
    assert order.status == OrderState.REJECTED.value
    assert order.reject_reason == "RMS:Insufficient funds"


# --------------------------------------------------------------------------
# get_trades
# --------------------------------------------------------------------------


def _trade_row(
    trade_id: str,
    order_id: str,
    *,
    tradingsymbol: str = "INFY",
    exchange: str = "NSE",
    transaction_type: str = "BUY",
    quantity: int = 10,
    average_price: float = 1500.0,
    product: str = "CNC",
) -> dict[str, object]:
    return {
        "trade_id": trade_id,
        "order_id": order_id,
        "tradingsymbol": tradingsymbol,
        "exchange": exchange,
        "transaction_type": transaction_type,
        "quantity": quantity,
        "average_price": average_price,
        "product": product,
        "fill_timestamp": "2024-06-03 09:16:00",
    }


def test_get_trades_without_order_id_hits_the_all_trades_endpoint() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script("GET", "/trades", _ok([_trade_row("T1", "K1")]))
    fills = broker.get_trades()
    assert len(fills) == 1
    assert fills[0].trade_id == "T1"
    assert fills[0].order_id == "K1"  # no mapping known -> falls back to kite id


def test_get_trades_with_order_id_hits_the_per_order_endpoint() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script("GET", "/orders/K1/trades", _ok([_trade_row("T1", "K1")]))
    fills = broker.get_trades("K1")
    assert [fill.trade_id for fill in fills] == ["T1"]


# --------------------------------------------------------------------------
# get_quotes
# --------------------------------------------------------------------------


def test_get_quotes_uses_top_of_book_depth_for_bid_and_ask() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script(
        "GET",
        "/quote",
        _ok(
            {
                "NSE:INFY": {
                    "last_price": 1500.0,
                    "timestamp": "2024-06-03 09:20:00",
                    "depth": {
                        "buy": [{"price": 1499.5, "quantity": 100, "orders": 2}],
                        "sell": [{"price": 1500.5, "quantity": 80, "orders": 1}],
                    },
                }
            }
        ),
    )
    quotes = broker.get_quotes(["NSE:INFY"])
    assert len(quotes) == 1
    assert quotes[0].bid == pytest.approx(1499.5)
    assert quotes[0].ask == pytest.approx(1500.5)
    assert quotes[0].last_price == pytest.approx(1500.0)


def test_get_quotes_falls_back_to_last_price_with_no_depth() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script(
        "GET",
        "/quote",
        _ok({"NSE:INFY": {"last_price": 1500.0, "depth": {"buy": [], "sell": []}}}),
    )
    quotes = broker.get_quotes(["NSE:INFY"])
    assert quotes[0].bid == pytest.approx(1500.0)
    assert quotes[0].ask == pytest.approx(1500.0)


def test_get_quotes_skips_an_instrument_missing_from_the_response() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script("GET", "/quote", _ok({}))  # instrument absent, per Kite's own convention
    assert broker.get_quotes(["NSE:INFY"]) == []


def test_get_quotes_with_no_instruments_makes_no_request() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    assert broker.get_quotes([]) == []
    assert transport.calls[-1][1] != "/quote"


def test_get_quotes_repeats_the_i_query_parameter_per_instrument() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script("GET", "/quote", _ok({}))
    broker.get_quotes(["NSE:INFY", "NSE:TCS"])
    _method, _path, params, _headers = transport.calls[-1]
    assert params == {"i": ["NSE:INFY", "NSE:TCS"]}


# --------------------------------------------------------------------------
# subscribe_market_data
# --------------------------------------------------------------------------


class _StubTicker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []

    def subscribe(
        self,
        api_key: str,
        access_token: str,
        instrument_ids: list[str],
        on_tick: Callable[[BrokerQuote], None],
    ) -> Callable[[], None]:
        self.calls.append((api_key, access_token, instrument_ids))
        return lambda: None


def test_subscribe_market_data_without_a_ticker_raises() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    with pytest.raises(BrokerCapabilityError, match="no WebSocket transport"):
        broker.subscribe_market_data(["NSE:INFY"], lambda quote: None)


def test_subscribe_market_data_before_authentication_raises() -> None:
    ticker = _StubTicker()
    broker = _broker(_FakeHttpTransport(), ticker=ticker)
    with pytest.raises(BrokerAuthenticationError):
        broker.subscribe_market_data(["NSE:INFY"], lambda quote: None)


def test_subscribe_market_data_delegates_to_the_configured_ticker() -> None:
    transport = _FakeHttpTransport()
    ticker = _StubTicker()
    broker = _broker(transport, ticker=ticker)
    _authenticate(broker, transport)

    broker.subscribe_market_data(["NSE:INFY"], lambda quote: None)
    assert ticker.calls == [("testkey", "acc-token-1", ["NSE:INFY"])]
    assert broker.capabilities().supports_market_data_streaming is True


# --------------------------------------------------------------------------
# place_order / modify_order / cancel_order -- gated by enable_live_trading
# --------------------------------------------------------------------------


def _new_order(
    client_order_id: str = "c-1",
    *,
    side: str = "buy",
    quantity: int = 10,
    limit_price: float = 1500.0,
) -> BrokerOrder:
    return BrokerOrder(
        client_order_id=client_order_id,
        broker_order_id=None,
        instrument_id="NSE:INFY",
        side=side,
        quantity=quantity,
        order_type="LIMIT",
        limit_price=limit_price,
        status=OrderState.CREATED.value,
    )


def test_place_order_refused_while_live_trading_disabled() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    with pytest.raises(BrokerCapabilityError, match="live trading is disabled"):
        broker.place_order(_new_order())
    assert transport.calls[-1][1] == "/user/profile"  # only the authenticate() calls happened


def test_place_order_succeeds_when_live_trading_is_enabled() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport, enable_live_trading=True)
    _authenticate(broker, transport)
    transport.script("POST", "/orders/regular", _ok({"order_id": "K1"}))
    transport.script("GET", "/orders/K1", _ok([_order_row("K1", "OPEN")]))

    order = broker.place_order(_new_order())
    assert order.broker_order_id == "K1"
    assert order.status == OrderState.OPEN.value

    _method, _path, params, _headers = transport.calls[-2]
    assert params is not None
    assert params["tradingsymbol"] == "INFY"
    assert params["exchange"] == "NSE"
    assert params["transaction_type"] == "BUY"
    assert params["product"] == "CNC"
    assert params["validity"] == "DAY"


def test_place_order_tags_the_order_with_a_truncated_client_order_id() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport, enable_live_trading=True)
    _authenticate(broker, transport)
    transport.script("POST", "/orders/regular", _ok({"order_id": "K1"}))
    transport.script("GET", "/orders/K1", _ok([_order_row("K1", "OPEN")]))

    long_id = "12345678-abcd-4321-aaaa-000000000000"
    broker.place_order(_new_order(client_order_id=long_id))
    _method, _path, params, _headers = transport.calls[-2]
    assert params is not None
    assert params["tag"] == long_id.replace("-", "")[:20]
    assert len(str(params["tag"])) <= 20


def test_place_order_is_idempotent_by_client_order_id() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport, enable_live_trading=True)
    _authenticate(broker, transport)
    transport.script("POST", "/orders/regular", _ok({"order_id": "K1"}))
    transport.script("GET", "/orders/K1", _ok([_order_row("K1", "OPEN")]))

    first = broker.place_order(_new_order())
    second = broker.place_order(_new_order())
    assert first == second

    place_calls = [call for call in transport.calls if call[:2] == ("POST", "/orders/regular")]
    assert len(place_calls) == 1


def test_place_order_rejects_an_unsupported_order_type_before_any_network_call() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport, enable_live_trading=True)
    _authenticate(broker, transport)
    calls_before = len(transport.calls)
    with pytest.raises(BrokerCapabilityError, match="order_type"):
        broker.place_order(
            BrokerOrder(
                client_order_id="c-1",
                broker_order_id=None,
                instrument_id="NSE:INFY",
                side="buy",
                quantity=10,
                order_type="MARKET",
                limit_price=1500.0,
                status=OrderState.CREATED.value,
            )
        )
    assert len(transport.calls) == calls_before


def test_place_order_rejects_an_unsupported_product() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport, enable_live_trading=True)
    _authenticate(broker, transport)
    order = BrokerOrder(
        client_order_id="c-1",
        broker_order_id=None,
        instrument_id="NSE:INFY",
        side="buy",
        quantity=10,
        order_type="LIMIT",
        limit_price=1500.0,
        status=OrderState.CREATED.value,
        product="MIS",
    )
    with pytest.raises(BrokerCapabilityError, match="product"):
        broker.place_order(order)


def test_place_order_rejects_a_missing_limit_price() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport, enable_live_trading=True)
    _authenticate(broker, transport)
    with pytest.raises(BrokerRequestError, match="limit_price"):
        broker.place_order(_new_order(limit_price=None))  # type: ignore[arg-type]


def test_modify_order_refused_while_live_trading_disabled() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    with pytest.raises(BrokerCapabilityError, match="live trading is disabled"):
        broker.modify_order("c-1", {"limit_price": 1501.0})


def test_modify_order_sends_only_the_provided_fields() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport, enable_live_trading=True)
    _authenticate(broker, transport)
    transport.script("POST", "/orders/regular", _ok({"order_id": "K1"}))
    transport.script("GET", "/orders/K1", _ok([_order_row("K1", "OPEN")]))
    broker.place_order(_new_order())

    transport.script("PUT", "/orders/regular/K1", _ok({"order_id": "K1"}))
    broker.modify_order("c-1", {"limit_price": 1501.0})

    _method, _path, params, _headers = transport.calls[-2]
    assert params == {"price": 1501.0}


def test_cancel_order_refused_while_live_trading_disabled() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    with pytest.raises(BrokerCapabilityError, match="live trading is disabled"):
        broker.cancel_order("c-1")


def test_cancel_order_deletes_and_reports_the_new_status() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport, enable_live_trading=True)
    _authenticate(broker, transport)
    transport.script("POST", "/orders/regular", _ok({"order_id": "K1"}))
    transport.script("GET", "/orders/K1", _ok([_order_row("K1", "OPEN")]))
    broker.place_order(_new_order())

    transport.script("DELETE", "/orders/regular/K1", _ok({"order_id": "K1"}))
    transport.script("GET", "/orders/K1", _ok([_order_row("K1", "CANCELLED")]))
    cancelled = broker.cancel_order("c-1")
    assert cancelled.status == OrderState.CANCELLED.value


# --------------------------------------------------------------------------
# close_position / close_all_positions
# --------------------------------------------------------------------------


def _quote_response(instrument_id: str, last_price: float, bid: float) -> HttpResponse:
    depth = {"buy": [{"price": bid}], "sell": []}
    return _ok({instrument_id: {"last_price": last_price, "depth": depth}})


def test_close_position_refused_while_live_trading_disabled() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    with pytest.raises(BrokerCapabilityError, match="live trading is disabled"):
        broker.close_position("NSE:INFY")


def test_close_position_with_nothing_held_raises() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport, enable_live_trading=True)
    _authenticate(broker, transport)
    transport.script("GET", "/portfolio/positions", _ok({"net": []}))
    with pytest.raises(BrokerRequestError, match="no open long position"):
        broker.close_position("NSE:INFY")


def test_close_position_sells_the_full_quantity_at_the_bid() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport, enable_live_trading=True)
    _authenticate(broker, transport)
    transport.script(
        "GET",
        "/portfolio/positions",
        _ok(
            {
                "net": [
                    {
                        "tradingsymbol": "INFY",
                        "exchange": "NSE",
                        "quantity": 10,
                        "average_price": 1500.0,
                        "product": "CNC",
                    }
                ]
            }
        ),
    )
    transport.script("GET", "/quote", _quote_response("NSE:INFY", 1510.0, 1509.5))
    transport.script("POST", "/orders/regular", _ok({"order_id": "K1"}))
    transport.script(
        "GET",
        "/orders/K1",
        _ok(
            [
                _order_row(
                    "K1", "COMPLETE", quantity=10, filled_quantity=10, transaction_type="SELL"
                )
            ]
        ),
    )

    order = broker.close_position("NSE:INFY")
    assert order.side == "sell"

    post_calls = [call for call in transport.calls if call[0] == "POST"]
    _method, _path, params, _headers = post_calls[-1]
    assert params is not None
    assert params["quantity"] == 10
    assert params["price"] == pytest.approx(1509.5)


def test_close_all_positions_closes_every_held_instrument() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport, enable_live_trading=True)
    _authenticate(broker, transport)
    transport.script(
        "GET",
        "/portfolio/positions",
        _ok(
            {
                "net": [
                    {
                        "tradingsymbol": "INFY",
                        "exchange": "NSE",
                        "quantity": 10,
                        "average_price": 1500.0,
                        "product": "CNC",
                    },
                    {
                        "tradingsymbol": "TCS",
                        "exchange": "NSE",
                        "quantity": 5,
                        "average_price": 3500.0,
                        "product": "CNC",
                    },
                ]
            }
        ),
    )
    transport.script("GET", "/quote", _quote_response("NSE:INFY", 1510.0, 1509.5))
    transport.script("GET", "/quote", _quote_response("NSE:TCS", 3510.0, 3509.5))
    transport.script("POST", "/orders/regular", _ok({"order_id": "K1"}))
    transport.script(
        "GET",
        "/orders/K1",
        _ok(
            [
                _order_row(
                    "K1", "COMPLETE", quantity=10, filled_quantity=10, transaction_type="SELL"
                )
            ]
        ),
    )
    transport.script("POST", "/orders/regular", _ok({"order_id": "K2"}))
    transport.script(
        "GET",
        "/orders/K2",
        _ok(
            [
                _order_row(
                    "K2",
                    "COMPLETE",
                    quantity=5,
                    filled_quantity=5,
                    tradingsymbol="TCS",
                    transaction_type="SELL",
                )
            ]
        ),
    )

    orders = broker.close_all_positions()
    assert len(orders) == 2


# --------------------------------------------------------------------------
# health_check
# --------------------------------------------------------------------------


def test_health_check_before_authentication_reports_unhealthy() -> None:
    broker = _broker(_FakeHttpTransport())
    status = broker.health_check()
    assert status.healthy is False
    assert status.session_active is False


def test_health_check_after_authentication_reports_healthy_with_login_time() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script("GET", "/user/profile", _ok({"user_id": "AB1234"}))
    status = broker.health_check()
    assert status.healthy is True
    assert status.session_active is True
    assert status.login_time is not None
    assert status.login_time.replace(tzinfo=None) == dt.datetime(2024, 6, 3, 9, 0, 0)


def test_health_check_reports_session_expiry_and_clears_the_access_token() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script("GET", "/user/profile", _err(403, "session expired", "TokenException"))
    status = broker.health_check()
    assert status.healthy is False
    assert status.session_active is False

    with pytest.raises(BrokerAuthenticationError):
        broker.get_account()  # access_token was cleared


def test_health_check_reports_unhealthy_but_still_active_on_a_generic_error() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script("GET", "/user/profile", _err(500, "internal error", "GeneralException"))
    status = broker.health_check()
    assert status.healthy is False
    assert status.session_active is True


# --------------------------------------------------------------------------
# Error-envelope mapping
# --------------------------------------------------------------------------


def test_rate_limit_response_raises_rate_limit_error() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script(
        "GET", "/user/margins/equity", _err(429, "too many requests", "NetworkException")
    )
    with pytest.raises(BrokerRateLimitError):
        broker.get_account()


def test_input_exception_raises_a_generic_request_error_with_error_type() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script("GET", "/user/margins/equity", _err(400, "bad params", "InputException"))
    with pytest.raises(BrokerRequestError) as excinfo:
        broker.get_account()
    assert excinfo.value.error_type == "InputException"


def test_token_exception_at_any_status_code_raises_session_expired() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script("GET", "/user/margins/equity", _err(400, "token invalid", "TokenException"))
    with pytest.raises(BrokerSessionExpiredError):
        broker.get_account()


def test_non_json_response_raises_a_generic_broker_error() -> None:
    transport = _FakeHttpTransport()
    broker = _broker(transport)
    _authenticate(broker, transport)
    transport.script(
        "GET", "/user/margins/equity", HttpResponse(status_code=200, body=b"<html>not json</html>")
    )
    with pytest.raises(BrokerError):
        broker.get_account()


# --------------------------------------------------------------------------
# End-to-end through OrderManager -- the generic-interface guarantee
# --------------------------------------------------------------------------


def test_order_manager_end_to_end_submission_through_a_real_kite_broker() -> None:
    """The same OrderManager code that drives PaperBroker (Phase 14)
    drives KiteBroker unchanged -- this is the concrete proof that
    strategy code never needs to know which adapter it is talking to.
    """
    transport = _FakeHttpTransport()
    broker = _broker(transport, enable_live_trading=True)
    _authenticate(broker, transport)
    transport.script("POST", "/orders/regular", _ok({"order_id": "K1"}))
    transport.script("GET", "/orders/K1", _ok([_order_row("K1", "COMPLETE", filled_quantity=10)]))

    order_manager = OrderManager()
    created = order_manager.create(
        "NSE:INFY",
        "buy",
        10,
        "LIMIT",
        1500.0,
        idempotency_key="strategy-signal-1",
        signal_id="sig-1",
        risk_decision_id="rd-1",
    )
    record = order_manager.submit(created.order.client_order_id, broker)

    assert record.state is OrderState.FILLED
    assert record.filled_quantity == 10
    assert record.broker_order_id == "K1"
