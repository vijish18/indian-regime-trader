"""Unit tests for the Kite WebSocket control-plane and binary tick
decoder (``broker/zerodha/kite_ticker.py``, Phase 15).

Byte layouts are hand-built here exactly per the verified spec quoted in
that module's docstring -- no real socket, no real Kite connection.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from broker.base import BrokerQuote
from broker.zerodha.kite_ticker import (
    KiteTicker,
    build_mode_message,
    build_subscribe_message,
    build_unsubscribe_message,
    build_ws_url,
    decode_binary_ticks,
)


def _int32(value: int) -> bytes:
    return int(value).to_bytes(4, "big", signed=True)


def _int16(value: int) -> bytes:
    return int(value).to_bytes(2, "big", signed=True)


def _ltp_packet(token: int, price: float) -> bytes:
    return _int32(token) + _int32(round(price * 100))


def _quote_packet(
    token: int,
    price: float,
    *,
    ltq: int = 10,
    atp: float = 100.0,
    volume: int = 500_000,
    buy_qty: int = 1000,
    sell_qty: int = 900,
    open_: float = 99.0,
    high: float = 101.0,
    low: float = 98.5,
    close: float = 100.5,
) -> bytes:
    return (
        _int32(token)
        + _int32(round(price * 100))
        + _int32(ltq)
        + _int32(round(atp * 100))
        + _int32(volume)
        + _int32(buy_qty)
        + _int32(sell_qty)
        + _int32(round(open_ * 100))
        + _int32(round(high * 100))
        + _int32(round(low * 100))
        + _int32(round(close * 100))
    )


def _depth_level(quantity: int, price: float, orders: int) -> bytes:
    return _int32(quantity) + _int32(round(price * 100)) + _int16(orders) + b"\x00\x00"


def _full_packet(
    token: int,
    price: float,
    *,
    last_traded_epoch: int = 1_700_000_000,
    open_interest: int = 0,
    exchange_epoch: int = 1_700_000_005,
) -> bytes:
    quote = _quote_packet(token, price)
    tail = _int32(last_traded_epoch) + _int32(open_interest) + _int32(0) + _int32(0)
    tail += _int32(exchange_epoch)
    buy_levels = b"".join(_depth_level(100 * (5 - i), price - 0.05 * i, 3) for i in range(5))
    sell_levels = b"".join(_depth_level(80 * (5 - i), price + 0.05 * (i + 1), 2) for i in range(5))
    packet = quote + tail + buy_levels + sell_levels
    assert len(packet) == 184
    return packet


def _frame(*packets: bytes) -> bytes:
    body = b"".join(_int16(len(packet)) + packet for packet in packets)
    return _int16(len(packets)) + body


# --------------------------------------------------------------------------
# URL and control-message construction
# --------------------------------------------------------------------------


def test_build_ws_url_carries_api_key_and_access_token() -> None:
    url = build_ws_url("mykey", "mytoken")
    assert url == "wss://ws.kite.trade?api_key=mykey&access_token=mytoken"


def test_build_subscribe_message_is_the_documented_json_shape() -> None:
    message = json.loads(build_subscribe_message([408065, 884737]))
    assert message == {"a": "subscribe", "v": [408065, 884737]}


def test_build_unsubscribe_message_is_the_documented_json_shape() -> None:
    message = json.loads(build_unsubscribe_message([408065]))
    assert message == {"a": "unsubscribe", "v": [408065]}


def test_build_mode_message_is_the_documented_json_shape() -> None:
    message = json.loads(build_mode_message("full", [408065]))
    assert message == {"a": "mode", "v": ["full", [408065]]}


def test_build_mode_message_rejects_an_unrecognized_mode() -> None:
    with pytest.raises(ValueError, match="ltp/quote/full"):
        build_mode_message("bogus", [408065])


# --------------------------------------------------------------------------
# Binary tick decoding
# --------------------------------------------------------------------------


def test_decode_ltp_mode_packet() -> None:
    frame = _frame(_ltp_packet(408065, 1502.50))
    ticks = decode_binary_ticks(frame)
    assert len(ticks) == 1
    tick = ticks[0]
    assert tick.instrument_token == 408065
    assert tick.last_price == pytest.approx(1502.50)
    assert tick.volume is None
    assert tick.depth is None


def test_decode_quote_mode_packet() -> None:
    frame = _frame(_quote_packet(408065, 1502.50, volume=123_456))
    ticks = decode_binary_ticks(frame)
    assert len(ticks) == 1
    tick = ticks[0]
    assert tick.last_price == pytest.approx(1502.50)
    assert tick.volume == 123_456
    assert tick.open_price == pytest.approx(99.0)
    assert tick.close_price == pytest.approx(100.5)
    assert tick.last_traded_timestamp is None
    assert tick.depth is None


def test_decode_full_mode_packet_includes_depth_and_timestamps() -> None:
    frame = _frame(_full_packet(408065, 1502.50))
    ticks = decode_binary_ticks(frame)
    assert len(ticks) == 1
    tick = ticks[0]
    assert tick.last_price == pytest.approx(1502.50)
    assert tick.last_traded_timestamp == dt.datetime.fromtimestamp(1_700_000_000, tz=dt.UTC)
    assert tick.exchange_timestamp == dt.datetime.fromtimestamp(1_700_000_005, tz=dt.UTC)
    assert tick.depth is not None
    assert len(tick.depth.buy) == 5
    assert len(tick.depth.sell) == 5
    assert tick.depth.buy[0].price == pytest.approx(1502.50)
    assert tick.depth.buy[0].quantity == 500
    assert tick.depth.sell[0].price == pytest.approx(1502.55)


def test_decode_multiple_packets_in_one_frame() -> None:
    frame = _frame(_ltp_packet(111, 10.0), _full_packet(222, 20.0), _quote_packet(333, 30.0))
    ticks = decode_binary_ticks(frame)
    assert [tick.instrument_token for tick in ticks] == [111, 222, 333]
    assert [tick.last_price for tick in ticks] == pytest.approx([10.0, 20.0, 30.0])


def test_decode_empty_or_too_short_payload_returns_no_ticks() -> None:
    assert decode_binary_ticks(b"") == []
    assert decode_binary_ticks(b"\x00") == []


def test_decode_skips_a_packet_of_unrecognized_length_without_raising() -> None:
    weird_packet = _int32(1) + b"\x01\x02\x03"  # 7 bytes -- not any known mode
    frame = _frame(weird_packet, _ltp_packet(555, 5.0))
    ticks = decode_binary_ticks(frame)
    assert [tick.instrument_token for tick in ticks] == [555]


def test_decode_zero_packet_count_returns_no_ticks() -> None:
    assert decode_binary_ticks(_frame()) == []


# --------------------------------------------------------------------------
# KiteTicker: subscription control and dispatch, driven by the caller
# --------------------------------------------------------------------------


class _FakeTickerTransport:
    def __init__(self) -> None:
        self.connected_url: str | None = None
        self.sent: list[str] = []
        self.closed = False
        self._frames: list[bytes | str] = []

    def queue_frame(self, frame: bytes | str) -> None:
        self._frames.append(frame)

    def connect(self, url: str) -> None:
        self.connected_url = url

    def send_text(self, message: str) -> None:
        self.sent.append(message)

    def recv(self) -> bytes | str:
        return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True


def test_subscribe_connects_and_sends_the_subscribe_message() -> None:
    transport = _FakeTickerTransport()
    tokens_by_id = {"NSE:INFY": 408065}
    ticker = KiteTicker(transport, token_resolver=lambda instrument_id: tokens_by_id[instrument_id])

    ticker.subscribe("key", "token", ["NSE:INFY"], on_tick=lambda quote: None)

    assert transport.connected_url == "wss://ws.kite.trade?api_key=key&access_token=token"
    assert json.loads(transport.sent[-1]) == {"a": "subscribe", "v": [408065]}


def test_process_next_frame_dispatches_a_decoded_tick_as_a_broker_quote() -> None:
    transport = _FakeTickerTransport()
    ticker = KiteTicker(transport, token_resolver=lambda instrument_id: 408065)
    received: list[BrokerQuote] = []
    ticker.subscribe("key", "token", ["NSE:INFY"], on_tick=received.append)

    transport.queue_frame(_frame(_full_packet(408065, 1502.50)))
    dispatched = ticker.process_next_frame()

    assert dispatched == 1
    assert len(received) == 1
    quote = received[0]
    assert quote.instrument_id == "NSE:INFY"
    assert quote.last_price == pytest.approx(1502.50)
    assert quote.bid == pytest.approx(1502.50)  # depth.buy[0].price
    assert quote.ask == pytest.approx(1502.55)  # depth.sell[0].price


def test_process_next_frame_ignores_a_text_postback_message() -> None:
    transport = _FakeTickerTransport()
    ticker = KiteTicker(transport, token_resolver=lambda instrument_id: 408065)
    ticker.subscribe("key", "token", ["NSE:INFY"], on_tick=lambda quote: None)

    transport.queue_frame(json.dumps({"type": "order", "data": {}}))
    assert ticker.process_next_frame() == 0


def test_process_next_frame_ignores_a_tick_for_an_unsubscribed_token() -> None:
    transport = _FakeTickerTransport()
    ticker = KiteTicker(transport, token_resolver=lambda instrument_id: 408065)
    received: list[BrokerQuote] = []
    ticker.subscribe("key", "token", ["NSE:INFY"], on_tick=received.append)

    transport.queue_frame(_frame(_ltp_packet(999999, 1.0)))
    assert ticker.process_next_frame() == 0
    assert received == []


def test_unsubscribe_sends_the_unsubscribe_message_and_stops_dispatch() -> None:
    transport = _FakeTickerTransport()
    ticker = KiteTicker(transport, token_resolver=lambda instrument_id: 408065)
    received: list[BrokerQuote] = []
    unsubscribe = ticker.subscribe("key", "token", ["NSE:INFY"], on_tick=received.append)

    unsubscribe()
    assert json.loads(transport.sent[-1]) == {"a": "unsubscribe", "v": [408065]}

    transport.queue_frame(_frame(_ltp_packet(408065, 1.0)))
    assert ticker.process_next_frame() == 0
    assert received == []


def test_close_closes_the_underlying_transport() -> None:
    transport = _FakeTickerTransport()
    ticker = KiteTicker(transport, token_resolver=lambda instrument_id: 408065)
    ticker.close()
    assert transport.closed is True
