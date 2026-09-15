"""Kite Connect v3 WebSocket streaming: connection URL/control-message
construction and binary tick decoding, verified against
https://kite.trade/docs/connect/v3/websocket/.

## What's implemented, and why no socket ships by default

The WebSocket URL, the JSON subscribe/unsubscribe/mode control messages,
and the binary tick frame format (packet-count header, per-packet length
prefix, and the full/quote/ltp field layouts for equities) are all
verified against Zerodha's published documentation and implemented here
as pure, fully unit-testable logic -- no live connection required to
test any of it.

Actually opening a real ``wss://`` socket needs a WebSocket client this
codebase does not otherwise depend on, and this phase's scope is
explicit: default mode stays PAPER, nothing here connects to anything
real. :class:`KiteTickerTransport` is the seam -- inject a real
implementation (e.g. backed by the ``websocket-client`` package, the
same one Zerodha's own official Python SDK uses) only at the point live
streaming is actually being enabled; nothing in this codebase does that
today.

## Driven by the caller, not a background thread

:meth:`KiteTicker.process_next_frame` receives and dispatches exactly
one frame per call -- a live/paper trading loop is expected to call it
repeatedly, the same "the caller drives the tick" pattern
``broker.adapters.paper_broker.PaperBroker.process_resting_orders``
already established in this codebase (Phase 14). This class starts no
thread of its own, which is what keeps it deterministic and safe to
unit test with a scripted fake transport.

## Documented gap: index instruments

The byte layout verified and implemented below is for **equity**
packets. Kite's index packets (NIFTY 50, India VIX) use a shorter,
differently-terminated quote-mode layout the published docs describe
only in prose, not a precise byte table -- decoding it here without a
verified table would be exactly the kind of invented behavior this
phase must not produce. Index ticks are therefore out of scope for
:func:`decode_binary_ticks`; this system's regime features already read
NIFTY/VIX from ``data.interfaces.MarketDataProvider`` (Phase 4), not
from this streaming path.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from broker.base import BrokerQuote

_WS_URL = "wss://ws.kite.trade"

# Price scaling (docs/connect/v3/websocket/, verified): "For currencies,
# the int32 price values should be divided by 10000000 ... For
# everything else, the price values should be divided by 100." This
# system trades NSE/BSE cash equities only (V1 scope) -- the currency
# segment is never subscribed to, so the /100 rule is applied
# unconditionally rather than requiring a segment lookup this system has
# no other use for.
_PRICE_DIVISOR = 100.0


@dataclass(frozen=True, slots=True)
class DepthLevel:
    quantity: int
    price: float
    orders: int


@dataclass(frozen=True, slots=True)
class MarketDepth:
    buy: tuple[DepthLevel, ...]
    sell: tuple[DepthLevel, ...]


@dataclass(frozen=True, slots=True)
class KiteTick:
    """One decoded equity tick packet. Fields beyond ``instrument_token``/
    ``last_price`` are ``None`` when the packet's mode did not include
    them (LTP mode: only token + last_price; quote mode: adds the fields
    up to OHLC; full mode: adds timestamps, open interest, and depth).
    """

    instrument_token: int
    last_price: float
    last_traded_quantity: int | None = None
    average_traded_price: float | None = None
    volume: int | None = None
    total_buy_quantity: int | None = None
    total_sell_quantity: int | None = None
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    last_traded_timestamp: dt.datetime | None = None
    open_interest: int | None = None
    open_interest_day_high: int | None = None
    open_interest_day_low: int | None = None
    exchange_timestamp: dt.datetime | None = None
    depth: MarketDepth | None = None


def build_ws_url(api_key: str, access_token: str) -> str:
    """docs/connect/v3/websocket/ (verified): the endpoint is
    ``wss://ws.kite.trade`` with required query parameters ``api_key``
    and ``access_token``."""
    return f"{_WS_URL}?api_key={api_key}&access_token={access_token}"


def build_subscribe_message(instrument_tokens: list[int]) -> str:
    return json.dumps({"a": "subscribe", "v": instrument_tokens})


def build_unsubscribe_message(instrument_tokens: list[int]) -> str:
    return json.dumps({"a": "unsubscribe", "v": instrument_tokens})


def build_mode_message(mode: str, instrument_tokens: list[int]) -> str:
    if mode not in ("ltp", "quote", "full"):
        raise ValueError(f"mode must be one of ltp/quote/full, got {mode!r}")
    return json.dumps({"a": "mode", "v": [mode, instrument_tokens]})


def _read_int32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "big", signed=True)


def _read_int16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "big", signed=True)


def _read_price(data: bytes, offset: int) -> float:
    return _read_int32(data, offset) / _PRICE_DIVISOR


def _read_epoch_seconds(data: bytes, offset: int) -> dt.datetime | None:
    value = _read_int32(data, offset)
    if value <= 0:
        return None
    return dt.datetime.fromtimestamp(value, tz=dt.UTC)


def _decode_depth(data: bytes) -> MarketDepth:
    """docs/connect/v3/websocket/ (verified): 10 depth entries in
    succession, 5 buy (bytes 0-60 of this 120-byte slice) then 5 sell
    (bytes 60-120), each 12 bytes: int32 quantity, int32 price, int16
    orders, 2 bytes padding."""

    def _levels(start: int) -> tuple[DepthLevel, ...]:
        levels = []
        for i in range(5):
            base = start + i * 12
            levels.append(
                DepthLevel(
                    quantity=_read_int32(data, base),
                    price=_read_price(data, base + 4),
                    orders=_read_int16(data, base + 8),
                )
            )
        return tuple(levels)

    return MarketDepth(buy=_levels(0), sell=_levels(60))


def _decode_one_packet(packet: bytes) -> KiteTick | None:
    """Decode one equity tick packet. Returns ``None`` for a packet
    length this function does not recognize (e.g. an index packet --
    see this module's docstring) rather than guessing a layout.
    """
    if len(packet) < 8:
        return None

    instrument_token = _read_int32(packet, 0)
    last_price = _read_price(packet, 4)

    if len(packet) == 8:
        return KiteTick(instrument_token=instrument_token, last_price=last_price)

    if len(packet) not in (44, 184):
        return None

    return KiteTick(
        instrument_token=instrument_token,
        last_price=last_price,
        last_traded_quantity=_read_int32(packet, 8),
        average_traded_price=_read_price(packet, 12),
        volume=_read_int32(packet, 16),
        total_buy_quantity=_read_int32(packet, 20),
        total_sell_quantity=_read_int32(packet, 24),
        open_price=_read_price(packet, 28),
        high_price=_read_price(packet, 32),
        low_price=_read_price(packet, 36),
        close_price=_read_price(packet, 40),
        last_traded_timestamp=_read_epoch_seconds(packet, 44) if len(packet) == 184 else None,
        open_interest=_read_int32(packet, 48) if len(packet) == 184 else None,
        open_interest_day_high=_read_int32(packet, 52) if len(packet) == 184 else None,
        open_interest_day_low=_read_int32(packet, 56) if len(packet) == 184 else None,
        exchange_timestamp=_read_epoch_seconds(packet, 60) if len(packet) == 184 else None,
        depth=_decode_depth(packet[64:184]) if len(packet) == 184 else None,
    )


def decode_binary_ticks(payload: bytes) -> list[KiteTick]:
    """docs/connect/v3/websocket/ (verified): "The first two bytes ...
    represent the number of packets in the message" followed by, for
    each packet, "two bytes ... represent the length ... of the ...
    packet", then the packet's own bytes. Any individual packet this
    function cannot recognize (see :func:`_decode_one_packet`) is
    skipped, not fatal to the rest of the frame.
    """
    if len(payload) < 2:
        return []
    packet_count = int.from_bytes(payload[0:2], "big", signed=False)
    ticks: list[KiteTick] = []
    offset = 2
    for _ in range(packet_count):
        if offset + 2 > len(payload):
            break
        length = int.from_bytes(payload[offset : offset + 2], "big", signed=False)
        offset += 2
        packet = payload[offset : offset + length]
        offset += length
        tick = _decode_one_packet(packet)
        if tick is not None:
            ticks.append(tick)
    return ticks


def _tick_to_quote(instrument_id: str, tick: KiteTick) -> BrokerQuote:
    bid = tick.depth.buy[0].price if tick.depth and tick.depth.buy else tick.last_price
    ask = tick.depth.sell[0].price if tick.depth and tick.depth.sell else tick.last_price
    as_of = tick.exchange_timestamp or tick.last_traded_timestamp or dt.datetime.now(dt.UTC)
    return BrokerQuote(
        instrument_id=instrument_id, bid=bid, ask=ask, last_price=tick.last_price, as_of=as_of
    )


class KiteTickerTransport(Protocol):
    """Duck-typed WebSocket transport -- no concrete implementation ships
    in this codebase (see this module's docstring)."""

    def connect(self, url: str) -> None: ...

    def send_text(self, message: str) -> None: ...

    def recv(self) -> bytes | str: ...

    def close(self) -> None: ...


class KiteTicker:
    """Subscription control plus tick decoding for Kite's WebSocket feed.
    Owns no thread and no real socket -- see this module's docstring.
    """

    def __init__(
        self, transport: KiteTickerTransport, token_resolver: Callable[[str], int]
    ) -> None:
        self._transport = transport
        self._token_resolver = token_resolver
        self._on_tick: Callable[[BrokerQuote], None] | None = None
        self._token_to_instrument_id: dict[int, str] = {}
        self._connected = False

    def subscribe(
        self,
        api_key: str,
        access_token: str,
        instrument_ids: list[str],
        on_tick: Callable[[BrokerQuote], None],
    ) -> Callable[[], None]:
        tokens = [self._token_resolver(instrument_id) for instrument_id in instrument_ids]
        self._token_to_instrument_id.update(zip(tokens, instrument_ids, strict=True))
        if not self._connected:
            self._transport.connect(build_ws_url(api_key, access_token))
            self._connected = True
        self._on_tick = on_tick
        self._transport.send_text(build_subscribe_message(tokens))

        subscribed = list(tokens)

        def _unsubscribe() -> None:
            self._transport.send_text(build_unsubscribe_message(subscribed))
            for token in subscribed:
                self._token_to_instrument_id.pop(token, None)

        return _unsubscribe

    def process_next_frame(self) -> int:
        """Receive and dispatch exactly one frame. A live/paper trading
        loop calls this repeatedly; this class never schedules itself.
        Returns the number of ticks dispatched (``0`` for a non-tick
        frame, e.g. a JSON postback).
        """
        frame = self._transport.recv()
        if isinstance(frame, str):
            return 0
        dispatched = 0
        for tick in decode_binary_ticks(frame):
            instrument_id = self._token_to_instrument_id.get(tick.instrument_token)
            if instrument_id is None or self._on_tick is None:
                continue
            self._on_tick(_tick_to_quote(instrument_id, tick))
            dispatched += 1
        return dispatched

    def close(self) -> None:
        self._transport.close()
        self._connected = False
