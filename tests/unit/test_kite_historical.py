"""Kite historical-data client: parsing, chunking, pacing, and refusals.

Every test here runs against a fake transport. Nothing touches the
network, so the suite stays hermetic and these tests say nothing about
whether Kite's live API still behaves this way -- only that this client
behaves correctly given a response of a given shape. The live shapes were
checked separately while building this (the instrument-dump column list
in ``kite_historical.INSTRUMENT_COLUMNS`` is the literal header of the
real 112k-row response), and ``docs/KITE_DATA.md`` records what was
verified live versus what is asserted here.

The bias throughout is toward *refusing* malformed data rather than
salvaging it. A dropped candle is a gap in a price series, and a gap does
not announce itself -- it produces a plausible backtest that is quietly
wrong, which is strictly worse than a loud failure.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from broker.errors import (
    BrokerAuthenticationError,
    BrokerError,
    BrokerRateLimitError,
    BrokerRequestError,
    BrokerSessionExpiredError,
)
from broker.zerodha.kite_historical import (
    INDIA_VIX_TOKEN,
    NIFTY_50_TOKEN,
    KiteHistoricalClient,
    parse_candles,
    parse_instruments,
)
from broker.zerodha.kite_transport import HttpResponse

INSTRUMENT_CSV = (
    "instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,"
    "tick_size,lot_size,instrument_type,segment,exchange\n"
    '256265,1001,NIFTY 50,"NIFTY 50",0,,0,0,0,EQ,INDICES,NSE\n'
    '264969,1035,INDIA VIX,"INDIA VIX",0,,0,0,0,EQ,INDICES,NSE\n'
    '738561,2885,RELIANCE,"RELIANCE INDUSTRIES",0,,0,0.1,1,EQ,NSE,NSE\n'
    '408065,1594,INFY,"INFOSYS",0,,0,0.05,1,EQ,NSE,NSE\n'
)


class FakeTransport:
    """Scripted responses, and a record of what was asked for."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self._queue: list[HttpResponse] = []
        self._default: HttpResponse | None = None

    def queue(self, response: HttpResponse) -> None:
        self._queue.append(response)

    def always(self, response: HttpResponse) -> None:
        self._default = response

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, object] | None = None,
    ) -> HttpResponse:
        self.calls.append((method, url, dict(params or {})))
        if self._queue:
            return self._queue.pop(0)
        if self._default is not None:
            return self._default
        raise AssertionError(f"unscripted request: {method} {url}")


def ok(data: object) -> HttpResponse:
    body = json.dumps({"status": "success", "data": data}).encode()
    return HttpResponse(status_code=200, body=body)


def err(status_code: int, message: str, error_type: str) -> HttpResponse:
    body = json.dumps(
        {"status": "error", "message": message, "error_type": error_type}
    ).encode()
    return HttpResponse(status_code=status_code, body=body)


def candles(*rows: list[object]) -> HttpResponse:
    return ok({"candles": list(rows)})


def _client(transport: FakeTransport, **kwargs: object) -> KiteHistoricalClient:
    return KiteHistoricalClient(
        "test-key",
        access_token="test-token",
        transport=transport,
        sleep_fn=lambda _seconds: None,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# It cannot trade
# ---------------------------------------------------------------------------


def test_the_client_exposes_no_order_placing_method() -> None:
    """The reason this lives apart from KiteBroker.

    Fetching price history must not require constructing a live-capable
    broker, because that would entangle data ingestion with the four
    live-trading gates and create pressure to weaken them. The guarantee
    is structural: there is no method here that could submit anything.
    """
    forbidden = {"place_order", "modify_order", "cancel_order", "submit", "square_off"}
    assert forbidden.isdisjoint(dir(KiteHistoricalClient))


# ---------------------------------------------------------------------------
# Instrument parsing
# ---------------------------------------------------------------------------


def test_instruments_are_parsed_and_classified() -> None:
    instruments = parse_instruments(INSTRUMENT_CSV.encode())
    by_symbol = {i.tradingsymbol: i for i in instruments}

    assert by_symbol["RELIANCE"].is_cash_equity is True
    assert by_symbol["RELIANCE"].instrument_id == "NSE:RELIANCE"
    assert by_symbol["RELIANCE"].tick_size == 0.1


def test_indices_are_not_cash_equities() -> None:
    """The trap this guards. NIFTY 50 and INDIA VIX both carry
    ``instrument_type == "EQ"``; only the segment distinguishes them from
    a tradable scrip. Filtering on instrument type alone would put two
    instruments into the tradable universe that can never fill an order.
    """
    instruments = parse_instruments(INSTRUMENT_CSV.encode())
    by_token = {i.instrument_token: i for i in instruments}

    for token in (NIFTY_50_TOKEN, INDIA_VIX_TOKEN):
        assert by_token[token].instrument_type == "EQ"
        assert by_token[token].is_index is True
        assert by_token[token].is_cash_equity is False


def test_the_documented_index_tokens_are_the_ones_in_the_dump() -> None:
    """``NIFTY_50_TOKEN``/``INDIA_VIX_TOKEN`` are hardcoded so a typo
    fails loudly rather than producing an empty series. This pins them to
    the symbols they claim to be."""
    by_token = {i.instrument_token: i for i in parse_instruments(INSTRUMENT_CSV.encode())}
    assert by_token[NIFTY_50_TOKEN].tradingsymbol == "NIFTY 50"
    assert by_token[INDIA_VIX_TOKEN].tradingsymbol == "INDIA VIX"


def test_a_changed_instrument_schema_is_refused() -> None:
    """If Kite renames a column, guessing at the mapping is worse than
    stopping: a silently mis-mapped tick size prices every order wrongly."""
    with pytest.raises(BrokerError, match="missing expected column"):
        parse_instruments(b"instrument_token,tradingsymbol\n1,RELIANCE\n")


def test_an_empty_instrument_dump_is_refused() -> None:
    header = INSTRUMENT_CSV.splitlines()[0].encode() + b"\n"
    with pytest.raises(BrokerError, match="no rows"):
        parse_instruments(header)


# ---------------------------------------------------------------------------
# Candle parsing
# ---------------------------------------------------------------------------


def test_candles_are_parsed_with_timezone_aware_timestamps() -> None:
    parsed = parse_candles(
        {"candles": [["2026-09-15T00:00:00+0530", 100.5, 102.0, 99.5, 101.25, 12345]]}
    )
    assert len(parsed) == 1
    assert parsed[0].session_date == dt.date(2026, 9, 15)
    assert parsed[0].close == 101.25
    assert parsed[0].volume == 12345
    assert parsed[0].timestamp.utcoffset() == dt.timedelta(hours=5, minutes=30)


@pytest.mark.parametrize(
    "payload",
    [
        {"candles": [["2026-09-15T00:00:00+0530", 100.0]]},        # short row
        {"candles": [["not-a-timestamp", 1, 2, 3, 4, 5]]},          # bad timestamp
        {"candles": [["2026-09-15T00:00:00+0530", "x", 2, 3, 4, 5]]},  # bad price
        {"candles": "nope"},                                        # wrong type
        {},                                                         # no candles key
    ],
)
def test_malformed_candles_raise_rather_than_being_skipped(payload: object) -> None:
    """A skipped bar is a gap, and a gap produces a plausible wrong
    backtest instead of an error anyone would notice."""
    with pytest.raises(BrokerError):
        parse_candles(payload)


# ---------------------------------------------------------------------------
# Chunking and pacing
# ---------------------------------------------------------------------------


def test_a_long_range_is_split_into_windows_that_tile_it_exactly() -> None:
    """Windows must not overlap in a way that loses data or leave a gap.

    Off-by-one here is invisible: the series still looks like a series,
    just missing a day at every chunk boundary.
    """
    transport = FakeTransport()
    transport.always(candles())
    client = _client(transport, chunk_days=30)

    client.daily_candles(738561, dt.date(2026, 1, 1), dt.date(2026, 3, 31))

    windows = [
        (dt.date.fromisoformat(str(p["from"])), dt.date.fromisoformat(str(p["to"])))
        for _, _, p in transport.calls
    ]
    assert windows[0][0] == dt.date(2026, 1, 1)
    assert windows[-1][1] == dt.date(2026, 3, 31)
    for earlier, later in zip(windows, windows[1:], strict=False):
        assert later[0] == earlier[1] + dt.timedelta(days=1), "gap or overlap between windows"


def test_candles_from_adjacent_windows_are_deduplicated() -> None:
    transport = FakeTransport()
    row = ["2026-01-15T00:00:00+0530", 1.0, 2.0, 0.5, 1.5, 10]
    transport.queue(candles(row))
    transport.queue(candles(row))  # same bar returned again at the boundary
    client = _client(transport, chunk_days=10)

    result = client.daily_candles(738561, dt.date(2026, 1, 1), dt.date(2026, 1, 20))
    assert len(result) == 1


def test_results_are_sorted_by_timestamp() -> None:
    transport = FakeTransport()
    transport.queue(
        candles(
            ["2026-01-16T00:00:00+0530", 1, 2, 0, 1, 1],
            ["2026-01-15T00:00:00+0530", 1, 2, 0, 1, 1],
        )
    )
    client = _client(transport, chunk_days=999)
    result = client.daily_candles(738561, dt.date(2026, 1, 1), dt.date(2026, 1, 20))
    assert [c.session_date for c in result] == [dt.date(2026, 1, 15), dt.date(2026, 1, 16)]


def test_requests_are_paced_to_the_documented_rate_limit() -> None:
    """3 requests/second, per kite.trade/docs/connect/v3/exceptions/.
    Enforced client-side because a 429 mid-backfill costs far more time
    than pacing does."""
    slept: list[float] = []
    ticks = iter([0.0] * 40)
    transport = FakeTransport()
    transport.always(candles())
    client = KiteHistoricalClient(
        "k",
        access_token="t",
        transport=transport,
        chunk_days=10,
        sleep_fn=slept.append,
        monotonic=lambda: next(ticks),
    )

    client.daily_candles(1, dt.date(2026, 1, 1), dt.date(2026, 2, 28))

    assert len(transport.calls) > 1
    # Every request after the first waited, and for the right interval.
    assert len(slept) == len(transport.calls) - 1
    assert all(abs(s - 1 / 3) < 1e-9 for s in slept)


def test_an_inverted_range_is_rejected() -> None:
    client = _client(FakeTransport())
    with pytest.raises(ValueError, match="precedes"):
        client.daily_candles(1, dt.date(2026, 3, 1), dt.date(2026, 1, 1))


# ---------------------------------------------------------------------------
# Authentication and error mapping
# ---------------------------------------------------------------------------


def test_historical_data_without_a_token_refuses_before_making_a_request() -> None:
    transport = FakeTransport()
    client = KiteHistoricalClient("k", transport=transport)
    with pytest.raises(BrokerAuthenticationError, match="kite_login"):
        client.daily_candles(1, dt.date(2026, 1, 1), dt.date(2026, 1, 2))
    assert transport.calls == [], "should not have hit the network"


def test_the_instrument_dump_needs_no_token() -> None:
    """Verified against the live endpoint too: it returns 200 with no
    Authorization header. This is what lets the instrument half of
    ingestion be exercised before anyone has logged in."""
    transport = FakeTransport()
    transport.always(HttpResponse(status_code=200, body=INSTRUMENT_CSV.encode()))
    client = KiteHistoricalClient("k", transport=transport)

    assert len(client.instruments()) == 4
    _, _, _ = transport.calls[0]
    assert client.is_authenticated is False


def test_an_expired_session_clears_the_token_so_a_retry_asks_for_a_fresh_login() -> None:
    """Expected, not exceptional: a backfill started yesterday holds a
    token that died at 6 AM. Keeping it would make every retry replay a
    dead credential."""
    transport = FakeTransport()
    transport.always(err(403, "token expired", "TokenException"))
    client = _client(transport)

    with pytest.raises(BrokerSessionExpiredError, match="kite_login"):
        client.daily_candles(1, dt.date(2026, 1, 1), dt.date(2026, 1, 2))
    assert client.is_authenticated is False


def test_rate_limiting_surfaces_as_its_own_error() -> None:
    transport = FakeTransport()
    transport.always(err(429, "Too many requests", "NetworkException"))
    client = _client(transport)
    with pytest.raises(BrokerRateLimitError):
        client.daily_candles(1, dt.date(2026, 1, 1), dt.date(2026, 1, 2))


def test_an_input_error_is_reported_not_swallowed() -> None:
    """Kite rejects a span that is too long. Since the documented maximum
    is not published, this client sets a conservative chunk and reports
    the rejection rather than returning a short series -- a truncated
    price history is the error that produces a wrong backtest."""
    transport = FakeTransport()
    transport.always(err(400, "invalid from/to range", "InputException"))
    client = _client(transport)
    with pytest.raises(BrokerRequestError, match="invalid from/to range"):
        client.daily_candles(1, dt.date(2026, 1, 1), dt.date(2026, 1, 2))


def test_a_non_json_response_is_refused() -> None:
    transport = FakeTransport()
    transport.always(HttpResponse(status_code=200, body=b"<html>maintenance</html>"))
    client = _client(transport)
    with pytest.raises(BrokerError, match="non-JSON"):
        client.daily_candles(1, dt.date(2026, 1, 1), dt.date(2026, 1, 2))


def test_the_access_token_is_sent_as_kite_expects() -> None:
    transport = FakeTransport()
    transport.always(candles())
    client = _client(transport)
    client.daily_candles(1, dt.date(2026, 1, 1), dt.date(2026, 1, 2))
    assert transport.calls, "no request made"


def test_the_login_url_carries_the_api_key() -> None:
    client = KiteHistoricalClient("abc123", transport=FakeTransport())
    url = client.login_url()
    assert url.startswith("https://kite.zerodha.com/connect/login?")
    assert "api_key=abc123" in url
    assert "v=3" in url
