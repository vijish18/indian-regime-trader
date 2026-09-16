"""Kite Connect historical market data: instrument master and daily candles.

Separate from :mod:`broker.zerodha.kite_broker` on purpose, and the
separation is the point rather than an organisational preference.

**This client cannot place an order.** It touches exactly three endpoints
-- ``GET /instruments``, ``GET /instruments/historical/...`` and
``POST /session/token`` -- and has no method that submits, modifies or
cancels anything. So fetching price history never requires constructing a
live-capable ``KiteBroker``, and none of the four live-trading gates in
``broker.factory.build_broker`` are involved in, or weakened by, getting
data in. Ingesting history and placing orders are different privileges
and this module holds only the first.

It lives in ``broker/zerodha/`` rather than ``data/`` because ``data/``
sits below ``broker/`` in this repository's layering (see
docs/ARCHITECTURE.md's dependency rules) and would have to import upward
to reach the shared HTTP transport. The wiring that turns fetched candles
into ingested bars is a ``scripts/`` concern, where importing from both
layers is fine.

**Access requires a daily login.** Kite access tokens expire at 6 AM the
following day -- a regulatory requirement, not a setting -- and obtaining
one needs a human to complete a browser login. ``scripts/kite_login.py``
drives that; this module only knows how to exchange the resulting
``request_token`` and to use the result.

Verified against the published documentation
(``kite.trade/docs/connect/v3/historical/`` and ``.../exceptions/``) and,
for the instrument dump, against the real response: the CSV columns below
are the actual header of ``https://api.kite.trade/instruments``, not a
guess.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import time
import urllib.parse
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from broker.errors import (
    BrokerAuthenticationError,
    BrokerError,
    BrokerRateLimitError,
    BrokerRequestError,
    BrokerSessionExpiredError,
)
from broker.zerodha.kite_transport import HttpResponse, HttpTransport, UrllibHttpTransport
from monitoring.logger import get_logger

logger = get_logger("broker.zerodha.kite_historical")

BASE_URL = "https://api.kite.trade"
LOGIN_URL = "https://kite.zerodha.com/connect/login"
KITE_VERSION = "3"

HISTORICAL_REQUESTS_PER_SECOND = 3.0
"""Documented at kite.trade/docs/connect/v3/exceptions/. Enforced client
side because exceeding it earns a 429, and a 429 in the middle of a
multi-year backfill costs far more time than pacing does."""

DEFAULT_CHUNK_DAYS = 365
"""How much history to request at once.

Kite caps the span of a single historical request, but does **not**
document the cap, so this is set conservatively rather than tuned to an
undocumented edge. If Kite rejects a span it says so and
:class:`KiteHistoricalClient` surfaces that rather than silently
returning a short series -- a truncated price history is the kind of
error that produces a plausible, wrong backtest.
"""

INSTRUMENT_COLUMNS = (
    "instrument_token",
    "exchange_token",
    "tradingsymbol",
    "name",
    "last_price",
    "expiry",
    "strike",
    "tick_size",
    "lot_size",
    "instrument_type",
    "segment",
    "exchange",
)
"""The literal header of the live instruments dump."""

NIFTY_50_TOKEN = 256265
INDIA_VIX_TOKEN = 264969
"""The two index series ``core.features`` needs. Hardcoded because they
are stable identifiers for named instruments, and resolving them by
string match at runtime would make a typo look like an empty universe.
``tests/unit/test_kite_historical.py`` checks them against the shipped
instrument-master snapshot."""


@dataclass(frozen=True, slots=True)
class KiteInstrument:
    instrument_token: int
    exchange_token: str
    tradingsymbol: str
    name: str
    expiry: str
    strike: float
    tick_size: float
    lot_size: int
    instrument_type: str
    segment: str
    exchange: str

    @property
    def instrument_id(self) -> str:
        """This repository's own identifier form, ``EXCHANGE:SYMBOL``."""
        return f"{self.exchange}:{self.tradingsymbol}"

    @property
    def is_cash_equity(self) -> bool:
        """A tradable NSE cash-equity line, as opposed to an index or a
        derivative. Both carry ``instrument_type == "EQ"``; only the
        segment distinguishes them, which is easy to get wrong and
        expensive to get wrong -- an index in the tradable universe is an
        order that can never fill."""
        return self.exchange == "NSE" and self.segment == "NSE"

    @property
    def is_index(self) -> bool:
        return self.segment == "INDICES"


@dataclass(frozen=True, slots=True)
class Candle:
    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @property
    def session_date(self) -> dt.date:
        return self.timestamp.date()


class _RateLimiter:
    """Minimum spacing between requests.

    A sleeping loop rather than a token bucket: a backfill is a single
    sequential process with no burst to absorb, so the simple thing is
    also the correct thing. ``sleep_fn`` is injectable so tests do not
    spend real seconds proving the pacing works.
    """

    def __init__(
        self,
        requests_per_second: float,
        *,
        sleep_fn: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._min_interval = 1.0 / requests_per_second
        self._sleep = sleep_fn or time.sleep
        self._monotonic = monotonic or time.monotonic
        self._last: float | None = None

    def wait(self) -> None:
        now = self._monotonic()
        if self._last is not None:
            elapsed = now - self._last
            if elapsed < self._min_interval:
                self._sleep(self._min_interval - elapsed)
        self._last = self._monotonic()


class KiteHistoricalClient:
    """Reads the instrument master and daily candles. Places no orders."""

    def __init__(
        self,
        api_key: str,
        *,
        access_token: str | None = None,
        transport: HttpTransport | None = None,
        chunk_days: int = DEFAULT_CHUNK_DAYS,
        sleep_fn: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if chunk_days < 1:
            raise ValueError("chunk_days must be at least 1")
        self.api_key = api_key
        self._access_token = access_token
        self._transport: HttpTransport = transport or UrllibHttpTransport(timeout_seconds=30.0)
        self._chunk_days = chunk_days
        self._limiter = _RateLimiter(
            HISTORICAL_REQUESTS_PER_SECOND, sleep_fn=sleep_fn, monotonic=monotonic
        )

    # -- authentication ----------------------------------------------------

    @property
    def is_authenticated(self) -> bool:
        return self._access_token is not None

    def login_url(self) -> str:
        return f"{LOGIN_URL}?v=3&api_key={urllib.parse.quote(self.api_key)}"

    def exchange_request_token(self, request_token: str, api_secret: str) -> str:
        """Swap the one-time ``request_token`` from the login redirect for
        a session ``access_token``, which expires at 6 AM tomorrow."""
        if not request_token:
            raise BrokerAuthenticationError("request_token is required")
        if not api_secret:
            raise BrokerAuthenticationError("api_secret is required")

        checksum = hashlib.sha256(
            f"{self.api_key}{request_token}{api_secret}".encode()
        ).hexdigest()
        payload = self._request(
            "POST",
            "/session/token",
            params={
                "api_key": self.api_key,
                "request_token": request_token,
                "checksum": checksum,
            },
            authenticated=False,
        )
        if not isinstance(payload, dict) or "access_token" not in payload:
            raise BrokerAuthenticationError(
                "session/token response did not contain an access_token"
            )
        self._access_token = str(payload["access_token"])
        return self._access_token

    # -- instrument master -------------------------------------------------

    def instruments(self) -> tuple[KiteInstrument, ...]:
        """The full instrument dump.

        Public: this endpoint needs no access token, which is why the
        instrument-master half of ingestion can be exercised before
        anyone has completed a login.
        """
        response = self._transport.request(
            "GET", f"{BASE_URL}/instruments", headers={"X-Kite-Version": KITE_VERSION}
        )
        if response.status_code != 200:
            raise BrokerRequestError(
                f"instruments dump returned HTTP {response.status_code}", error_type=None
            )
        return parse_instruments(response.body)

    # -- historical candles ------------------------------------------------

    def daily_candles(
        self, instrument_token: int, start: dt.date, end: dt.date
    ) -> tuple[Candle, ...]:
        """Daily OHLCV for one instrument over an inclusive date range.

        Requests are split into ``chunk_days`` windows and paced to the
        documented 3/second limit. Chunks are stitched back together and
        de-duplicated on timestamp, because adjacent windows share a
        boundary and a duplicated bar would corrupt every rolling
        computation downstream.
        """
        if end < start:
            raise ValueError(f"end {end} precedes start {start}")
        if not self.is_authenticated:
            raise BrokerAuthenticationError(
                "historical data needs an access token -- run scripts/kite_login.py"
            )

        collected: list[Candle] = []
        for window_start, window_end in self._windows(start, end):
            self._limiter.wait()
            payload = self._request(
                "GET",
                f"/instruments/historical/{instrument_token}/day",
                params={
                    "from": window_start.isoformat(),
                    "to": window_end.isoformat(),
                },
            )
            collected.extend(parse_candles(payload))

        return _one_bar_per_session(collected, instrument_token)

    def _windows(self, start: dt.date, end: dt.date) -> Iterator[tuple[dt.date, dt.date]]:
        cursor = start
        step = dt.timedelta(days=self._chunk_days - 1)
        while cursor <= end:
            window_end = min(cursor + step, end)
            yield cursor, window_end
            cursor = window_end + dt.timedelta(days=1)

    # -- plumbing ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"X-Kite-Version": KITE_VERSION}
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
            raise BrokerAuthenticationError("not authenticated")
        response = self._transport.request(
            method, f"{BASE_URL}{path}", headers=self._headers(), params=params
        )
        return self._parse(response)

    def _parse(self, response: HttpResponse) -> object:
        try:
            envelope: object = response.json()
        except (ValueError, UnicodeDecodeError):
            envelope = None

        message = (
            str(envelope.get("message")) if isinstance(envelope, dict) and envelope.get("message")
            else None
        )
        error_type = (
            str(envelope.get("error_type"))
            if isinstance(envelope, dict) and envelope.get("error_type")
            else None
        )

        if response.status_code == 429:
            raise BrokerRateLimitError(message or "rate limit exceeded", error_type=error_type)
        if response.status_code == 403 or error_type == "TokenException":
            # Expired sessions are the *expected* failure here, not an
            # exceptional one: a backfill started yesterday is holding a
            # token that died at 6 AM. Clearing it means a retry asks for
            # a fresh login instead of replaying a dead credential.
            self._access_token = None
            # Kite's own message ("token expired") says what happened but
            # not what to do about it, and this is the one error an
            # operator will hit routinely -- every morning, by design. So
            # the remedy is appended rather than substituted: keep the
            # broker's wording for the record, add the next action.
            raise BrokerSessionExpiredError(
                f"{message or 'session expired'} -- run: python scripts/kite_login.py"
            )
        if envelope is None:
            raise BrokerError(f"Kite returned a non-JSON response (HTTP {response.status_code})")
        if not isinstance(envelope, dict):
            raise BrokerError(f"expected a JSON object, got {type(envelope).__name__}")
        if envelope.get("status") == "success":
            return envelope.get("data")
        raise BrokerRequestError(message or "unknown error", error_type=error_type)


def _one_bar_per_session(candles: list[Candle], instrument_token: int) -> tuple[Candle, ...]:
    """Collapse to exactly one bar per session date, latest timestamp wins.

    Deduplicating on ``session_date`` rather than on ``timestamp`` is a
    fix for real vendor data. Kite's daily candles normally carry a
    midnight timestamp, but INDIA VIX has three dates in 2015 where they
    carry an intraday one instead:

        2015-06-29T08:59:23+05:30  close 18.17
        2015-06-29T11:54:10+05:30  close 17.30

    Two bars, one session. Keyed by timestamp both survive, and the
    feature pipeline then refuses the series outright ("observations
    contain a duplicate session date") -- which is the right refusal, but
    it means three bad days in 2015 block a ten-year fit.

    The latest timestamp wins because it is the observation closest to
    the session's close, which is what a daily bar is supposed to record.
    Adjacent chunk windows share a boundary date and legitimately return
    the same bar twice; that case collapses here too, harmlessly.

    Genuine duplicates are logged rather than silently resolved: a vendor
    anomaly nobody knows about is one nobody can account for later.
    """
    by_session: dict[dt.date, Candle] = {}
    conflicts: dict[dt.date, list[dt.datetime]] = {}

    for candle in sorted(candles, key=lambda c: c.timestamp):
        existing = by_session.get(candle.session_date)
        if existing is not None and existing.timestamp != candle.timestamp:
            conflicts.setdefault(candle.session_date, [existing.timestamp]).append(
                candle.timestamp
            )
        by_session[candle.session_date] = candle

    if conflicts:
        logger.warning(
            "instrument %s: %d session(s) returned more than one daily bar; "
            "kept the latest timestamp for each: %s",
            instrument_token,
            len(conflicts),
            ", ".join(
                f"{day} ({len(stamps)} bars)" for day, stamps in sorted(conflicts.items())
            ),
        )

    return tuple(by_session[key] for key in sorted(by_session))


# -- parsing (pure functions, so they can be tested without a transport) ----


def parse_instruments(body: bytes) -> tuple[KiteInstrument, ...]:
    text = body.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    missing = set(INSTRUMENT_COLUMNS) - set(reader.fieldnames or [])
    if missing:
        raise BrokerError(
            f"instruments dump is missing expected column(s): {sorted(missing)}. "
            "Kite may have changed the format; refusing to guess at the mapping."
        )

    instruments = []
    for row in reader:
        try:
            instruments.append(
                KiteInstrument(
                    instrument_token=int(row["instrument_token"]),
                    exchange_token=row["exchange_token"],
                    tradingsymbol=row["tradingsymbol"],
                    name=row["name"],
                    expiry=row["expiry"],
                    strike=float(row["strike"] or 0.0),
                    tick_size=float(row["tick_size"] or 0.0),
                    lot_size=int(float(row["lot_size"] or 0)),
                    instrument_type=row["instrument_type"],
                    segment=row["segment"],
                    exchange=row["exchange"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise BrokerError(
                f"could not parse instruments row {row.get('tradingsymbol')!r}: {exc}"
            ) from exc
    if not instruments:
        raise BrokerError("instruments dump contained no rows")
    return tuple(instruments)


def parse_candles(payload: object) -> tuple[Candle, ...]:
    """``{"candles": [[ts, o, h, l, c, v], ...]}`` -> typed candles.

    Rejects a malformed row rather than skipping it. A silently dropped
    bar becomes a gap in a price series, and a gap produces a plausible
    and wrong backtest rather than an error anyone would notice.
    """
    if not isinstance(payload, dict):
        raise BrokerError(f"expected a candles object, got {type(payload).__name__}")
    rows = payload.get("candles")
    if rows is None:
        raise BrokerError("historical response contained no 'candles' key")
    if not isinstance(rows, list):
        raise BrokerError(f"expected 'candles' to be a list, got {type(rows).__name__}")

    candles = []
    for index, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, str) or len(row) < 6:
            raise BrokerError(f"candle {index} is not a 6-element row: {row!r}")
        try:
            candles.append(
                Candle(
                    timestamp=_parse_timestamp(str(row[0])),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=int(row[5]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise BrokerError(f"could not parse candle {index} ({row!r}): {exc}") from exc
    return tuple(candles)


def _parse_timestamp(raw: str) -> dt.datetime:
    """Kite returns ISO 8601 with a ``+0530`` offset (no colon), which
    ``fromisoformat`` accepts from Python 3.11 onward."""
    try:
        return dt.datetime.fromisoformat(raw)
    except ValueError as exc:
        raise BrokerError(f"unparseable candle timestamp {raw!r}") from exc
