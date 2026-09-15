"""Vocabulary translation between Kite Connect's own enums/status strings
and this system's generic ``execution.order_manager.OrderState`` and
``backtest.costs.TradeSide``.

Every enum value and status string here is taken verbatim from Zerodha's
published Kite Connect v3 documentation
(https://kite.trade/docs/connect/v3/orders/ and
https://kite.trade/docs/connect/v3/market-quotes/) -- nothing invented.
"""

from __future__ import annotations

from backtest.costs import TradeSide
from execution.order_manager import OrderState

# docs/connect/v3/orders/ -- "Exchange" enum, quoted verbatim.
KITE_EXCHANGES: frozenset[str] = frozenset({"NSE", "BSE", "NFO", "CDS", "BCD", "MCX"})

# This system trades NSE/BSE cash equity only (V1 scope, docs/SPECIFICATION.md
# section 1) -- Kite supports more, but nothing above this adapter ever
# requests them.
SUPPORTED_EXCHANGES: frozenset[str] = frozenset({"NSE", "BSE"})

# "Product" enum, quoted verbatim. This system's V1 scope (cash equity,
# no leverage, no intraday) only ever *places* CNC -- the others are
# recognized here because a status/report call must not choke on a real
# value Kite can legitimately report (e.g. a position or order that
# predates this system, or was placed manually outside it).
KITE_PRODUCTS: frozenset[str] = frozenset({"CNC", "NRML", "MIS", "MTF"})
SUPPORTED_PRODUCTS: frozenset[str] = frozenset({"CNC"})

# "Order type" enum, quoted verbatim. This adapter only ever places
# LIMIT (docs/SPECIFICATION.md section 12.2/13 -- NSE's retail-algo
# framework does not permit market orders for algo-originated flow;
# enforced structurally by config.models.ExecutionConfig.order_type).
KITE_ORDER_TYPES: frozenset[str] = frozenset({"MARKET", "LIMIT", "SL", "SL-M"})
SUPPORTED_ORDER_TYPES: frozenset[str] = frozenset({"LIMIT"})

# "Validity" enum, quoted verbatim.
KITE_VALIDITIES: frozenset[str] = frozenset({"DAY", "IOC", "TTL"})

# "Variety" enum, quoted verbatim. This adapter only ever places
# "regular" -- the others (amo/co/iceberg/auction) are distinct routes
# with their own rules this system does not use in V1.
KITE_VARIETIES: frozenset[str] = frozenset({"regular", "amo", "co", "iceberg", "auction"})
SUPPORTED_VARIETIES: frozenset[str] = frozenset({"regular"})

_SIDE_TO_KITE = {TradeSide.BUY: "BUY", TradeSide.SELL: "SELL"}
_KITE_TO_SIDE = {value: key for key, value in _SIDE_TO_KITE.items()}


def side_to_kite(side: TradeSide) -> str:
    return _SIDE_TO_KITE[side]


def side_from_kite(value: str) -> TradeSide:
    try:
        return _KITE_TO_SIDE[value]
    except KeyError:
        raise ValueError(f"unrecognized Kite transaction_type: {value!r}") from None


def split_instrument_id(instrument_id: str) -> tuple[str, str]:
    """``"NSE:INFY"`` -> ``("NSE", "INFY")`` -- this project's own
    ``instrument_id`` convention (``data/models.py``, Phase 2-3) already
    matches the "exchange:tradingsymbol" format Kite's own quote and
    instrument keys use (docs/connect/v3/market-quotes/: "NSE:INFY",
    "BSE:SENSEX"), so no translation table is needed, only a split.
    """
    try:
        exchange, tradingsymbol = instrument_id.split(":", 1)
    except ValueError:
        raise ValueError(
            f"instrument_id must be 'EXCHANGE:TRADINGSYMBOL', got {instrument_id!r}"
        ) from None
    return exchange, tradingsymbol


def join_instrument_id(exchange: str, tradingsymbol: str) -> str:
    return f"{exchange}:{tradingsymbol}"


# docs/connect/v3/orders/ -- "Order Status" enum, quoted verbatim:
# COMPLETE, REJECTED, CANCELLED, OPEN, PUT ORDER REQ RECEIVED,
# VALIDATION PENDING, OPEN PENDING, MODIFY VALIDATION PENDING,
# MODIFY PENDING, TRIGGER PENDING, CANCEL PENDING, AMO REQ RECEIVED,
# MODIFIED.
_SUBMITTED_LIKE: frozenset[str] = frozenset(
    {
        "PUT ORDER REQ RECEIVED",
        "VALIDATION PENDING",
        "OPEN PENDING",
        "AMO REQ RECEIVED",
    }
)
_OPEN_LIKE: frozenset[str] = frozenset(
    {
        "OPEN",
        "MODIFY VALIDATION PENDING",
        "MODIFY PENDING",
        "MODIFIED",
        "TRIGGER PENDING",
    }
)


def order_state_from_kite(kite_status: str, filled_quantity: int, quantity: int) -> OrderState:
    """Map one of Kite's documented order-status strings onto this
    system's :class:`~execution.order_manager.OrderState`.

    Kite has no distinct "partially filled" status string of its own --
    an order sitting at ``OPEN`` (or one of the open-like pending states)
    with ``0 < filled_quantity < quantity`` is this adapter's own derived
    ``PARTIALLY_FILLED``, computed from the same ``filled_quantity``/
    ``pending_quantity`` fields Kite's order-book response already
    reports, not a status string Kite sends.

    Any status this function does not recognize maps to ``UNKNOWN``
    rather than raising -- Zerodha adding a new status string in the
    future must not crash this adapter; ``OrderManager`` already has a
    defined, safe way to handle ``UNKNOWN`` (resolve it by re-querying,
    never guess).
    """
    if kite_status == "COMPLETE":
        return OrderState.FILLED
    if kite_status == "REJECTED":
        return OrderState.REJECTED
    if kite_status == "CANCELLED":
        return OrderState.CANCELLED
    if kite_status == "CANCEL PENDING":
        return OrderState.CANCEL_REQUESTED
    if kite_status in _SUBMITTED_LIKE:
        return OrderState.SUBMITTED
    if kite_status in _OPEN_LIKE:
        if 0 < filled_quantity < quantity:
            return OrderState.PARTIALLY_FILLED
        return OrderState.OPEN
    return OrderState.UNKNOWN
