"""Domain data models for the Indian market data layer.

Design notes that matter for correctness:

- **Prices are ``Decimal``, not ``float``.** Tick-size arithmetic, cost
  computation, and price-band checks all round-trip through these values;
  binary floating point silently breaks exact comparisons (a 0.05 tick is not
  representable in binary). Analytics code converts to float at the pandas
  boundary, never the other way round.
- **Bars do not self-validate their OHLC relationships.** A bar must be
  *representable* even when the vendor sent nonsense, otherwise the data
  quality layer cannot detect, report, and quarantine bad vendor data
  (docs/SPECIFICATION.md section 4.1). Structural invariants that indicate a
  *programming* error (a negative tick size, an inverted effective-date range)
  are enforced here; vendor-data problems are reported by ``data.data_quality``.
- **Timestamps are timezone-aware or rejected.** Naive datetimes are the usual
  root cause of session-boundary bugs, so ``Quote`` and ``TradingSession``
  refuse them at construction (docs/SPECIFICATION.md section 14).

Daily bars are keyed by ``session_date`` (a calendar date) rather than a
timestamp: a daily bar belongs to a trading session, not to an instant, and
storing it as an instant invites timezone drift at the date boundary.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

PRICE_PRECISION = Decimal("0.0001")
"""Quantum used when reconstructing prices from storage. NSE equity prices
carry at most 4 decimal places (India VIX uses 4; cash equities use 2)."""


class Exchange(StrEnum):
    NSE = "NSE"
    BSE = "BSE"


class Segment(StrEnum):
    EQUITY = "equity"
    INDEX = "index"


class InstrumentStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELISTED = "delisted"


class SessionType(StrEnum):
    CLOSED = "closed"
    PRE_OPEN = "pre_open"
    REGULAR = "regular"
    SPECIAL = "special"  # e.g. Muhurat trading


class PriceBasis(StrEnum):
    """Whether a bar's prices have been adjusted for corporate actions."""

    RAW = "raw"
    ADJUSTED = "adjusted"


class CorporateActionType(StrEnum):
    SPLIT = "split"
    BONUS = "bonus"
    DIVIDEND = "dividend"
    RIGHTS = "rights"
    MERGER = "merger"
    DEMERGER = "demerger"
    DELISTING = "delisting"


@dataclass(frozen=True, slots=True)
class Instrument:
    """One point-in-time instrument record, valid over
    ``[effective_from, effective_to]`` (``effective_to`` of ``None`` means
    "still in force").

    Reference data changes over time -- symbols get renamed, tick sizes and
    lot sizes change, instruments get suspended and delisted. Every lookup is
    therefore ``as_of``-qualified rather than "current", so that a backtest
    and live trading go through the same code path.
    """

    instrument_id: str
    symbol: str
    exchange: Exchange
    segment: Segment
    tick_size: Decimal
    price_precision: int
    effective_from: dt.date
    effective_to: dt.date | None = None
    isin: str | None = None
    lot_size: int | None = None
    status: InstrumentStatus = InstrumentStatus.ACTIVE
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must not be empty")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")
        if self.price_precision < 0:
            raise ValueError(f"price_precision must be >= 0, got {self.price_precision}")
        if self.lot_size is not None and self.lot_size < 1:
            raise ValueError(f"lot_size must be >= 1 when set, got {self.lot_size}")
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError(
                f"effective_to ({self.effective_to}) precedes "
                f"effective_from ({self.effective_from}) for {self.instrument_id}"
            )

    def is_effective_on(self, as_of: dt.date) -> bool:
        """True if this record is the valid version of the instrument on ``as_of``."""
        if as_of < self.effective_from:
            return False
        return self.effective_to is None or as_of <= self.effective_to

    def is_tradable_on(self, as_of: dt.date) -> bool:
        """True if the instrument is both effective and in ACTIVE status."""
        return self.is_effective_on(as_of) and self.status is InstrumentStatus.ACTIVE

    def round_to_tick(self, price: Decimal) -> Decimal:
        """Round ``price`` to the nearest valid tick multiple.

        Uses exact decimal arithmetic on the tick multiple; order placement
        must never send a price the exchange will reject for violating tick
        size (docs/SPECIFICATION.md section 12.2).
        """
        ticks = (price / self.tick_size).to_integral_value(rounding=ROUND_HALF_UP)
        return (ticks * self.tick_size).quantize(
            Decimal(1).scaleb(-self.price_precision)
        )


@dataclass(frozen=True, slots=True)
class DailyBar:
    """One daily OHLCV observation for one instrument.

    Deliberately permissive about its own contents -- see the module docstring.
    ``data.data_quality`` decides whether a bar is usable.
    """

    instrument_id: str
    session_date: dt.date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    price_basis: PriceBasis = PriceBasis.RAW
    adjustment_factor: Decimal = Decimal(1)
    data_source: str = "unknown"

    def has_valid_ohlc(self) -> bool:
        """True if the OHLC relationships are internally consistent."""
        return (
            self.low <= self.high
            and self.low <= self.open <= self.high
            and self.low <= self.close <= self.high
        )

    def has_positive_prices(self) -> bool:
        return min(self.open, self.high, self.low, self.close) > 0

    def adjusted(self, factor: Decimal) -> DailyBar:
        """Return a copy with prices scaled by ``factor`` (corporate-action
        back-adjustment). Volume is scaled inversely so that traded value is
        preserved.
        """
        if factor <= 0:
            raise ValueError(f"adjustment factor must be positive, got {factor}")
        return DailyBar(
            instrument_id=self.instrument_id,
            session_date=self.session_date,
            open=(self.open * factor).quantize(PRICE_PRECISION),
            high=(self.high * factor).quantize(PRICE_PRECISION),
            low=(self.low * factor).quantize(PRICE_PRECISION),
            close=(self.close * factor).quantize(PRICE_PRECISION),
            volume=int(Decimal(self.volume) / factor) if factor != 1 else self.volume,
            price_basis=PriceBasis.ADJUSTED,
            adjustment_factor=self.adjustment_factor * factor,
            data_source=self.data_source,
        )


@dataclass(frozen=True, slots=True)
class Quote:
    """A point-in-time bid/ask snapshot, used only for live/paper execution
    checks (spread, staleness). Never used by the backtester.
    """

    instrument_id: str
    bid: Decimal
    ask: Decimal
    last_price: Decimal
    as_of: dt.datetime
    bid_quantity: int | None = None
    ask_quantity: int | None = None

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError(
                f"Quote.as_of must be timezone-aware, got naive {self.as_of!r}"
            )

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> Decimal:
        """Bid-ask spread in basis points of the mid price."""
        mid = self.mid
        if mid <= 0:
            raise ValueError(f"cannot compute spread on non-positive mid {mid}")
        return (self.ask - self.bid) / mid * Decimal(10_000)

    def is_crossed(self) -> bool:
        """True if bid exceeds ask -- a broken or stale book."""
        return self.bid > self.ask

    def age_seconds(self, now: dt.datetime) -> float:
        if now.tzinfo is None:
            raise ValueError(f"'now' must be timezone-aware, got naive {now!r}")
        return (now - self.as_of).total_seconds()

    def is_stale(self, now: dt.datetime, max_age_seconds: float) -> bool:
        return self.age_seconds(now) > max_age_seconds


@dataclass(frozen=True, slots=True)
class TradingSession:
    """One exchange session. Session boundaries come from the calendar, never
    from wall-clock assumptions (docs/SPECIFICATION.md section 14).
    """

    session_date: dt.date
    session_type: SessionType
    regular_open: dt.datetime
    regular_close: dt.datetime
    pre_open_start: dt.datetime | None = None
    pre_open_end: dt.datetime | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("regular_open", self.regular_open),
            ("regular_close", self.regular_close),
            ("pre_open_start", self.pre_open_start),
            ("pre_open_end", self.pre_open_end),
        ):
            if value is not None and value.tzinfo is None:
                raise ValueError(
                    f"TradingSession.{name} must be timezone-aware, got naive {value!r}"
                )
        if self.regular_close <= self.regular_open:
            raise ValueError(
                f"session {self.session_date}: close {self.regular_close} "
                f"does not follow open {self.regular_open}"
            )

    @property
    def is_trading_day(self) -> bool:
        return self.session_type in (SessionType.REGULAR, SessionType.SPECIAL)

    def contains(self, moment: dt.datetime) -> bool:
        """True if ``moment`` falls inside the regular session."""
        if moment.tzinfo is None:
            raise ValueError(f"'moment' must be timezone-aware, got naive {moment!r}")
        return self.regular_open <= moment <= self.regular_close

    def type_at(self, moment: dt.datetime) -> SessionType:
        """The session phase in force at ``moment``."""
        if moment.tzinfo is None:
            raise ValueError(f"'moment' must be timezone-aware, got naive {moment!r}")
        if self.pre_open_start is not None and self.pre_open_end is not None:
            if self.pre_open_start <= moment < self.pre_open_end:
                return SessionType.PRE_OPEN
        if self.contains(moment):
            return self.session_type
        return SessionType.CLOSED


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """A corporate action and the terms needed to adjust prices for it.

    Ratio semantics are explicit because split and bonus ratios are quoted
    differently in Indian market data and conflating them produces a silently
    wrong adjustment factor:

    - **Split** ``ratio_new:ratio_old`` means one old share becomes
      ``ratio_new/ratio_old`` shares. A 5-for-1 split is ``ratio_new=5,
      ratio_old=1`` and scales historical prices by ``1/5``.
    - **Bonus** ``ratio_new:ratio_old`` means ``ratio_new`` free shares are
      issued for every ``ratio_old`` held. A 1:1 bonus is ``ratio_new=1,
      ratio_old=1`` and scales historical prices by ``1/2``.
    - **Dividend** is a cash event: it carries ``cash_amount`` and does not
      adjust the price series, because the backtester credits dividends to
      cash instead (docs/ARCHITECTURE.md, total-return convention).
    - **Rights, merger and demerger** have no closed-form factor computable
      from the action alone, so they require ``explicit_price_factor`` supplied
      by the data vendor or operations.
    """

    instrument_id: str
    action_type: CorporateActionType
    ex_date: dt.date
    ratio_new: Decimal | None = None
    ratio_old: Decimal | None = None
    cash_amount: Decimal | None = None
    explicit_price_factor: Decimal | None = None
    record_date: dt.date | None = None
    announcement_date: dt.date | None = None
    successor_instrument_id: str | None = None
    adjustment_version: int = 1

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must not be empty")
        for name, value in (("ratio_new", self.ratio_new), ("ratio_old", self.ratio_old)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")
        if self.explicit_price_factor is not None and self.explicit_price_factor <= 0:
            raise ValueError(
                f"explicit_price_factor must be positive, got {self.explicit_price_factor}"
            )
        if self.announcement_date is not None and self.announcement_date > self.ex_date:
            raise ValueError(
                f"{self.instrument_id}: announcement_date {self.announcement_date} "
                f"is after ex_date {self.ex_date}"
            )

    @property
    def is_identity_change(self) -> bool:
        """True for actions that end or replace the instrument, which position
        tracking must handle explicitly rather than by price adjustment.
        """
        return self.action_type in (
            CorporateActionType.MERGER,
            CorporateActionType.DEMERGER,
            CorporateActionType.DELISTING,
        )

    def price_adjustment_factor(self) -> Decimal:
        """Multiplier applied to prices *before* ``ex_date`` to make them
        comparable with prices on and after it.

        Raises:
            ValueError: if the action's terms are incomplete for its type.
        """
        if self.explicit_price_factor is not None:
            return self.explicit_price_factor

        match self.action_type:
            case CorporateActionType.SPLIT:
                if self.ratio_new is None or self.ratio_old is None:
                    raise ValueError(
                        f"{self.instrument_id} split on {self.ex_date} is missing "
                        "ratio_new/ratio_old"
                    )
                return self.ratio_old / self.ratio_new
            case CorporateActionType.BONUS:
                if self.ratio_new is None or self.ratio_old is None:
                    raise ValueError(
                        f"{self.instrument_id} bonus on {self.ex_date} is missing "
                        "ratio_new/ratio_old"
                    )
                return self.ratio_old / (self.ratio_old + self.ratio_new)
            case CorporateActionType.DIVIDEND | CorporateActionType.DELISTING:
                return Decimal(1)
            case _:
                raise ValueError(
                    f"{self.instrument_id} {self.action_type} on {self.ex_date} "
                    "requires an explicit_price_factor; it cannot be derived from "
                    "the action terms alone"
                )


@dataclass(frozen=True, slots=True)
class IndexObservation:
    """A daily index observation (NIFTY 50, India VIX).

    Only ``close`` is required: India VIX is published as a level and some
    vendors supply no OHLC for it (docs/SPECIFICATION.md section 4).
    """

    index_symbol: str
    session_date: dt.date
    close: Decimal
    open: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    volume: int | None = None
    data_source: str = "unknown"

    def has_valid_ohlc(self) -> bool:
        """True if any supplied OHLC values are internally consistent."""
        if self.open is None or self.high is None or self.low is None:
            return True  # nothing to contradict
        return (
            self.low <= self.high
            and self.low <= self.open <= self.high
            and self.low <= self.close <= self.high
        )


@dataclass(frozen=True, slots=True)
class IndexMembership:
    """Point-in-time index constituency.

    Constituents are never hardcoded: this record is loaded from a maintained
    membership dataset so that a backtest on any historical date sees the
    index as it actually was, not as it is today (docs/SPECIFICATION.md
    section 2.1 -- the primary survivorship-bias control).
    """

    index_symbol: str
    instrument_id: str
    effective_from: dt.date
    effective_to: dt.date | None = None
    inclusion_reason: str | None = None
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError(
                f"{self.index_symbol}/{self.instrument_id}: effective_to "
                f"({self.effective_to}) precedes effective_from ({self.effective_from})"
            )

    def is_member_on(self, as_of: dt.date) -> bool:
        if as_of < self.effective_from:
            return False
        return self.effective_to is None or as_of <= self.effective_to


@dataclass(frozen=True, slots=True)
class DataSnapshot:
    """Identifies exactly which data a decision was made from, so a historical
    run can be reproduced (docs/SPECIFICATION.md section 4.1).
    """

    snapshot_id: str
    created_at: dt.datetime
    instrument_count: int
    bar_count: int
    sources: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("DataSnapshot.created_at must be timezone-aware")
