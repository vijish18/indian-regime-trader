"""Broker-independent data interfaces.

Everything above the data layer -- features, regime, selection, portfolio,
risk, backtest -- depends only on these abstractions, never on a vendor SDK,
a broker API, or a file format. That is what lets the same strategy code run
against local historical files today and a live feed later without edits, and
what lets the backtester and live system share one code path.

Two rules every implementation must honor:

1. **Point-in-time.** Every reference-data lookup takes an explicit ``as_of``
   date and must answer as of that date, never "as of today". This is the
   survivorship-bias control (docs/SPECIFICATION.md section 2.1).
2. **Fail closed.** Unknown instruments, uncovered calendar ranges and absent
   series raise; they never return an empty or defaulted result that a caller
   could mistake for a valid answer.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from decimal import Decimal

from data.models import (
    CorporateAction,
    DailyBar,
    IndexMembership,
    IndexObservation,
    Instrument,
    PriceBasis,
    Quote,
    Segment,
    TradingSession,
)


class TradingCalendar(ABC):
    """Authoritative answer to "was/is the exchange open, and when".

    Strategy-facing timestamps are Asia/Kolkata; persisted timestamps are UTC
    (docs/SPECIFICATION.md section 14).
    """

    @abstractmethod
    def is_trading_day(self, day: dt.date) -> bool:
        """True if the exchange trades on ``day`` (regular or special session)."""

    @abstractmethod
    def session(self, day: dt.date) -> TradingSession:
        """The session for ``day``, including a CLOSED session for non-trading days.

        Raises:
            CalendarCoverageError: if ``day`` falls outside the calendar's
                known coverage range.
        """

    @abstractmethod
    def sessions_between(self, start: dt.date, end: dt.date) -> list[TradingSession]:
        """All trading sessions in ``[start, end]``, ascending. Non-trading
        days are omitted.
        """

    @abstractmethod
    def trading_days_between(self, start: dt.date, end: dt.date) -> list[dt.date]:
        """All trading dates in ``[start, end]``, ascending."""

    @abstractmethod
    def next_trading_day(self, day: dt.date) -> dt.date:
        """The first trading date strictly after ``day``."""

    @abstractmethod
    def previous_trading_day(self, day: dt.date) -> dt.date:
        """The last trading date strictly before ``day``."""

    @abstractmethod
    def sessions_offset(self, day: dt.date, sessions: int) -> dt.date:
        """The trading date ``sessions`` sessions away from ``day``.

        Used for next-session execution timing (``sessions=1``) and for
        lookback windows (negative values), so that "t+1" always means the
        next *session*, never the next calendar day.
        """


class InstrumentRepository(ABC):
    """Point-in-time instrument reference data."""

    @abstractmethod
    def get(self, instrument_id: str, as_of: dt.date) -> Instrument:
        """The instrument record effective on ``as_of``.

        Raises:
            InstrumentNotFoundError: if no record is effective on that date.
        """

    @abstractmethod
    def get_by_symbol(self, symbol: str, exchange: str, as_of: dt.date) -> Instrument:
        """Look up by trading symbol, which is only unique *at a point in time*
        -- symbols are reassigned after delisting, so the ``as_of`` date is
        part of the key, not a convenience.
        """

    @abstractmethod
    def list_effective(
        self, as_of: dt.date, segment: Segment | None = None
    ) -> list[Instrument]:
        """Every instrument record effective on ``as_of``, optionally filtered
        by segment.
        """

    @abstractmethod
    def history(self, instrument_id: str) -> list[Instrument]:
        """Every version of one instrument, ascending by ``effective_from``."""

    @abstractmethod
    def snapshot_date(self) -> dt.date | None:
        """When the underlying instrument master was last refreshed, for the
        startup freshness check (docs/SPECIFICATION.md section 15.1 step 6).
        """


class CorporateActionProvider(ABC):
    """Corporate actions and the price adjustments derived from them."""

    @abstractmethod
    def actions_for(
        self, instrument_id: str, start: dt.date, end: dt.date
    ) -> list[CorporateAction]:
        """Actions with ``ex_date`` in ``[start, end]``, ascending."""

    @abstractmethod
    def cumulative_adjustment_factor(
        self, instrument_id: str, price_date: dt.date, as_of: dt.date
    ) -> Decimal:
        """Factor that makes a price observed on ``price_date`` comparable with
        prices as of ``as_of``.

        Implementations must use only actions with ``ex_date <= as_of``.
        Applying an action that had not yet gone ex as of the decision date is
        a look-ahead leak: it rewrites history with information that did not
        exist yet.
        """

    @abstractmethod
    def identity_change(
        self, instrument_id: str, as_of: dt.date
    ) -> CorporateAction | None:
        """The first merger/demerger/delisting on or after ``as_of``, if any.

        Position tracking must handle these explicitly -- a held position whose
        instrument ceases to exist cannot be resolved by price adjustment.
        """


class MarketDataProvider(ABC):
    """Historical and (later) live market data, independent of any vendor.

    The local implementation serves history from files; a live implementation
    added in a later phase serves quotes from a broker feed behind this same
    interface. No caller distinguishes between them.
    """

    @abstractmethod
    def get_equity_bars(
        self,
        instrument_id: str,
        start: dt.date,
        end: dt.date,
        price_basis: PriceBasis = PriceBasis.RAW,
    ) -> list[DailyBar]:
        """Daily bars for one equity in ``[start, end]``, ascending.

        ``price_basis=ADJUSTED`` returns corporate-action-adjusted prices
        computed as of ``end`` -- so a walk-forward fold that ends on date T
        never sees an adjustment announced after T.
        """

    @abstractmethod
    def get_index_observations(
        self, index_symbol: str, start: dt.date, end: dt.date
    ) -> list[IndexObservation]:
        """Daily observations for an index (NIFTY 50, India VIX), ascending."""

    @abstractmethod
    def available_range(self, instrument_id: str) -> tuple[dt.date, dt.date] | None:
        """First and last session available locally, or ``None`` if absent."""

    @abstractmethod
    def get_quote(self, instrument_id: str) -> Quote:
        """Latest live quote, for execution-time spread and staleness checks.

        Historical-only implementations raise ``DataNotAvailableError``: a
        backtest must never consult a live book.
        """


class IndexMembershipProvider(ABC):
    """Point-in-time index constituency -- the input to a survivorship-free
    universe.

    Implementations must load membership from maintained data. Hardcoding
    today's constituents and applying them to history is exactly the
    survivorship bias this interface exists to prevent.
    """

    @abstractmethod
    def members_on(self, index_symbol: str, as_of: dt.date) -> frozenset[str]:
        """Instrument ids that were index constituents on ``as_of``."""

    @abstractmethod
    def membership_history(self, index_symbol: str) -> list[IndexMembership]:
        """Every membership record for the index, ascending by ``effective_from``."""
