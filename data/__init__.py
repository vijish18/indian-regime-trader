"""Broker-independent Indian market data layer.

Import the abstractions from ``data.interfaces`` and the records from
``data.models``; depend on a concrete implementation only at the composition
root (``main.py`` or a test fixture). Nothing above this package should know
whether data came from a file, a vendor API, or a broker feed
(docs/SPECIFICATION.md section 4).
"""

from data.errors import (
    CalendarCoverageError,
    DataError,
    DataNotAvailableError,
    DataValidationError,
    DuplicateRecordError,
    InstrumentNotFoundError,
)
from data.interfaces import (
    CorporateActionProvider,
    IndexMembershipProvider,
    InstrumentRepository,
    MarketDataProvider,
    TradingCalendar,
)
from data.models import (
    CorporateAction,
    CorporateActionType,
    DailyBar,
    Exchange,
    IndexMembership,
    IndexObservation,
    Instrument,
    InstrumentStatus,
    PriceBasis,
    Quote,
    Segment,
    SessionType,
    TradingSession,
)

__all__ = [
    "CalendarCoverageError",
    "CorporateAction",
    "CorporateActionProvider",
    "CorporateActionType",
    "DailyBar",
    "DataError",
    "DataNotAvailableError",
    "DataValidationError",
    "DuplicateRecordError",
    "Exchange",
    "IndexMembership",
    "IndexMembershipProvider",
    "IndexObservation",
    "Instrument",
    "InstrumentNotFoundError",
    "InstrumentRepository",
    "InstrumentStatus",
    "MarketDataProvider",
    "PriceBasis",
    "Quote",
    "Segment",
    "SessionType",
    "TradingCalendar",
    "TradingSession",
]
