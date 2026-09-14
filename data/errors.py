"""Exception types for the data layer.

The data layer fails closed: when something is unknown or inconsistent it
raises rather than guessing, so that a caller cannot accidentally trade on
data that was silently defaulted (docs/SPECIFICATION.md section 11).
"""

from __future__ import annotations


class DataError(Exception):
    """Base class for every data-layer failure."""


class InstrumentNotFoundError(DataError):
    """No instrument record is effective for the requested id/symbol and date."""


class DataNotAvailableError(DataError):
    """The requested series is absent from local storage."""


class CalendarCoverageError(DataError):
    """A calendar question was asked about a date the holiday dataset does not
    cover.

    Answering it anyway would silently assume "no holidays", which turns a
    missing-data problem into a wrong-data problem: a backtest would place
    trades on dates the exchange was closed.
    """


class DataValidationError(DataError):
    """Ingested data failed validation and was rejected."""


class DuplicateRecordError(DataValidationError):
    """The same logical record appeared more than once in an ingest."""
