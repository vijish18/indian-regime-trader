"""Data-quality validation.

Validation is separate from parsing on purpose. Parsing must be able to
represent whatever the vendor actually sent -- including impossible bars --
so that this layer can *detect and report* the problem instead of a parser
silently dropping the row and leaving a hole nobody notices
(docs/SPECIFICATION.md section 4.1).

Every check returns a report rather than raising, so one pass surfaces all
problems in a file instead of stopping at the first. Callers that must fail
closed call :meth:`DataQualityReport.raise_if_errors`.

Severity is the difference between "this data is wrong" and "this data is
suspicious":

- **ERROR** -- internally impossible, and the series cannot be trusted:
  duplicate bars, ``high < low``, non-positive prices, a bar on a date the
  exchange was closed.
- **WARNING** -- plausible but worth a human look: a zero-volume session, a
  run of identical closes, a volume spike far outside the local distribution.
"""

from __future__ import annotations

import datetime as dt
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from data.errors import CalendarCoverageError, DataValidationError
from data.interfaces import TradingCalendar
from data.models import (
    CorporateAction,
    DailyBar,
    IndexObservation,
    Instrument,
)

DEFAULT_STALE_RUN_SESSIONS = 5
DEFAULT_VOLUME_SPIKE_MULTIPLE = Decimal(20)


class Severity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


class IssueCode(StrEnum):
    # Bars
    DUPLICATE_BAR = "duplicate_bar"
    IMPOSSIBLE_OHLC = "impossible_ohlc"
    NON_POSITIVE_PRICE = "non_positive_price"
    NEGATIVE_VOLUME = "negative_volume"
    ZERO_VOLUME_SESSION = "zero_volume_session"
    ZERO_VOLUME_WITH_PRICE_MOVE = "zero_volume_with_price_move"
    MISSING_SESSION = "missing_session"
    NON_TRADING_DATE = "non_trading_date"
    CALENDAR_COVERAGE_UNKNOWN = "calendar_coverage_unknown"
    FUTURE_DATE = "future_date"
    STALE_PRICE = "stale_price"
    SUSPICIOUS_VOLUME_SPIKE = "suspicious_volume_spike"
    MIXED_INSTRUMENTS = "mixed_instruments"
    # Reference data
    DUPLICATE_INSTRUMENT = "duplicate_instrument"
    OVERLAPPING_EFFECTIVE_DATES = "overlapping_effective_dates"
    DUPLICATE_ISIN = "duplicate_isin"
    # Corporate actions
    DUPLICATE_CORPORATE_ACTION = "duplicate_corporate_action"
    UNDERIVABLE_ADJUSTMENT = "underivable_adjustment"
    ACTION_ON_NON_TRADING_DATE = "action_on_non_trading_date"
    ACTION_BEFORE_LISTING = "action_before_listing"


@dataclass(frozen=True, slots=True)
class DataQualityIssue:
    code: IssueCode
    severity: Severity
    detail: str
    instrument_id: str | None = None
    session_date: dt.date | None = None

    def __str__(self) -> str:
        where = " ".join(
            part
            for part in (self.instrument_id, self.session_date and str(self.session_date))
            if part
        )
        return f"[{self.severity}] {self.code}: {self.detail}" + (f" ({where})" if where else "")


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """The outcome of validating one dataset."""

    checked_rows: int
    issues: tuple[DataQualityIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[DataQualityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[DataQualityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.WARNING)

    @property
    def is_clean(self) -> bool:
        """True if nothing at all was flagged."""
        return not self.issues

    @property
    def is_usable(self) -> bool:
        """True if no ERROR-severity issue was found; warnings are tolerated."""
        return not self.errors

    def codes(self) -> frozenset[IssueCode]:
        return frozenset(issue.code for issue in self.issues)

    def raise_if_errors(self, context: str) -> None:
        """Fail closed on ERROR-severity issues.

        Raises:
            DataValidationError: listing every error found.
        """
        errors = self.errors
        if errors:
            detail = "; ".join(str(issue) for issue in errors)
            raise DataValidationError(f"{context}: {len(errors)} error(s): {detail}")

    def merged_with(self, other: DataQualityReport) -> DataQualityReport:
        return DataQualityReport(
            checked_rows=self.checked_rows + other.checked_rows,
            issues=self.issues + other.issues,
        )


class BarValidator:
    """Validates daily bar series against structural rules and the calendar."""

    def __init__(
        self,
        calendar: TradingCalendar | None = None,
        *,
        stale_run_sessions: int = DEFAULT_STALE_RUN_SESSIONS,
        volume_spike_multiple: Decimal = DEFAULT_VOLUME_SPIKE_MULTIPLE,
    ) -> None:
        self._calendar = calendar
        self._stale_run_sessions = stale_run_sessions
        self._volume_spike_multiple = volume_spike_multiple

    def validate(
        self,
        bars: Sequence[DailyBar],
        *,
        instrument_id: str | None = None,
        expected_start: dt.date | None = None,
        expected_end: dt.date | None = None,
        today: dt.date | None = None,
    ) -> DataQualityReport:
        """Check one instrument's bar series.

        ``expected_start``/``expected_end`` bound the window checked for
        missing sessions; without them, the series' own first and last dates
        are used, so an entirely absent tail is not detectable here.
        """
        issues: list[DataQualityIssue] = []
        issues.extend(self._check_single_instrument(bars, instrument_id))
        issues.extend(self._check_duplicates(bars))
        issues.extend(self._check_rows(bars, today=today))
        issues.extend(
            self._check_calendar_alignment(bars, expected_start, expected_end)
        )
        issues.extend(self._check_stale_runs(bars))
        issues.extend(self._check_volume_spikes(bars))
        return DataQualityReport(checked_rows=len(bars), issues=tuple(issues))

    def _check_single_instrument(
        self, bars: Sequence[DailyBar], instrument_id: str | None
    ) -> list[DataQualityIssue]:
        found = {bar.instrument_id for bar in bars}
        if len(found) > 1:
            return [
                DataQualityIssue(
                    code=IssueCode.MIXED_INSTRUMENTS,
                    severity=Severity.ERROR,
                    detail=f"series contains multiple instrument ids: {sorted(found)}",
                )
            ]
        if instrument_id is not None and found and instrument_id not in found:
            return [
                DataQualityIssue(
                    code=IssueCode.MIXED_INSTRUMENTS,
                    severity=Severity.ERROR,
                    detail=(
                        f"series is for {sorted(found)[0]!r} but was loaded as "
                        f"{instrument_id!r}"
                    ),
                )
            ]
        return []

    def _check_duplicates(self, bars: Sequence[DailyBar]) -> list[DataQualityIssue]:
        counts = Counter(bar.session_date for bar in bars)
        return [
            DataQualityIssue(
                code=IssueCode.DUPLICATE_BAR,
                severity=Severity.ERROR,
                detail=f"{count} bars for the same session",
                instrument_id=_first_instrument(bars),
                session_date=session_date,
            )
            for session_date, count in sorted(counts.items())
            if count > 1
        ]

    def _check_rows(
        self, bars: Sequence[DailyBar], today: dt.date | None
    ) -> list[DataQualityIssue]:
        issues: list[DataQualityIssue] = []
        for bar in bars:
            if not bar.has_positive_prices():
                issues.append(
                    DataQualityIssue(
                        code=IssueCode.NON_POSITIVE_PRICE,
                        severity=Severity.ERROR,
                        detail=(
                            f"O={bar.open} H={bar.high} L={bar.low} C={bar.close} "
                            "contains a non-positive price"
                        ),
                        instrument_id=bar.instrument_id,
                        session_date=bar.session_date,
                    )
                )
            elif not bar.has_valid_ohlc():
                issues.append(
                    DataQualityIssue(
                        code=IssueCode.IMPOSSIBLE_OHLC,
                        severity=Severity.ERROR,
                        detail=(
                            f"O={bar.open} H={bar.high} L={bar.low} C={bar.close} "
                            "violates low <= open/close <= high"
                        ),
                        instrument_id=bar.instrument_id,
                        session_date=bar.session_date,
                    )
                )
            if bar.volume < 0:
                issues.append(
                    DataQualityIssue(
                        code=IssueCode.NEGATIVE_VOLUME,
                        severity=Severity.ERROR,
                        detail=f"volume {bar.volume} is negative",
                        instrument_id=bar.instrument_id,
                        session_date=bar.session_date,
                    )
                )
            elif bar.volume == 0:
                issues.append(self._classify_zero_volume(bar))
            if today is not None and bar.session_date > today:
                issues.append(
                    DataQualityIssue(
                        code=IssueCode.FUTURE_DATE,
                        severity=Severity.ERROR,
                        detail=f"bar dated after {today}",
                        instrument_id=bar.instrument_id,
                        session_date=bar.session_date,
                    )
                )
        return issues

    def _classify_zero_volume(self, bar: DailyBar) -> DataQualityIssue:
        """Distinguish a genuine no-trade session from corrupt data.

        Zero volume with a flat bar is plausible (nothing traded). Zero volume
        with a price range is not: a price cannot move without a trade, so the
        row is corrupt or the volume field is missing rather than zero
        (docs/SPECIFICATION.md section 4.1).
        """
        moved = not (bar.open == bar.high == bar.low == bar.close)
        if moved:
            return DataQualityIssue(
                code=IssueCode.ZERO_VOLUME_WITH_PRICE_MOVE,
                severity=Severity.ERROR,
                detail=(
                    f"zero volume but price moved (O={bar.open} H={bar.high} "
                    f"L={bar.low} C={bar.close}); volume is missing, not zero"
                ),
                instrument_id=bar.instrument_id,
                session_date=bar.session_date,
            )
        return DataQualityIssue(
            code=IssueCode.ZERO_VOLUME_SESSION,
            severity=Severity.WARNING,
            detail="zero volume with a flat bar; plausible no-trade session",
            instrument_id=bar.instrument_id,
            session_date=bar.session_date,
        )

    def _check_calendar_alignment(
        self,
        bars: Sequence[DailyBar],
        expected_start: dt.date | None,
        expected_end: dt.date | None,
    ) -> list[DataQualityIssue]:
        if self._calendar is None or not bars:
            return []
        issues: list[DataQualityIssue] = []
        present = {bar.session_date for bar in bars}
        instrument_id = _first_instrument(bars)

        for bar in bars:
            try:
                open_that_day = self._calendar.is_trading_day(bar.session_date)
            except CalendarCoverageError as exc:
                issues.append(
                    _coverage_issue(exc, bar.instrument_id, bar.session_date)
                )
                continue
            if not open_that_day:
                issues.append(
                    DataQualityIssue(
                        code=IssueCode.NON_TRADING_DATE,
                        severity=Severity.ERROR,
                        detail="bar on a date the exchange was closed",
                        instrument_id=bar.instrument_id,
                        session_date=bar.session_date,
                    )
                )

        start = expected_start if expected_start is not None else min(present)
        end = expected_end if expected_end is not None else max(present)
        try:
            expected_sessions = self._calendar.trading_days_between(start, end)
        except CalendarCoverageError as exc:
            issues.append(_coverage_issue(exc, instrument_id, None))
            return issues
        issues.extend(
            DataQualityIssue(
                code=IssueCode.MISSING_SESSION,
                severity=Severity.ERROR,
                detail="exchange was open but no bar was supplied",
                instrument_id=instrument_id,
                session_date=session_date,
            )
            for session_date in expected_sessions
            if session_date not in present
        )
        return issues

    def _check_stale_runs(self, bars: Sequence[DailyBar]) -> list[DataQualityIssue]:
        """Flag runs of identical closes, which usually mean a vendor repeated
        the last known price through a gap rather than reporting no data.
        """
        if len(bars) < self._stale_run_sessions:
            return []
        ordered = sorted(bars, key=lambda bar: bar.session_date)
        issues: list[DataQualityIssue] = []
        run_start = 0
        for index in range(1, len(ordered) + 1):
            same = index < len(ordered) and ordered[index].close == ordered[run_start].close
            if same:
                continue
            run_length = index - run_start
            if run_length >= self._stale_run_sessions:
                issues.append(
                    DataQualityIssue(
                        code=IssueCode.STALE_PRICE,
                        severity=Severity.WARNING,
                        detail=(
                            f"close {ordered[run_start].close} repeated for "
                            f"{run_length} consecutive sessions from "
                            f"{ordered[run_start].session_date}"
                        ),
                        instrument_id=ordered[run_start].instrument_id,
                        session_date=ordered[run_start].session_date,
                    )
                )
            run_start = index
        return issues

    def _check_volume_spikes(self, bars: Sequence[DailyBar]) -> list[DataQualityIssue]:
        volumes = [bar.volume for bar in bars if bar.volume > 0]
        if len(volumes) < self._stale_run_sessions:
            return []
        median = Decimal(str(statistics.median(volumes)))
        if median <= 0:
            return []
        threshold = median * self._volume_spike_multiple
        return [
            DataQualityIssue(
                code=IssueCode.SUSPICIOUS_VOLUME_SPIKE,
                severity=Severity.WARNING,
                detail=(
                    f"volume {bar.volume} exceeds {self._volume_spike_multiple}x the "
                    f"series median ({median})"
                ),
                instrument_id=bar.instrument_id,
                session_date=bar.session_date,
            )
            for bar in bars
            if Decimal(bar.volume) > threshold
        ]


class IndexObservationValidator:
    """Validates index series (NIFTY 50, India VIX)."""

    def __init__(self, calendar: TradingCalendar | None = None) -> None:
        self._calendar = calendar

    def validate(
        self,
        observations: Sequence[IndexObservation],
        *,
        expected_start: dt.date | None = None,
        expected_end: dt.date | None = None,
    ) -> DataQualityReport:
        issues: list[DataQualityIssue] = []
        counts = Counter(observation.session_date for observation in observations)
        issues.extend(
            DataQualityIssue(
                code=IssueCode.DUPLICATE_BAR,
                severity=Severity.ERROR,
                detail=f"{count} observations for the same session",
                session_date=session_date,
            )
            for session_date, count in sorted(counts.items())
            if count > 1
        )
        for observation in observations:
            if observation.close <= 0:
                issues.append(
                    DataQualityIssue(
                        code=IssueCode.NON_POSITIVE_PRICE,
                        severity=Severity.ERROR,
                        detail=f"close {observation.close} is not positive",
                        instrument_id=observation.index_symbol,
                        session_date=observation.session_date,
                    )
                )
            elif not observation.has_valid_ohlc():
                issues.append(
                    DataQualityIssue(
                        code=IssueCode.IMPOSSIBLE_OHLC,
                        severity=Severity.ERROR,
                        detail=(
                            f"O={observation.open} H={observation.high} "
                            f"L={observation.low} C={observation.close} is inconsistent"
                        ),
                        instrument_id=observation.index_symbol,
                        session_date=observation.session_date,
                    )
                )
        if self._calendar is not None and observations:
            present = {observation.session_date for observation in observations}
            symbol = observations[0].index_symbol
            for observation in observations:
                try:
                    open_that_day = self._calendar.is_trading_day(observation.session_date)
                except CalendarCoverageError as exc:
                    issues.append(
                        _coverage_issue(
                            exc, observation.index_symbol, observation.session_date
                        )
                    )
                    continue
                if not open_that_day:
                    issues.append(
                        DataQualityIssue(
                            code=IssueCode.NON_TRADING_DATE,
                            severity=Severity.ERROR,
                            detail="observation on a date the exchange was closed",
                            instrument_id=observation.index_symbol,
                            session_date=observation.session_date,
                        )
                    )
            start = expected_start if expected_start is not None else min(present)
            end = expected_end if expected_end is not None else max(present)
            try:
                expected_sessions = self._calendar.trading_days_between(start, end)
            except CalendarCoverageError as exc:
                expected_sessions = []
                issues.append(_coverage_issue(exc, symbol, None))
            issues.extend(
                DataQualityIssue(
                    code=IssueCode.MISSING_SESSION,
                    severity=Severity.ERROR,
                    detail="exchange was open but no index observation was supplied",
                    instrument_id=symbol,
                    session_date=session_date,
                )
                for session_date in expected_sessions
                if session_date not in present
            )
        return DataQualityReport(
            checked_rows=len(observations), issues=tuple(issues)
        )


def validate_instruments(instruments: Iterable[Instrument]) -> DataQualityReport:
    """Check an instrument master for duplicate and overlapping records."""
    records = list(instruments)
    issues: list[DataQualityIssue] = []

    exact = Counter(
        (record.instrument_id, record.effective_from) for record in records
    )
    issues.extend(
        DataQualityIssue(
            code=IssueCode.DUPLICATE_INSTRUMENT,
            severity=Severity.ERROR,
            detail=f"{count} records with effective_from {effective_from}",
            instrument_id=instrument_id,
        )
        for (instrument_id, effective_from), count in sorted(exact.items())
        if count > 1
    )

    by_id: dict[str, list[Instrument]] = {}
    for record in records:
        by_id.setdefault(record.instrument_id, []).append(record)
    for instrument_id, versions in sorted(by_id.items()):
        ordered = sorted(versions, key=lambda record: record.effective_from)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            overlaps = earlier.effective_to is None or later.effective_from <= earlier.effective_to
            if overlaps:
                issues.append(
                    DataQualityIssue(
                        code=IssueCode.OVERLAPPING_EFFECTIVE_DATES,
                        severity=Severity.ERROR,
                        detail=(
                            f"record from {earlier.effective_from} (to "
                            f"{earlier.effective_to or 'open'}) overlaps record from "
                            f"{later.effective_from}"
                        ),
                        instrument_id=instrument_id,
                    )
                )

    isin_owners: dict[str, set[str]] = {}
    for record in records:
        if record.isin:
            isin_owners.setdefault(record.isin, set()).add(record.instrument_id)
    issues.extend(
        DataQualityIssue(
            code=IssueCode.DUPLICATE_ISIN,
            severity=Severity.ERROR,
            detail=f"ISIN {isin} is claimed by {sorted(owners)}",
        )
        for isin, owners in sorted(isin_owners.items())
        if len(owners) > 1
    )

    return DataQualityReport(checked_rows=len(records), issues=tuple(issues))


def validate_corporate_actions(
    actions: Iterable[CorporateAction],
    calendar: TradingCalendar | None = None,
    instruments: InstrumentLookup | None = None,
) -> DataQualityReport:
    """Check corporate actions for duplicates, underivable adjustments, and
    dates that do not line up with the calendar or the instrument's listing.
    """
    records = list(actions)
    issues: list[DataQualityIssue] = []

    counts = Counter(
        (record.instrument_id, record.action_type, record.ex_date) for record in records
    )
    issues.extend(
        DataQualityIssue(
            code=IssueCode.DUPLICATE_CORPORATE_ACTION,
            severity=Severity.ERROR,
            detail=f"{count} {action_type} actions on the same ex-date",
            instrument_id=instrument_id,
            session_date=ex_date,
        )
        for (instrument_id, action_type, ex_date), count in sorted(counts.items())
        if count > 1
    )

    for record in records:
        try:
            record.price_adjustment_factor()
        except ValueError as exc:
            issues.append(
                DataQualityIssue(
                    code=IssueCode.UNDERIVABLE_ADJUSTMENT,
                    severity=Severity.ERROR,
                    detail=str(exc),
                    instrument_id=record.instrument_id,
                    session_date=record.ex_date,
                )
            )
        if calendar is not None:
            try:
                open_that_day = calendar.is_trading_day(record.ex_date)
            except CalendarCoverageError as exc:
                issues.append(
                    _coverage_issue(exc, record.instrument_id, record.ex_date)
                )
            else:
                if not open_that_day:
                    issues.append(
                        DataQualityIssue(
                            code=IssueCode.ACTION_ON_NON_TRADING_DATE,
                            severity=Severity.WARNING,
                            detail=(
                                "ex-date falls on a non-trading day; the effective "
                                "session is the next trading day"
                            ),
                            instrument_id=record.instrument_id,
                            session_date=record.ex_date,
                        )
                    )
        if instruments is not None:
            listing = instruments.listed_from(record.instrument_id)
            if listing is not None and record.ex_date < listing:
                issues.append(
                    DataQualityIssue(
                        code=IssueCode.ACTION_BEFORE_LISTING,
                        severity=Severity.ERROR,
                        detail=f"ex-date precedes the instrument's listing on {listing}",
                        instrument_id=record.instrument_id,
                        session_date=record.ex_date,
                    )
                )

    return DataQualityReport(checked_rows=len(records), issues=tuple(issues))


class InstrumentLookup:
    """Minimal listing-date lookup used by corporate-action validation, so the
    validator does not need a full repository.
    """

    def __init__(self, instruments: Iterable[Instrument]) -> None:
        self._listed_from: dict[str, dt.date] = {}
        for instrument in instruments:
            current = self._listed_from.get(instrument.instrument_id)
            if current is None or instrument.effective_from < current:
                self._listed_from[instrument.instrument_id] = instrument.effective_from

    def listed_from(self, instrument_id: str) -> dt.date | None:
        return self._listed_from.get(instrument_id)


def _first_instrument(bars: Sequence[DailyBar]) -> str | None:
    return bars[0].instrument_id if bars else None


def _coverage_issue(
    error: CalendarCoverageError,
    instrument_id: str | None,
    session_date: dt.date | None,
) -> DataQualityIssue:
    """Report an uncheckable date instead of propagating the error.

    Validation's contract is to return a report, so one pass surfaces every
    problem in a file. A date the calendar cannot cover is honestly "not
    checked" -- a warning -- rather than either a silent pass or an exception
    that hides the other issues in the same file.
    """
    return DataQualityIssue(
        code=IssueCode.CALENDAR_COVERAGE_UNKNOWN,
        severity=Severity.WARNING,
        detail=f"calendar could not verify this date: {error}",
        instrument_id=instrument_id,
        session_date=session_date,
    )
