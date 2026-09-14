"""Validation: duplicate bars, missing sessions, bad OHLC, bad reference data,
and corporate-action date handling.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from data.calendar import NSETradingCalendar
from data.data_quality import (
    BarValidator,
    IndexObservationValidator,
    InstrumentLookup,
    IssueCode,
    Severity,
    validate_corporate_actions,
    validate_instruments,
)
from data.errors import DataValidationError
from data.models import (
    CorporateAction,
    CorporateActionType,
    DailyBar,
    Exchange,
    IndexObservation,
    Instrument,
    Segment,
)
from tests.conftest import make_bar


def _instrument(
    instrument_id: str = "NSE:INFY",
    *,
    effective_from: dt.date = dt.date(2020, 1, 1),
    effective_to: dt.date | None = None,
    isin: str | None = "INE009A01021",
) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        symbol=instrument_id.split(":")[-1],
        exchange=Exchange.NSE,
        segment=Segment.EQUITY,
        tick_size=Decimal("0.05"),
        price_precision=2,
        effective_from=effective_from,
        effective_to=effective_to,
        isin=isin,
    )


# --------------------------------------------------------------------------
# Duplicate bars
# --------------------------------------------------------------------------


def test_duplicate_bars_are_an_error() -> None:
    bars = [
        make_bar(dt.date(2024, 1, 2), "100"),
        make_bar(dt.date(2024, 1, 2), "101"),
        make_bar(dt.date(2024, 1, 3), "102"),
    ]
    report = BarValidator().validate(bars)
    assert IssueCode.DUPLICATE_BAR in report.codes()
    assert not report.is_usable
    duplicate = next(i for i in report.issues if i.code is IssueCode.DUPLICATE_BAR)
    assert duplicate.session_date == dt.date(2024, 1, 2)
    assert duplicate.severity is Severity.ERROR


def test_clean_series_produces_no_issues(calendar: NSETradingCalendar) -> None:
    bars = [
        make_bar(dt.date(2024, 1, 2), "100", high="101", low="99", open_="99.5"),
        make_bar(dt.date(2024, 1, 3), "102", high="103", low="100", open_="100.5"),
        make_bar(dt.date(2024, 1, 4), "101", high="103", low="100", open_="102"),
    ]
    report = BarValidator(calendar).validate(bars)
    assert report.is_clean
    assert report.checked_rows == 3


# --------------------------------------------------------------------------
# Missing sessions and calendar alignment
# --------------------------------------------------------------------------


def test_missing_session_is_detected_against_the_calendar(
    calendar: NSETradingCalendar,
) -> None:
    """Jan 3rd is a Wednesday the exchange was open, so its absence is a gap,
    not a holiday.
    """
    bars = [make_bar(dt.date(2024, 1, 2)), make_bar(dt.date(2024, 1, 4))]
    report = BarValidator(calendar).validate(bars)
    missing = [i for i in report.issues if i.code is IssueCode.MISSING_SESSION]
    assert [issue.session_date for issue in missing] == [dt.date(2024, 1, 3)]


def test_weekend_gaps_are_not_missing_sessions(calendar: NSETradingCalendar) -> None:
    bars = [make_bar(dt.date(2024, 1, 5)), make_bar(dt.date(2024, 1, 8))]  # Fri, Mon
    report = BarValidator(calendar).validate(bars)
    assert IssueCode.MISSING_SESSION not in report.codes()


def test_expected_window_detects_a_missing_tail(calendar: NSETradingCalendar) -> None:
    """Without an expected window, an absent tail is invisible: the series
    looks complete because it ends where it ends.
    """
    bars = [make_bar(dt.date(2024, 1, 2)), make_bar(dt.date(2024, 1, 3))]
    report = BarValidator(calendar).validate(
        bars, expected_start=dt.date(2024, 1, 2), expected_end=dt.date(2024, 1, 5)
    )
    missing = {i.session_date for i in report.issues if i.code is IssueCode.MISSING_SESSION}
    assert missing == {dt.date(2024, 1, 4), dt.date(2024, 1, 5)}


def test_bar_on_a_closed_date_is_an_error(calendar: NSETradingCalendar) -> None:
    bars = [make_bar(dt.date(2024, 1, 26))]  # Republic Day
    report = BarValidator(calendar).validate(bars)
    assert IssueCode.NON_TRADING_DATE in report.codes()


def test_bar_on_a_weekend_is_an_error(calendar: NSETradingCalendar) -> None:
    bars = [make_bar(dt.date(2024, 1, 6))]  # Saturday
    report = BarValidator(calendar).validate(bars)
    assert IssueCode.NON_TRADING_DATE in report.codes()


def test_special_session_bar_is_accepted(calendar: NSETradingCalendar) -> None:
    bars = [make_bar(dt.date(2024, 11, 3))]  # Muhurat, a Sunday
    report = BarValidator(calendar).validate(bars)
    assert IssueCode.NON_TRADING_DATE not in report.codes()


def test_uncheckable_date_is_reported_not_raised(calendar: NSETradingCalendar) -> None:
    """Validation must surface every problem in one pass, so a date outside
    calendar coverage becomes a warning rather than an exception that hides
    the rest of the file's issues.
    """
    bars = [make_bar(dt.date(2019, 5, 2))]  # calendar covers 2024 only
    report = BarValidator(calendar).validate(bars)
    assert IssueCode.CALENDAR_COVERAGE_UNKNOWN in report.codes()
    assert report.is_usable  # warning: not checked, not known-bad


def test_uncheckable_date_still_reports_other_problems(
    calendar: NSETradingCalendar,
) -> None:
    bars = [
        make_bar(dt.date(2019, 5, 2), "101", open_="100", high="99", low="98"),
    ]
    report = BarValidator(calendar).validate(bars)
    assert IssueCode.CALENDAR_COVERAGE_UNKNOWN in report.codes()
    assert IssueCode.IMPOSSIBLE_OHLC in report.codes()
    assert not report.is_usable


# --------------------------------------------------------------------------
# OHLC relationships and prices
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("open_", "high", "low", "close"),
    [
        ("100", "95", "99", "98"),   # high below low
        ("105", "102", "99", "100"),  # open above high
        ("100", "102", "99", "98"),   # close below low
        ("100", "99", "99", "99"),    # open above high, degenerate range
    ],
)
def test_impossible_ohlc_is_an_error(
    open_: str, high: str, low: str, close: str
) -> None:
    bar = DailyBar(
        instrument_id="NSE:INFY",
        session_date=dt.date(2024, 1, 2),
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=1_000,
    )
    report = BarValidator().validate([bar])
    assert IssueCode.IMPOSSIBLE_OHLC in report.codes()
    assert not report.is_usable


@pytest.mark.parametrize("price", ["0", "-10"])
def test_non_positive_price_is_an_error(price: str) -> None:
    bar = DailyBar(
        instrument_id="NSE:INFY",
        session_date=dt.date(2024, 1, 2),
        open=Decimal(price),
        high=Decimal(price),
        low=Decimal(price),
        close=Decimal(price),
        volume=1_000,
    )
    report = BarValidator().validate([bar])
    assert IssueCode.NON_POSITIVE_PRICE in report.codes()


def test_negative_volume_is_an_error() -> None:
    bar = make_bar(dt.date(2024, 1, 2), volume=-5)
    report = BarValidator().validate([bar])
    assert IssueCode.NEGATIVE_VOLUME in report.codes()


def test_zero_volume_flat_bar_is_only_a_warning() -> None:
    """A genuine no-trade session is plausible and must be distinguishable
    from missing data (docs/SPECIFICATION.md section 4.1).
    """
    bar = make_bar(dt.date(2024, 1, 2), "100", volume=0)
    report = BarValidator().validate([bar])
    assert IssueCode.ZERO_VOLUME_SESSION in report.codes()
    assert report.is_usable  # warning only


def test_zero_volume_with_a_price_move_is_an_error() -> None:
    """A price cannot move without a trade, so this row is corrupt rather than
    a quiet session.
    """
    bar = make_bar(dt.date(2024, 1, 2), "101", open_="100", high="102", low="99", volume=0)
    report = BarValidator().validate([bar])
    assert IssueCode.ZERO_VOLUME_WITH_PRICE_MOVE in report.codes()
    assert not report.is_usable


def test_future_dated_bar_is_an_error() -> None:
    bars = [make_bar(dt.date(2024, 6, 1))]
    report = BarValidator().validate(bars, today=dt.date(2024, 1, 31))
    assert IssueCode.FUTURE_DATE in report.codes()


def test_series_mixing_instruments_is_an_error() -> None:
    bars = [
        make_bar(dt.date(2024, 1, 2), instrument_id="NSE:INFY"),
        make_bar(dt.date(2024, 1, 3), instrument_id="NSE:TCS"),
    ]
    report = BarValidator().validate(bars)
    assert IssueCode.MIXED_INSTRUMENTS in report.codes()


def test_mislabeled_series_is_detected() -> None:
    bars = [make_bar(dt.date(2024, 1, 2), instrument_id="NSE:TCS")]
    report = BarValidator().validate(bars, instrument_id="NSE:INFY")
    assert IssueCode.MIXED_INSTRUMENTS in report.codes()


# --------------------------------------------------------------------------
# Staleness and volume spikes (warnings)
# --------------------------------------------------------------------------


def test_repeated_closes_are_flagged_as_stale() -> None:
    bars = [make_bar(dt.date(2024, 1, day), "100") for day in range(2, 10)]
    report = BarValidator().validate(bars)
    assert IssueCode.STALE_PRICE in report.codes()
    assert report.is_usable  # warning, not error


def test_volume_spike_is_flagged() -> None:
    bars = [make_bar(dt.date(2024, 1, day), "100", volume=10_000) for day in range(2, 10)]
    bars.append(make_bar(dt.date(2024, 1, 10), "100", volume=10_000_000))
    report = BarValidator().validate(bars)
    assert IssueCode.SUSPICIOUS_VOLUME_SPIKE in report.codes()


# --------------------------------------------------------------------------
# Report behavior
# --------------------------------------------------------------------------


def test_report_raises_on_errors_with_context() -> None:
    bars = [make_bar(dt.date(2024, 1, 2)), make_bar(dt.date(2024, 1, 2))]
    report = BarValidator().validate(bars)
    with pytest.raises(DataValidationError, match="NSE:INFY bars"):
        report.raise_if_errors("NSE:INFY bars")


def test_report_does_not_raise_on_warnings_alone() -> None:
    report = BarValidator().validate([make_bar(dt.date(2024, 1, 2), "100", volume=0)])
    report.raise_if_errors("warnings only")  # must not raise


# --------------------------------------------------------------------------
# Instrument master validation
# --------------------------------------------------------------------------


def test_duplicate_instrument_records_are_detected() -> None:
    report = validate_instruments([_instrument(), _instrument()])
    assert IssueCode.DUPLICATE_INSTRUMENT in report.codes()


def test_overlapping_effective_dates_are_detected() -> None:
    report = validate_instruments(
        [
            _instrument(effective_from=dt.date(2020, 1, 1), effective_to=dt.date(2022, 12, 31)),
            _instrument(effective_from=dt.date(2022, 6, 1)),
        ]
    )
    assert IssueCode.OVERLAPPING_EFFECTIVE_DATES in report.codes()


def test_isin_shared_by_two_instruments_is_detected() -> None:
    """One ISIN identifies one security; two instrument ids claiming it means
    the master is wrong and positions could be booked against either.
    """
    report = validate_instruments(
        [
            _instrument("NSE:A", isin="INE111A01011"),
            _instrument("NSE:B", isin="INE111A01011"),
        ]
    )
    assert IssueCode.DUPLICATE_ISIN in report.codes()


def test_distinct_instruments_validate_cleanly() -> None:
    report = validate_instruments(
        [_instrument("NSE:A", isin="INE111A01011"), _instrument("NSE:B", isin="INE222A01012")]
    )
    assert report.is_clean


# --------------------------------------------------------------------------
# Corporate-action validation
# --------------------------------------------------------------------------


def _split(
    ex_date: dt.date, instrument_id: str = "NSE:INFY", **kwargs: object
) -> CorporateAction:
    return CorporateAction(
        instrument_id=instrument_id,
        action_type=CorporateActionType.SPLIT,
        ex_date=ex_date,
        ratio_new=Decimal(5),
        ratio_old=Decimal(1),
        **kwargs,  # type: ignore[arg-type]
    )


def test_duplicate_corporate_actions_are_detected() -> None:
    report = validate_corporate_actions([_split(dt.date(2024, 1, 3)), _split(dt.date(2024, 1, 3))])
    assert IssueCode.DUPLICATE_CORPORATE_ACTION in report.codes()


def test_action_with_underivable_factor_is_an_error() -> None:
    rights = CorporateAction(
        instrument_id="NSE:INFY",
        action_type=CorporateActionType.RIGHTS,
        ex_date=dt.date(2024, 1, 3),
    )
    report = validate_corporate_actions([rights])
    assert IssueCode.UNDERIVABLE_ADJUSTMENT in report.codes()
    assert not report.is_usable


def test_action_on_a_non_trading_date_is_a_warning(calendar: NSETradingCalendar) -> None:
    report = validate_corporate_actions([_split(dt.date(2024, 1, 26))], calendar=calendar)
    assert IssueCode.ACTION_ON_NON_TRADING_DATE in report.codes()
    assert report.is_usable


def test_action_before_listing_is_an_error() -> None:
    """An ex-date before the instrument existed means the action is attached to
    the wrong instrument, which would corrupt every adjusted price before it.
    """
    lookup = InstrumentLookup([_instrument(effective_from=dt.date(2020, 1, 1))])
    report = validate_corporate_actions([_split(dt.date(2019, 5, 1))], instruments=lookup)
    assert IssueCode.ACTION_BEFORE_LISTING in report.codes()


def test_valid_actions_pass(calendar: NSETradingCalendar) -> None:
    lookup = InstrumentLookup([_instrument(effective_from=dt.date(2020, 1, 1))])
    report = validate_corporate_actions(
        [_split(dt.date(2024, 1, 3))], calendar=calendar, instruments=lookup
    )
    assert report.is_clean


# --------------------------------------------------------------------------
# Index observation validation
# --------------------------------------------------------------------------


def test_duplicate_index_observations_are_detected() -> None:
    observations = [
        IndexObservation("NIFTY50", dt.date(2024, 1, 2), Decimal("21000")),
        IndexObservation("NIFTY50", dt.date(2024, 1, 2), Decimal("21010")),
    ]
    report = IndexObservationValidator().validate(observations)
    assert IssueCode.DUPLICATE_BAR in report.codes()


def test_missing_index_session_is_detected(calendar: NSETradingCalendar) -> None:
    observations = [
        IndexObservation("INDIAVIX", dt.date(2024, 1, 2), Decimal("13.4")),
        IndexObservation("INDIAVIX", dt.date(2024, 1, 4), Decimal("13.9")),
    ]
    report = IndexObservationValidator(calendar).validate(observations)
    assert IssueCode.MISSING_SESSION in report.codes()


def test_close_only_vix_series_validates_cleanly(calendar: NSETradingCalendar) -> None:
    observations = [
        IndexObservation("INDIAVIX", dt.date(2024, 1, 2), Decimal("13.4")),
        IndexObservation("INDIAVIX", dt.date(2024, 1, 3), Decimal("13.6")),
    ]
    report = IndexObservationValidator(calendar).validate(observations)
    assert report.is_clean
