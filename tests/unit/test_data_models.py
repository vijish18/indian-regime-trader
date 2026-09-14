"""Model-level invariants: what the data layer refuses to represent at all,
and what it deliberately allows so the validator can report it.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from data.models import (
    CorporateAction,
    CorporateActionType,
    DailyBar,
    Exchange,
    IndexMembership,
    IndexObservation,
    Instrument,
    PriceBasis,
    Quote,
    Segment,
    SessionType,
    TradingSession,
)
from tests.conftest import BarFactory

IST = ZoneInfo("Asia/Kolkata")


# --------------------------------------------------------------------------
# Instrument metadata
# --------------------------------------------------------------------------


def test_instrument_carries_full_metadata(infosys: Instrument) -> None:
    assert infosys.exchange is Exchange.NSE
    assert infosys.segment is Segment.EQUITY
    assert infosys.isin == "INE009A01021"
    assert infosys.tick_size == Decimal("0.05")
    assert infosys.lot_size == 1
    assert infosys.price_precision == 2
    assert infosys.effective_to is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tick_size", Decimal("0")),
        ("tick_size", Decimal("-0.05")),
        ("price_precision", -1),
        ("lot_size", 0),
        ("instrument_id", ""),
        ("symbol", ""),
    ],
)
def test_instrument_rejects_impossible_metadata(
    infosys: Instrument, field: str, value: object
) -> None:
    payload = {
        "instrument_id": infosys.instrument_id,
        "symbol": infosys.symbol,
        "exchange": infosys.exchange,
        "segment": infosys.segment,
        "tick_size": infosys.tick_size,
        "price_precision": infosys.price_precision,
        "effective_from": infosys.effective_from,
        "lot_size": infosys.lot_size,
        field: value,
    }
    with pytest.raises(ValueError):
        Instrument(**payload)  # type: ignore[arg-type]


def test_instrument_rejects_inverted_effective_dates() -> None:
    """An inverted effective range is an invalid date pair, not bad vendor
    data: no lookup could ever resolve it.
    """
    with pytest.raises(ValueError, match="precedes"):
        Instrument(
            instrument_id="NSE:X",
            symbol="X",
            exchange=Exchange.NSE,
            segment=Segment.EQUITY,
            tick_size=Decimal("0.05"),
            price_precision=2,
            effective_from=dt.date(2024, 6, 1),
            effective_to=dt.date(2024, 1, 1),
        )


def test_instrument_effectiveness_is_inclusive_of_both_bounds() -> None:
    instrument = Instrument(
        instrument_id="NSE:X",
        symbol="X",
        exchange=Exchange.NSE,
        segment=Segment.EQUITY,
        tick_size=Decimal("0.05"),
        price_precision=2,
        effective_from=dt.date(2024, 1, 1),
        effective_to=dt.date(2024, 12, 31),
    )
    assert not instrument.is_effective_on(dt.date(2023, 12, 31))
    assert instrument.is_effective_on(dt.date(2024, 1, 1))
    assert instrument.is_effective_on(dt.date(2024, 12, 31))
    assert not instrument.is_effective_on(dt.date(2025, 1, 1))


def test_delisted_instrument_is_effective_but_not_tradable(infosys: Instrument) -> None:
    from data.models import InstrumentStatus

    delisted = Instrument(
        instrument_id=infosys.instrument_id,
        symbol=infosys.symbol,
        exchange=infosys.exchange,
        segment=infosys.segment,
        tick_size=infosys.tick_size,
        price_precision=infosys.price_precision,
        effective_from=dt.date(2024, 1, 1),
        status=InstrumentStatus.DELISTED,
    )
    assert delisted.is_effective_on(dt.date(2024, 6, 1))
    assert not delisted.is_tradable_on(dt.date(2024, 6, 1))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("100.02", "100.00"), ("100.03", "100.05"), ("100.07", "100.05"), ("100.00", "100.00")],
)
def test_round_to_tick_uses_exact_decimal_arithmetic(
    infosys: Instrument, raw: str, expected: str
) -> None:
    assert infosys.round_to_tick(Decimal(raw)) == Decimal(expected)


# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------


def test_bar_allows_impossible_ohlc_but_reports_it() -> None:
    """Bad vendor data must be representable, or the quality layer can never
    tell anyone about it.
    """
    bar = DailyBar(
        instrument_id="NSE:INFY",
        session_date=dt.date(2024, 1, 2),
        open=Decimal("100"),
        high=Decimal("95"),  # high below low
        low=Decimal("99"),
        close=Decimal("98"),
        volume=1_000,
    )
    assert not bar.has_valid_ohlc()


def test_bar_recognizes_valid_ohlc(bar_factory: BarFactory) -> None:
    bar = bar_factory(
        dt.date(2024, 1, 2), close="101", open_="100", high="102", low="99"
    )
    assert bar.has_valid_ohlc()
    assert bar.has_positive_prices()


def test_bar_adjustment_scales_prices_and_preserves_traded_value(
    bar_factory: BarFactory,
) -> None:
    bar = bar_factory(
        dt.date(2024, 1, 2), close="500", open_="500", high="500", low="500", volume=1_000
    )
    adjusted = bar.adjusted(Decimal("0.2"))  # 5-for-1 split
    assert adjusted.close == Decimal("100.0000")
    assert adjusted.volume == 5_000
    assert adjusted.price_basis is PriceBasis.ADJUSTED


def test_bar_adjustment_rejects_non_positive_factor(bar_factory: BarFactory) -> None:
    with pytest.raises(ValueError, match="positive"):
        bar_factory(dt.date(2024, 1, 2)).adjusted(Decimal("0"))


# --------------------------------------------------------------------------
# Timezone handling
# --------------------------------------------------------------------------


def test_quote_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Quote(
            instrument_id="NSE:INFY",
            bid=Decimal("100"),
            ask=Decimal("100.05"),
            last_price=Decimal("100"),
            as_of=dt.datetime(2024, 1, 2, 10, 0),  # naive
        )


def test_quote_staleness_is_measured_against_aware_now() -> None:
    quote = Quote(
        instrument_id="NSE:INFY",
        bid=Decimal("100"),
        ask=Decimal("100.10"),
        last_price=Decimal("100.05"),
        as_of=dt.datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    assert not quote.is_stale(dt.datetime(2024, 1, 2, 10, 0, 10, tzinfo=IST), 15)
    assert quote.is_stale(dt.datetime(2024, 1, 2, 10, 0, 30, tzinfo=IST), 15)
    with pytest.raises(ValueError, match="timezone-aware"):
        quote.is_stale(dt.datetime(2024, 1, 2, 10, 0, 30), 15)


def test_quote_staleness_is_correct_across_timezones() -> None:
    """The same instant expressed in UTC and IST must give the same age --
    this is exactly the comparison a naive datetime would get wrong by 5h30m.
    """
    quote = Quote(
        instrument_id="NSE:INFY",
        bid=Decimal("100"),
        ask=Decimal("100.10"),
        last_price=Decimal("100.05"),
        as_of=dt.datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    same_instant_utc = dt.datetime(2024, 1, 2, 4, 30, tzinfo=dt.UTC)
    assert quote.age_seconds(same_instant_utc) == 0
    assert not quote.is_stale(same_instant_utc, 1)


def test_quote_spread_and_crossed_book() -> None:
    quote = Quote(
        instrument_id="NSE:INFY",
        bid=Decimal("100.00"),
        ask=Decimal("100.10"),
        last_price=Decimal("100.05"),
        as_of=dt.datetime(2024, 1, 2, 10, 0, tzinfo=IST),
    )
    assert quote.mid == Decimal("100.05")
    # 0.10 / 100.05 * 10_000 = 9.9950...
    assert quote.spread_bps.quantize(Decimal("0.01")) == Decimal("10.00")
    assert not quote.is_crossed()


def test_trading_session_rejects_naive_boundaries() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        TradingSession(
            session_date=dt.date(2024, 1, 2),
            session_type=SessionType.REGULAR,
            regular_open=dt.datetime(2024, 1, 2, 9, 15),  # naive
            regular_close=dt.datetime(2024, 1, 2, 15, 30, tzinfo=IST),
        )


def test_trading_session_phases() -> None:
    session = TradingSession(
        session_date=dt.date(2024, 1, 2),
        session_type=SessionType.REGULAR,
        regular_open=dt.datetime(2024, 1, 2, 9, 15, tzinfo=IST),
        regular_close=dt.datetime(2024, 1, 2, 15, 30, tzinfo=IST),
        pre_open_start=dt.datetime(2024, 1, 2, 9, 0, tzinfo=IST),
        pre_open_end=dt.datetime(2024, 1, 2, 9, 8, tzinfo=IST),
    )
    assert session.is_trading_day
    assert session.type_at(dt.datetime(2024, 1, 2, 9, 2, tzinfo=IST)) is SessionType.PRE_OPEN
    assert session.type_at(dt.datetime(2024, 1, 2, 10, 0, tzinfo=IST)) is SessionType.REGULAR
    assert session.type_at(dt.datetime(2024, 1, 2, 16, 0, tzinfo=IST)) is SessionType.CLOSED
    # 09:30 UTC is 15:00 IST -- inside the session despite the earlier clock time.
    assert session.type_at(dt.datetime(2024, 1, 2, 9, 30, tzinfo=dt.UTC)) is SessionType.REGULAR


def test_trading_session_rejects_close_before_open() -> None:
    with pytest.raises(ValueError, match="does not follow open"):
        TradingSession(
            session_date=dt.date(2024, 1, 2),
            session_type=SessionType.REGULAR,
            regular_open=dt.datetime(2024, 1, 2, 15, 30, tzinfo=IST),
            regular_close=dt.datetime(2024, 1, 2, 9, 15, tzinfo=IST),
        )


# --------------------------------------------------------------------------
# Corporate actions
# --------------------------------------------------------------------------


def test_split_factor() -> None:
    action = CorporateAction(
        instrument_id="NSE:INFY",
        action_type=CorporateActionType.SPLIT,
        ex_date=dt.date(2024, 5, 2),
        ratio_new=Decimal(5),
        ratio_old=Decimal(1),
    )
    assert action.price_adjustment_factor() == Decimal("0.2")


def test_bonus_factor_differs_from_split_with_the_same_ratio() -> None:
    """A 1:1 bonus halves the price; a 1:1 "split" would not change it. Getting
    these two the same way round is the classic corporate-action bug.
    """
    bonus = CorporateAction(
        instrument_id="NSE:INFY",
        action_type=CorporateActionType.BONUS,
        ex_date=dt.date(2024, 5, 2),
        ratio_new=Decimal(1),
        ratio_old=Decimal(1),
    )
    assert bonus.price_adjustment_factor() == Decimal("0.5")


def test_dividend_does_not_adjust_price() -> None:
    dividend = CorporateAction(
        instrument_id="NSE:INFY",
        action_type=CorporateActionType.DIVIDEND,
        ex_date=dt.date(2024, 5, 2),
        cash_amount=Decimal("18.00"),
    )
    assert dividend.price_adjustment_factor() == Decimal(1)


def test_rights_requires_an_explicit_factor() -> None:
    rights = CorporateAction(
        instrument_id="NSE:INFY",
        action_type=CorporateActionType.RIGHTS,
        ex_date=dt.date(2024, 5, 2),
        ratio_new=Decimal(1),
        ratio_old=Decimal(5),
    )
    with pytest.raises(ValueError, match="explicit_price_factor"):
        rights.price_adjustment_factor()

    priced = CorporateAction(
        instrument_id="NSE:INFY",
        action_type=CorporateActionType.RIGHTS,
        ex_date=dt.date(2024, 5, 2),
        explicit_price_factor=Decimal("0.96"),
    )
    assert priced.price_adjustment_factor() == Decimal("0.96")


def test_action_rejects_announcement_after_ex_date() -> None:
    """An action announced after it went ex is an impossible date ordering and
    usually means the two columns are swapped in the source file.
    """
    with pytest.raises(ValueError, match="announcement_date"):
        CorporateAction(
            instrument_id="NSE:INFY",
            action_type=CorporateActionType.SPLIT,
            ex_date=dt.date(2024, 5, 2),
            ratio_new=Decimal(5),
            ratio_old=Decimal(1),
            announcement_date=dt.date(2024, 6, 1),
        )


@pytest.mark.parametrize(
    "action_type",
    [CorporateActionType.MERGER, CorporateActionType.DEMERGER, CorporateActionType.DELISTING],
)
def test_identity_changing_actions_are_flagged(action_type: CorporateActionType) -> None:
    action = CorporateAction(
        instrument_id="NSE:X",
        action_type=action_type,
        ex_date=dt.date(2024, 5, 2),
        explicit_price_factor=Decimal(1),
    )
    assert action.is_identity_change


# --------------------------------------------------------------------------
# Index observations and membership
# --------------------------------------------------------------------------


def test_index_observation_allows_close_only() -> None:
    """India VIX arrives as a level with no OHLC from some vendors."""
    observation = IndexObservation(
        index_symbol="INDIAVIX",
        session_date=dt.date(2024, 1, 2),
        close=Decimal("13.45"),
    )
    assert observation.open is None
    assert observation.has_valid_ohlc()


def test_index_observation_detects_inconsistent_ohlc() -> None:
    observation = IndexObservation(
        index_symbol="NIFTY50",
        session_date=dt.date(2024, 1, 2),
        close=Decimal("21000"),
        open=Decimal("21500"),
        high=Decimal("21200"),
        low=Decimal("20900"),
    )
    assert not observation.has_valid_ohlc()


def test_membership_is_point_in_time() -> None:
    membership = IndexMembership(
        index_symbol="NIFTY50",
        instrument_id="NSE:INFY",
        effective_from=dt.date(2020, 1, 1),
        effective_to=dt.date(2022, 12, 31),
    )
    assert not membership.is_member_on(dt.date(2019, 12, 31))
    assert membership.is_member_on(dt.date(2021, 6, 1))
    assert not membership.is_member_on(dt.date(2023, 1, 1))


def test_membership_rejects_inverted_dates() -> None:
    with pytest.raises(ValueError, match="precedes"):
        IndexMembership(
            index_symbol="NIFTY50",
            instrument_id="NSE:INFY",
            effective_from=dt.date(2022, 1, 1),
            effective_to=dt.date(2021, 1, 1),
        )
