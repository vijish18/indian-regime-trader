"""Applying corporate actions to prices, end to end.

The arithmetic lives in ``CorporateAction.price_adjustment_factor`` and
``InMemoryCorporateActionProvider.cumulative_adjustment_factor``, and the
application in ``LocalMarketDataProvider._adjust``. All three existed
before real data did. These tests join them with a real corporate action
and real prices, which is the first time the chain has been exercised
against something that actually happened.

The case is Nestle India's 10:1 split, ex-date 2024-01-05. The closes
below are the real ones from NSE's bhavcopy. It is a good test precisely
because it is extreme: raw, the series shows a -90.2% overnight move that
never occurred, and any factor computed across it -- momentum, drawdown,
volatility -- is not slightly wrong but nonsense.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from data.corporate_actions import InMemoryCorporateActionProvider
from data.models import CorporateAction, CorporateActionType

NESTLE = "NSE:NESTLEIND"
EX_DATE = dt.date(2024, 1, 5)

# Real closes from NSE's bhavcopy, unadjusted.
REAL_CLOSES: dict[dt.date, Decimal] = {
    dt.date(2024, 1, 3): Decimal("26635.20"),
    dt.date(2024, 1, 4): Decimal("27116.40"),
    dt.date(2024, 1, 5): Decimal("2666.40"),
    dt.date(2024, 1, 8): Decimal("2619.30"),
}


def _split_10_for_1() -> CorporateAction:
    """As ``data.nse_corporate_actions`` records it.

    NSE publishes "Face Value Split (Sub-Division) - From Rs 10/- Per
    Share To Re 1/- Per Share", and the face values are kept as the ratio
    so the record still shows the numbers the circular used.
    """
    return CorporateAction(
        instrument_id=NESTLE,
        action_type=CorporateActionType.SPLIT,
        ex_date=EX_DATE,
        ratio_new=Decimal("10"),
        ratio_old=Decimal("1"),
    )


@pytest.fixture
def provider() -> InMemoryCorporateActionProvider:
    return InMemoryCorporateActionProvider([_split_10_for_1()])


# ---------------------------------------------------------------------------
# The factor
# ---------------------------------------------------------------------------


def test_a_ten_for_one_split_scales_earlier_prices_by_a_tenth() -> None:
    assert _split_10_for_1().price_adjustment_factor() == Decimal("0.1")


def test_a_one_for_one_bonus_halves_earlier_prices() -> None:
    """Bonus ratios are quoted differently from splits and conflating them
    produces a silently wrong factor. One free share per share held means
    twice as many shares, so earlier prices halve."""
    bonus = CorporateAction(
        instrument_id="NSE:ACME",
        action_type=CorporateActionType.BONUS,
        ex_date=EX_DATE,
        ratio_new=Decimal("1"),
        ratio_old=Decimal("1"),
    )
    assert bonus.price_adjustment_factor() == Decimal("0.5")


# ---------------------------------------------------------------------------
# Applied to the real series
# ---------------------------------------------------------------------------


def test_the_raw_series_contains_a_return_that_never_happened() -> None:
    """Establishes what is being fixed, so the fix is not just asserted to
    work but shown to be necessary."""
    before = REAL_CLOSES[dt.date(2024, 1, 4)]
    after = REAL_CLOSES[EX_DATE]
    raw_return = (after - before) / before
    assert raw_return < Decimal("-0.9")


def test_adjustment_turns_that_into_the_move_that_actually_occurred(
    provider: InMemoryCorporateActionProvider,
) -> None:
    as_of = dt.date(2024, 1, 8)
    before = REAL_CLOSES[dt.date(2024, 1, 4)] * provider.cumulative_adjustment_factor(
        NESTLE, dt.date(2024, 1, 4), as_of
    )
    after = REAL_CLOSES[EX_DATE] * provider.cumulative_adjustment_factor(
        NESTLE, EX_DATE, as_of
    )

    assert before == Decimal("2711.640")
    assert after == Decimal("2666.40")

    real_return = (after - before) / before
    assert Decimal("-0.02") < real_return < Decimal("-0.01")


def test_prices_on_and_after_the_ex_date_are_left_alone(
    provider: InMemoryCorporateActionProvider,
) -> None:
    """The split is already reflected in them. Adjusting again would
    divide by ten twice."""
    as_of = dt.date(2024, 1, 8)
    for day in (EX_DATE, dt.date(2024, 1, 8)):
        assert provider.cumulative_adjustment_factor(NESTLE, day, as_of) == Decimal(1)


# ---------------------------------------------------------------------------
# Point-in-time: the property that makes this safe to backtest on
# ---------------------------------------------------------------------------


def test_an_adjustment_is_invisible_before_the_action_happens(
    provider: InMemoryCorporateActionProvider,
) -> None:
    """The reason factors are applied at read time rather than baked into
    stored prices.

    Asked as of 2024-01-04, the split has not happened yet, so the price
    on 2024-01-03 must still be Rs 26,635 -- which is what a decision made
    on 2024-01-04 actually saw. Baking the factor in would rewrite history
    to a vantage point nobody had.
    """
    as_of_before = dt.date(2024, 1, 4)
    factor = provider.cumulative_adjustment_factor(NESTLE, dt.date(2024, 1, 3), as_of_before)
    assert factor == Decimal(1)

    as_of_after = dt.date(2024, 1, 8)
    factor_after = provider.cumulative_adjustment_factor(
        NESTLE, dt.date(2024, 1, 3), as_of_after
    )
    assert factor_after == Decimal("0.1")


def test_an_adjustment_cannot_be_computed_backwards_in_time(
    provider: InMemoryCorporateActionProvider,
) -> None:
    with pytest.raises(ValueError, match="cannot be computed backwards"):
        provider.cumulative_adjustment_factor(NESTLE, dt.date(2024, 1, 8), dt.date(2024, 1, 3))


def test_successive_actions_compound() -> None:
    """Two splits in a window multiply rather than replace one another. A
    single-action implementation looks right until the second one."""
    provider = InMemoryCorporateActionProvider(
        [
            _split_10_for_1(),
            CorporateAction(
                instrument_id=NESTLE,
                action_type=CorporateActionType.BONUS,
                ex_date=dt.date(2024, 6, 1),
                ratio_new=Decimal("1"),
                ratio_old=Decimal("1"),
            ),
        ]
    )
    factor = provider.cumulative_adjustment_factor(
        NESTLE, dt.date(2024, 1, 3), dt.date(2024, 7, 1)
    )
    assert factor == Decimal("0.05")  # 0.1 x 0.5


# ---------------------------------------------------------------------------
# Actions whose terms nobody has supplied
# ---------------------------------------------------------------------------


def test_an_action_without_terms_refuses_to_produce_a_factor() -> None:
    """Rights, mergers and demergers carry no computable terms in NSE's
    feed. Returning 1.0 for them would silently assert "no adjustment
    needed", which for a demerger is a price cliff the strategy reads as a
    crash and acts on."""
    demerger = CorporateAction(
        instrument_id=NESTLE,
        action_type=CorporateActionType.DEMERGER,
        ex_date=EX_DATE,
    )
    with pytest.raises(ValueError):
        demerger.price_adjustment_factor()


def test_an_operator_supplied_factor_is_used_when_present() -> None:
    """The escape hatch for exactly those actions: a human works out the
    factor from the scheme documents and supplies it."""
    demerger = CorporateAction(
        instrument_id=NESTLE,
        action_type=CorporateActionType.DEMERGER,
        ex_date=EX_DATE,
        explicit_price_factor=Decimal("0.82"),
    )
    assert demerger.price_adjustment_factor() == Decimal("0.82")
