"""Parsing NSE's corporate-actions feed.

The feed states terms as free text, so this is a grammar for prose that a
different team writes and occasionally misspells. Every string below was
taken verbatim from the live feed over 2019/2022/2024 (7,324 records) --
none is invented, because inventing the input is how a parser comes to
handle a format the source does not actually use.

The governing rule is that **an unrecognised line is reported, never
skipped**. A dropped split is indistinguishable from a 50% overnight
crash: it reads as momentum and drawdown that never happened, on a stock
that did nothing, and it corrupts every factor for that name across the
whole lookback. There is no test on the price series alone that can see
it, which is exactly why the parser must refuse rather than guess.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from data.models import CorporateActionType
from data.nse_corporate_actions import classify_subject, to_corporate_actions


def _record(subject: str, *, symbol: str = "ACME", ex: str = "16-Sep-2026") -> dict[str, str]:
    return {"symbol": symbol, "subject": subject, "series": "EQ", "exDate": ex, "recDate": "-"}


# ---------------------------------------------------------------------------
# Splits: the ratio comes from face values
# ---------------------------------------------------------------------------


def test_a_face_value_split_becomes_a_share_ratio() -> None:
    """A Rs 10 share becoming a Rs 2 share means one share becomes five.

    The face values are kept as the ratio rather than reduced to 5:1, so
    the stored record still shows the numbers the circular used and can be
    checked against it by eye.
    """
    parsed = classify_subject(
        "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
    )
    assert parsed is not None
    assert parsed.action_type is CorporateActionType.SPLIT
    assert parsed.ratio_new == Decimal("10")
    assert parsed.ratio_old == Decimal("2")
    assert parsed.ratio_new / parsed.ratio_old == Decimal("5")


def test_a_split_that_does_not_reduce_face_value_is_refused() -> None:
    """Face value going up is a consolidation, not a split. Parsing it as
    one would invert the adjustment -- the worst possible outcome, since
    the price series would move further from the truth, not closer."""
    assert (
        classify_subject(
            "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Rs 10/- Per Share"
        )
        is None
    )


# ---------------------------------------------------------------------------
# Bonuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("subject", "new", "old"),
    [
        ("Bonus 1:4", 1, 4),
        ("Bonus 1:1", 1, 1),
        ("Bonus- 1:2", 1, 2),  # NSE publishes this spelling too
    ],
)
def test_bonus_ratios_are_parsed(subject: str, new: int, old: int) -> None:
    parsed = classify_subject(subject)
    assert parsed is not None
    assert parsed.action_type is CorporateActionType.BONUS
    assert parsed.ratio_new == Decimal(new)
    assert parsed.ratio_old == Decimal(old)


def test_a_bonus_of_debentures_is_not_treated_as_a_share_bonus() -> None:
    """Verbatim from the feed (BRITANNIA, 2019).

    One debenture per equity share held does not change the share count,
    so there is no share ratio to compute. Refusing means the instrument
    is flagged for review rather than silently given a wrong adjustment.
    """
    assert (
        classify_subject("Scheme Of Arangement- Bonus - 1 Debenture For 1 Equity Share Held")
        is None
    )


# ---------------------------------------------------------------------------
# Dividends: the currency is inflected, and the feed is inconsistent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("subject", "amount"),
    [
        ("Dividend - Rs 10 Per Share", "10"),
        ("Interim Dividend - Re 1 Per Share", "1"),      # "Re" is singular rupee
        ("Interim Dividend Rs - 11 Per Share", "11"),    # dash after the unit
        ("Dividend Rs -2 Per Share", "2"),
        ("Int Div - Rs 0.71 Per Sh", "0.71"),            # abbreviated
        ("Interim Divdend - Rs 2 Per Share", "2"),       # NSE's own typo
        ("Dividend - Rs. 8 Per Share", "8"),
    ],
)
def test_dividend_amounts_are_parsed_across_the_feeds_spellings(
    subject: str, amount: str
) -> None:
    """"Re 1" and "Rs 2" are the same unit inflected for number. Matching
    only "Rs" left 788 of 7,324 records unparsed."""
    parsed = classify_subject(subject)
    assert parsed is not None
    assert parsed.action_type is CorporateActionType.DIVIDEND
    assert parsed.cash_amount == Decimal(amount)


def test_a_dividend_with_no_amount_is_reported_not_invented() -> None:
    """The feed really does publish a bare "Interim Dividend". Guessing an
    amount would corrupt cash accounting silently; reporting it costs a
    line in a report."""
    assert classify_subject("Interim Dividend") is None


# ---------------------------------------------------------------------------
# Events with no computable terms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("subject", "action_type"),
    [
        ("Demerger", CorporateActionType.DEMERGER),
        ("Scheme Of Amalgamation", CorporateActionType.MERGER),
        ("Composite Scheme Of Arrangement", CorporateActionType.MERGER),
        ("Rights 588:1000@ Premium Rs 2/-", CorporateActionType.RIGHTS),
        ("Capital Reduction", CorporateActionType.DEMERGER),
    ],
)
def test_events_without_terms_are_flagged_for_an_operator(
    subject: str, action_type: CorporateActionType
) -> None:
    """These move the price and the feed does not say by how much -- a
    demerger is literally the word "Demerger".

    They must never be recorded as though no adjustment were needed. A
    demerger with no adjustment is a price cliff the strategy reads as a
    crash, and acts on.
    """
    parsed = classify_subject(subject)
    assert parsed is not None
    assert parsed.action_type is action_type
    assert parsed.needs_explicit_factor is True


def test_a_rights_ratio_alone_does_not_produce_a_factor() -> None:
    """The ratio is present, which makes it tempting. But a rights factor
    also needs the issue price against the cum-rights price, and the feed
    quotes a premium, which is not the issue price."""
    parsed = classify_subject("Rights 21:100 @ Premium Rs 76 Per Share")
    assert parsed is not None
    assert parsed.ratio_new == Decimal("21")
    assert parsed.ratio_old == Decimal("100")
    assert parsed.needs_explicit_factor is True


@pytest.mark.parametrize(
    "subject",
    ["Buy Back", "Extra Ordinary General Meeting", "Annual General Meeting"],
)
def test_non_price_events_are_recognised_and_ignored(subject: str) -> None:
    """Recognised-and-ignored is a different outcome from unrecognised, and
    keeping them distinct is what makes the unparsed count meaningful."""
    parsed = classify_subject(subject)
    assert parsed is not None
    assert parsed.action_type is None


# ---------------------------------------------------------------------------
# Conversion to domain records
# ---------------------------------------------------------------------------


def test_records_convert_with_exchange_qualified_ids() -> None:
    result = to_corporate_actions([_record("Bonus 1:1", symbol="WIPRO")])
    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.instrument_id == "NSE:WIPRO"
    assert action.ex_date == dt.date(2026, 9, 16)


def test_non_equity_series_is_excluded() -> None:
    """The feed also carries debentures and other series whose actions do
    not apply to the cash-equity instrument this system trades."""
    record = _record("Bonus 1:1")
    record["series"] = "N1"
    assert to_corporate_actions([record]).actions == ()


def test_an_unusable_ex_date_is_reported_rather_than_dropped() -> None:
    """Without an ex-date an action cannot be placed on a timeline, so it
    cannot adjust anything -- but it still happened, and pretending it did
    not is how a price cliff goes unexplained."""
    result = to_corporate_actions([_record("Bonus 1:1", ex="-")])
    assert result.actions == ()
    assert len(result.unparsed) == 1
    assert "exDate" in result.unparsed[0][1]


def test_unparsed_subjects_are_returned_not_skipped() -> None:
    result = to_corporate_actions([_record("Something Entirely New", symbol="ACME")])
    assert result.actions == ()
    assert result.unparsed == (("ACME", "Something Entirely New"),)


def test_instruments_needing_review_covers_both_kinds_of_uncertainty() -> None:
    """docs/SPECIFICATION.md section 2.1's "explicit exclusion list for
    instruments with corporate-action anomalies".

    Two ways to land on it: an action whose factor nobody can compute, and
    a line nobody could parse. Both mean the same thing operationally --
    this instrument cannot be priced through its own event, so it must be
    excluded from the universe for any window spanning the ex-date rather
    than traded on prices that are wrong.
    """
    result = to_corporate_actions(
        [
            _record("Demerger", symbol="SPLITCO"),
            _record("Total Gibberish", symbol="WEIRDCO"),
            _record("Bonus 1:1", symbol="FINECO"),
        ]
    )
    review = result.instruments_needing_review()
    assert "NSE:SPLITCO" in review
    assert "WEIRDCO" in review
    assert "NSE:FINECO" not in review, "a fully-parsed bonus needs no review"


def test_a_parsed_action_never_carries_an_invented_price_factor() -> None:
    """``explicit_price_factor`` is for a value a human supplied. Nothing
    in this module may populate it, because nothing in the feed contains
    it."""
    result = to_corporate_actions(
        [_record("Demerger"), _record("Bonus 1:1"), _record("Rights 21:100 @ Premium Rs 76")]
    )
    assert all(action.explicit_price_factor is None for action in result.actions)


def test_a_malformed_amount_does_not_abort_the_ingest() -> None:
    r"""Regression: an eleven-year backfill died on this.

    The amount pattern was ``[\d.]+``, which matches "." and "1.2.3" as
    happily as "11"; Decimal then raised InvalidOperation and took down a
    run that had already parsed 2,885 sessions of universe data. One
    malformed record out of tens of thousands must be reported, not fatal.
    """
    for subject in ("Dividend - Rs . Per Share", "Dividend - Rs 1.2.3 Per Share"):
        result = to_corporate_actions([_record(subject)])
        assert result.actions == (), subject
        assert len(result.unparsed) == 1, subject
