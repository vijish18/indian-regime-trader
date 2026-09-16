"""The point-in-time liquid universe derived from bhavcopy.

This is the survivorship-bias control, so the tests that matter most are
the ones asserting what the universe *cannot* see: tomorrow's turnover,
and today's listing status applied to last year.

Run against real 2020 data, the screen behaves the way the market did --
422 eligible names in January, 360 at the end of March as liquidity dried
up in the crash, 488 by the end of June. A static universe would show a
flat count and quietly let the backtest trade names that were untradable
at the time.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from data.nse_bhavcopy import BhavcopyRow
from universe.bhavcopy_universe import (
    DERIVED_INDEX_SYMBOL,
    EligibilityRules,
    build_snapshots,
    snapshots_to_membership,
)

CRORE = Decimal("10000000")


def _row(
    symbol: str,
    day: dt.date,
    *,
    turnover: Decimal = CRORE * 10,
    close: Decimal = Decimal("100"),
    isin: str | None = None,
) -> BhavcopyRow:
    return BhavcopyRow(
        symbol=symbol,
        isin=isin or f"INE{symbol}",
        session_date=day,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000,
        traded_value=turnover,
        trades=10,
    )


def _sessions(count: int, start: dt.date = dt.date(2024, 1, 1)) -> list[dt.date]:
    return [start + dt.timedelta(days=i) for i in range(count)]


def _rules(**kwargs: object) -> EligibilityRules:
    defaults: dict[str, object] = {
        "min_avg_daily_value_inr": CRORE * 5,
        "lookback_sessions": 20,
        "min_sessions_traded": 15,
    }
    defaults.update(kwargs)
    return EligibilityRules(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------


def test_eligibility_never_uses_turnover_from_the_future() -> None:
    """The single most important property here.

    A stock that becomes heavily traded tomorrow must not be eligible
    today. Letting it in is the most direct look-ahead there is, and it
    flatters a backtest enormously: the days a stock is about to become
    heavily traded are rarely uneventful ones.
    """
    days = _sessions(40)
    rows = {}
    for i, day in enumerate(days):
        # Illiquid for the first 20 sessions, then very liquid.
        turnover = CRORE * Decimal(100) if i >= 20 else CRORE * Decimal("0.1")
        rows[day] = (_row("LATE", day, turnover=turnover),)

    snapshots = {s.session_date: s for s in build_snapshots(rows, _rules())}

    # Well before the jump, nothing about the coming liquidity is visible.
    assert "NSE:LATE" not in snapshots[days[10]].eligible
    # On the jump day itself, the trailing window is still nineteen
    # illiquid sessions and one liquid one, so it must not qualify.
    assert "NSE:LATE" not in snapshots[days[20]].eligible
    # Once the window has genuinely filled with liquid sessions, it does.
    assert "NSE:LATE" in snapshots[days[39]].eligible


def test_one_huge_session_does_not_qualify_an_otherwise_illiquid_name() -> None:
    """Why the screen is a median and not a mean.

    Nineteen sessions at Rs 0.1 crore and one at Rs 100 crore average
    Rs 5.1 crore, clearing a Rs 5 crore bar -- while the name is
    impossible to exit on nineteen days out of twenty. That is exactly
    what one block trade in a thin name looks like, and the mean cannot
    tell it apart from steady turnover.

    This test failed against the original mean-based screen, which is how
    the weakness was found.
    """
    days = _sessions(21)
    rows = {}
    for i, day in enumerate(days):
        turnover = CRORE * Decimal(100) if i == 20 else CRORE * Decimal("0.1")
        rows[day] = (_row("BLOCK", day, turnover=turnover),)

    snapshots = build_snapshots(rows, _rules())
    assert "NSE:BLOCK" not in snapshots[-1].eligible

    # And the mean really would have let it through -- the bar is not
    # simply set too high for the fixture.
    window = [CRORE * Decimal("0.1")] * 20 + [CRORE * Decimal(100)]
    assert sum(window[1:], Decimal(0)) / Decimal(20) > _rules().min_avg_daily_value_inr


def test_nothing_is_eligible_before_the_window_has_filled() -> None:
    """Warm-up must exclude, not approximate. Judging liquidity from three
    sessions is guessing, and guessing in the permissive direction puts
    untradable names in the universe."""
    days = _sessions(10)
    rows = {day: (_row("ACME", day),) for day in days}
    snapshots = build_snapshots(rows, _rules(min_sessions_traded=15))
    assert all(len(snapshot) == 0 for snapshot in snapshots)


# ---------------------------------------------------------------------------
# The screen itself
# ---------------------------------------------------------------------------


def test_a_liquid_name_becomes_eligible() -> None:
    days = _sessions(25)
    rows = {day: (_row("LIQUID", day, turnover=CRORE * 50),) for day in days}
    snapshots = build_snapshots(rows, _rules())
    assert "NSE:LIQUID" in snapshots[-1].eligible


def test_an_illiquid_name_never_becomes_eligible() -> None:
    days = _sessions(25)
    rows = {day: (_row("THIN", day, turnover=CRORE // 10),) for day in days}
    snapshots = build_snapshots(rows, _rules())
    assert all(len(snapshot) == 0 for snapshot in snapshots)


def test_a_name_that_trades_rarely_is_excluded_despite_a_high_average() -> None:
    """The case an average alone gets wrong.

    Three enormous block trades in twenty sessions produce a flattering
    mean while the stock is impossible to exit on any given day. That is
    exactly the illiquidity the average hides, which is why sessions
    traded is screened separately.
    """
    days = _sessions(25)
    rows = {}
    for i, day in enumerate(days):
        rows[day] = (_row("BLOCKY", day, turnover=CRORE * 500),) if i % 8 == 0 else ()
    snapshots = build_snapshots(rows, _rules())
    assert all("NSE:BLOCKY" not in snapshot.eligible for snapshot in snapshots)


def test_penny_stocks_are_excluded_on_price() -> None:
    """Tick-sized moves swamp any modelled edge, and the quoted spread is
    a large fraction of the price."""
    days = _sessions(25)
    rows = {
        day: (_row("PENNY", day, turnover=CRORE * 50, close=Decimal("2")),) for day in days
    }
    snapshots = build_snapshots(rows, _rules())
    assert all(len(snapshot) == 0 for snapshot in snapshots)


def test_a_name_that_stops_trading_decays_out_of_the_universe() -> None:
    """A delisted or suspended name must not keep its last average for
    ever. Its trailing window has to keep advancing even on sessions it
    did not trade, or it stays eligible indefinitely."""
    days = _sessions(45)
    rows = {}
    for i, day in enumerate(days):
        rows[day] = (_row("GONE", day, turnover=CRORE * 50),) if i < 25 else ()
    snapshots = {s.session_date: s for s in build_snapshots(rows, _rules())}
    assert "NSE:GONE" in snapshots[days[24]].eligible
    assert "NSE:GONE" not in snapshots[days[44]].eligible


# ---------------------------------------------------------------------------
# Membership spans
# ---------------------------------------------------------------------------


def test_membership_spans_cover_exactly_the_eligible_sessions() -> None:
    days = _sessions(30)
    rows = {day: (_row("ACME", day, turnover=CRORE * 50),) for day in days}
    snapshots = build_snapshots(rows, _rules())
    eligible_days = [s.session_date for s in snapshots if s.eligible]

    memberships = snapshots_to_membership(snapshots)
    assert len(memberships) == 1
    assert memberships[0].effective_from == eligible_days[0]
    assert memberships[0].effective_to is None  # still eligible on the last session
    assert memberships[0].index_symbol == DERIVED_INDEX_SYMBOL


def test_a_name_that_leaves_and_returns_gets_two_spans() -> None:
    """Merging them would assert membership across a gap the data says did
    not exist -- and the gap is usually the interesting part, because it is
    where the name was too illiquid to trade."""
    days = _sessions(80)
    rows = {}
    for i, day in enumerate(days):
        liquid = i < 25 or i >= 55
        rows[day] = (_row("INOUT", day, turnover=CRORE * 50 if liquid else CRORE // 100),)
    snapshots = build_snapshots(rows, _rules())

    spans = [m for m in snapshots_to_membership(snapshots) if m.instrument_id == "NSE:INOUT"]
    assert len(spans) == 2, (
        f"expected two spans, got {[(s.effective_from, s.effective_to) for s in spans]}"
    )
    assert spans[0].effective_to is not None
    assert spans[0].effective_to < spans[1].effective_from


def test_a_closed_span_does_not_claim_membership_after_it_ended() -> None:
    """``effective_to`` is the last eligible session and the range is
    inclusive, so a span must never assert eligibility on a day the screen
    had already failed."""
    days = _sessions(60)
    rows = {}
    for i, day in enumerate(days):
        rows[day] = (_row("FADE", day, turnover=CRORE * 50 if i < 25 else CRORE // 100),)
    snapshots = build_snapshots(rows, _rules())
    span = next(m for m in snapshots_to_membership(snapshots) if m.instrument_id == "NSE:FADE")

    assert span.effective_to is not None
    last_eligible = max(s.session_date for s in snapshots if "NSE:FADE" in s.eligible)
    assert span.effective_to == last_eligible
    assert span.is_member_on(last_eligible) is True
    assert span.is_member_on(last_eligible + dt.timedelta(days=1)) is False


def test_the_derived_universe_is_not_called_nifty50() -> None:
    """It is not NIFTY 50 membership, and naming it that would be a lie
    that survives into every audit record -- someone would eventually
    compare a holding against the real index and find it absent."""
    assert DERIVED_INDEX_SYMBOL != "NIFTY50"
    assert "NIFTY" not in DERIVED_INDEX_SYMBOL


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_incoherent_rules_are_refused() -> None:
    days = _sessions(5)
    rows = {day: (_row("ACME", day),) for day in days}
    with pytest.raises(ValueError, match="cannot exceed"):
        build_snapshots(rows, _rules(lookback_sessions=5, min_sessions_traded=10))
