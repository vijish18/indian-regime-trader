"""The per-position stop.

Every case here is about one distinction: what a daily bar proves versus
what it only suggests. A breach is provable -- the level traded. A fill is
not, so the assumption is pessimistic and is asserted as such.
"""

from __future__ import annotations

import pytest

from risk.stop_loss import breached, stop_fill_price, stop_price


def test_a_stop_is_a_fixed_distance_below_the_open() -> None:
    assert stop_price(100.0, 0.03) == pytest.approx(97.0)


def test_a_session_that_never_reached_the_level_is_not_a_breach() -> None:
    assert breached("NSE:ACME", open_price=100.0, low=97.5, close=99.0, stop_pct=0.03) is None


def test_an_intraday_dip_breaches_even_if_the_close_recovers() -> None:
    """The reason a close-only test is not enough: a stock that fell 5%
    intraday and closed down 1% did trade through the stop, and a real
    resting order would have been filled."""
    hit = breached("NSE:ACME", open_price=100.0, low=95.0, close=99.0, stop_pct=0.03)

    assert hit is not None
    assert hit.stop_price == pytest.approx(97.0)
    assert hit.fill_price == pytest.approx(97.0)


def test_a_gap_through_the_stop_fills_below_it_not_at_it() -> None:
    """A session that opened and stayed below the stop never offered the
    level. Filling there would credit the backtest with a price nobody
    could have got -- the single most flattering error a stop can make."""
    hit = breached("NSE:ACME", open_price=100.0, low=88.0, close=89.0, stop_pct=0.03)

    assert hit is not None
    assert hit.fill_price == pytest.approx(89.0), "must fill at the close, not the stop"
    assert hit.loss_pct_from_open == pytest.approx(-0.11, abs=1e-9)


def test_the_fill_is_never_better_than_the_stop_level() -> None:
    """A close above the stop cannot improve the exit: the position left at
    the level on the way down."""
    assert stop_fill_price(97.0, 103.0) == pytest.approx(97.0)


def test_exactly_touching_the_level_counts_as_a_breach() -> None:
    """A resting order at the level fills when the level trades."""
    assert breached("NSE:ACME", open_price=100.0, low=97.0, close=98.0, stop_pct=0.03) is not None


def test_a_nonsensical_stop_is_refused() -> None:
    """A stop of 0 exits everything on any tick; a stop of 1.0 or more can
    never trigger. Both are configuration errors, not strategies."""
    for bad in (0.0, 1.0, 1.5, -0.03):
        with pytest.raises(ValueError, match="stop_pct"):
            stop_price(100.0, bad)


def test_a_missing_open_price_is_not_a_breach() -> None:
    """No open means no reference, so no claim either way -- silently
    treating it as a breach would liquidate on a data gap."""
    assert breached("NSE:ACME", open_price=0.0, low=1.0, close=1.0, stop_pct=0.03) is None
