"""Per-position stop: exit a holding that falls far enough from the day's open.

Until now nothing limited a single name's loss. The circuit breakers act on
the *book* -- daily loss, rolling loss, drawdown -- so one position could
bleed 30% without tripping anything, and a holding only left the portfolio
when it stopped ranking in the top ten, which is usually well after the
damage.

**Measured from the session's open, not the entry price.** A stop from entry
drifts with the position: a name up 40% needs a 43% fall to trigger, a name
just bought needs 3%. From the open it is the same question every day --
"has this broken down today" -- which is what a hard stop is for.

**What a daily bar can and cannot tell you.** ``low <= open * (1 - pct)``
proves the level traded, so the breach is a fact, not an estimate. What the
bar cannot give is the fill: the stop might have been hit on the way down
and filled at the level, or the stock might have gapped straight through.
``stop_fill_price`` takes the worse of the stop level and the session's
close, which is pessimistic by construction -- a backtest that assumes every
stop filled exactly at its trigger reports protection the market does not
sell.

Nothing here decides position size or ranking. It answers one question about
one holding on one day.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StopBreach:
    """One holding that traded through its stop level during a session."""

    instrument_id: str
    open_price: float
    stop_price: float
    low: float
    fill_price: float

    @property
    def loss_pct_from_open(self) -> float:
        return (self.fill_price / self.open_price) - 1.0 if self.open_price else 0.0


def stop_price(open_price: float, stop_pct: float) -> float:
    """The level below which the position is exited."""
    if open_price <= 0:
        raise ValueError(f"open_price must be positive, got {open_price}")
    if not 0.0 < stop_pct < 1.0:
        raise ValueError(f"stop_pct must be a fraction in (0, 1), got {stop_pct}")
    return open_price * (1.0 - stop_pct)


def stop_fill_price(level: float, close: float) -> float:
    """What the exit is assumed to fill at.

    The worse of the stop level and the close. A session that gapped below
    the stop never offered the level, so filling there would credit the
    backtest with a price nobody could have got; a session that dipped and
    recovered did offer it. Taking the minimum covers both without needing
    intraday data to tell them apart.
    """
    return min(level, close) if close > 0 else level


def breached(
    instrument_id: str,
    *,
    open_price: float,
    low: float,
    close: float,
    stop_pct: float,
) -> StopBreach | None:
    """Whether this session traded through the stop, and at what price.

    ``low <= level`` is a fact about what traded, so a breach is never
    inferred from a close alone -- a stock that fell 5% intraday and closed
    down 1% did breach, and a close-only test would miss it.
    """
    if open_price <= 0:
        return None
    level = stop_price(open_price, stop_pct)
    if low > level:
        return None
    return StopBreach(
        instrument_id=instrument_id,
        open_price=open_price,
        stop_price=level,
        low=low,
        fill_price=stop_fill_price(level, close),
    )
